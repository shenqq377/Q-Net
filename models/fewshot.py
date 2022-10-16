"""
Query-Informed FSS
Extended from ADNet code by Hansen et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .encoder import Res101Encoder


class FewShotSeg(nn.Module):

    def __init__(self, pretrained_weights="deeplabv3", alpha=0.9):
        super().__init__()

        # Encoder
        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)  # or "resnet101"
        self.device = torch.device('cuda')
        self.scaler = 20.0
        self.criterion = nn.NLLLoss()
        self.alpha = torch.Tensor([alpha, 1-alpha])

    def forward(self, supp_imgs, supp_mask, qry_imgs, train=False, n_iters=0, lr=0.01):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """

        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  # B x Wa x Sh x H x W

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        img_fts, tao = self.encoder(imgs_concat)
        supp_fts = [img_fts[dic][:self.n_ways * self.n_shots * supp_bs].view(  # B x Wa x Sh x C x H' x W'
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
        qry_fts = [img_fts[dic][self.n_ways * self.n_shots * supp_bs:].view(  # B x N x C x H' x W'
            qry_bs, self.n_queries, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]

        ###### Get threshold #######
        self.t = tao[self.n_ways * self.n_shots * supp_bs:]  # t for query features
        self.thresh_pred = [self.t for _ in range(self.n_ways)]

        # self.t_ = tao[:self.n_ways * self.n_shots * supp_bs]  # t for support features
        # self.thresh_pred_ = [self.t_ for _ in range(self.n_ways)]

        ###### Compute loss ######
        align_loss = torch.zeros(1).to(self.device)
        outputs = []
        for epi in range(supp_bs):
            ###### Extract prototypes ######
            supp_fts_ = [[[self.getFeatures(supp_fts[n][[epi], way, shot], supp_mask[[epi], way, shot])
                           for shot in range(self.n_shots)] for way in range(self.n_ways)] for n in
                         range(len(supp_fts))]
            fg_prototypes = [self.getPrototype(supp_fts_[n]) for n in range(len(supp_fts))]

            # ###### Get support predictions #######
            # supp_preds = [torch.stack(
            #     [self.getPred(supp_fts[n][epi, way], fg_prototypes[n][way], self.thresh_pred[way]) for way in
            #      range(self.n_ways)],
            #     dim=1) for n in range(len(supp_fts))]  # N x Wa x H' x W'
            # supp_preds = [F.interpolate(supp_preds[n], size=img_size, mode='bilinear', align_corners=True)
            #              for n in range(len(supp_fts))]

            ###### Get query predictions ######
            qry_pred = [torch.stack(
                [self.getPred(qry_fts[n][epi], fg_prototypes[n][way], self.thresh_pred[way])
                 for way in range(self.n_ways)], dim=1) for n in range(len(qry_fts))]  # N x Wa x H' x W'

            ###### Prototype Refinement (only for test) ######
            fg_prototypes_ = []
            if (not train) and n_iters > 0:  # iteratively update prototypes
                for n in range(len(qry_fts)):
                    fg_prototypes_.append(
                        self.updatePrototype(qry_fts[n], fg_prototypes[n], qry_pred[n], n_iters, lr, epi))

                qry_pred = [torch.stack(
                    [self.getPred(qry_fts[n][epi], fg_prototypes_[n][way], self.thresh_pred[way]) for way in
                     range(self.n_ways)], dim=1) for n in range(len(qry_fts))]  # N x Wa x H' x W'

            ###### Combine predictions of different feature maps ######
            qry_pred_up = [F.interpolate(qry_pred[n], size=img_size, mode='bilinear', align_corners=True)
                           for n in range(len(qry_fts))]
            pred = [self.alpha[n] * qry_pred_up[n] for n in range(len(qry_fts))]
            preds = torch.sum(torch.stack(pred, dim=0), dim=0) / torch.sum(self.alpha)
            preds = torch.cat((1.0 - preds, preds), dim=1)
            outputs.append(preds)

            ###### Prototype alignment loss ######
            if train:
                align_loss_epi = self.alignLoss([supp_fts[n][epi] for n in range(len(supp_fts))],
                                                [qry_fts[n][epi] for n in range(len(qry_fts))],
                                                preds, supp_mask[epi])
                align_loss += align_loss_epi

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])

        return output, align_loss / supp_bs

    def updatePrototype(self, fts, prototype, pred, update_iters, lr, epi):

        prototype_0 = torch.stack(prototype, dim=0)
        prototype_ = Parameter(torch.stack(prototype, dim=0))

        optimizer = torch.optim.Adam([prototype_], lr=lr)

        while update_iters > 0:
            with torch.enable_grad():
                pred_mask = torch.sum(pred, dim=-3)
                pred_mask = torch.stack((1.0 - pred_mask, pred_mask), dim=1).argmax(dim=1, keepdim=True)
                pred_mask = pred_mask.repeat([*fts.shape[1:-2], 1, 1])
                bg_fts = fts[epi] * (1 - pred_mask)
                fg_fts = torch.zeros_like(fts[epi])
                for way in range(self.n_ways):
                    fg_fts += prototype_[way].unsqueeze(-1).unsqueeze(-1).repeat(*pred.shape) \
                              * pred_mask[way][None, ...]
                new_fts = bg_fts + fg_fts
                fts_norm = torch.sigmoid((fts[epi] - fts[epi].min()) / (fts[epi].max() - fts[epi].min()))
                new_fts_norm = torch.sigmoid((new_fts - new_fts.min()) / (new_fts.max() - new_fts.min()))
                bce_loss = nn.BCELoss()
                loss = bce_loss(fts_norm, new_fts_norm) # + beta * mse_loss(prototype_, prototype_0)

            optimizer.zero_grad()
            # loss.requires_grad_()
            loss.backward()
            optimizer.step()

            pred = torch.stack([self.getPred(fts[epi], prototype_[way], self.thresh_pred[way])
                                for way in range(self.n_ways)], dim=1)  # N x Wa x H' x W'

            update_iters += -1

        return prototype_

    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))

        return pred

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')

        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        return masked_fts

    def getPrototype(self, fg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  ## concat all fg_fts

        return fg_prototypes

    def alignLoss(self, supp_fts, qry_fts, pred, fore_mask):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                qry_fts_ = [[self.getFeatures(qry_fts[n], pred_mask[way + 1])] for n in range(len(qry_fts))]
                fg_prototypes = [self.getPrototype([qry_fts_[n]]) for n in range(len(supp_fts))]

                # Get predictions
                supp_pred = [self.getPred(supp_fts[n][way, [shot]], fg_prototypes[n][way], self.thresh_pred[way])
                             for n in range(len(supp_fts))]  # N x Wa x H' x W'
                supp_pred = [F.interpolate(supp_pred[n][None, ...], size=fore_mask.shape[-2:], mode='bilinear',
                                           align_corners=True)
                             for n in range(len(supp_fts))]

                # Combine predictions of different feature maps
                preds = [self.alpha[n] * supp_pred[n] for n in range(len(supp_fts))]
                preds = torch.sum(torch.stack(preds, dim=0), dim=0) / torch.sum(self.alpha)
                pred_ups = torch.cat((1.0 - preds, preds), dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss
