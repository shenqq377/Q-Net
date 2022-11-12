# Q-Net
Codes for the following paper:

@article{shen2022qnet,

  title={Q-Net: Query-Informed Few-Shot Medical Image Segmentation},
  
  author={Shen, Qianqian and Li, Yanan and Jin, Jiyong and Liu, Bin},
  
  journal={arXiv preprint arXiv:2208.11451},
  
  year={2022}
  
}

![](https://github.com/ZJLAB-AMMI/Q-Net/blob/main/the_framework.png?raw=true) 

### Abstract
Deep learning has achieved tremendous success in computer vision, while medical image segmentation (MIS) remains a challenge, due to the scarcity of data annotations. Meta-learning techniques for few-shot segmentation (Meta-FSS) have been widely used to tackle this challenge, while they neglect possible distribution shifts between the query image and the support set. In contrast, an experienced clinician can perceive and address such shifts by borrowing information from the query image, then fine-tune or calibrate his (her) prior cognitive model accordingly. Inspired by this, we propose Q-Net, a Query-informed Meta-FSS approach, which mimics in spirit the learning mechanism of an expert clinician. We build Q-Net based on ADNet, a recently proposed anomaly detection-inspired method. Specifically, we add two query-informed computation modules into ADNet, namely a query-informed threshold adaptation module and a query-informed prototype refinement module. Combining them with a dual-path extension of the feature extraction module, Q-Net achieves state-of-the-art performance on two widely used datasets, which are composed of abdominal MR images and cardiac MR images, respectively. Our work sheds light on a novel way to improve Meta-FSS techniques by leveraging query information.  

<!-- Illustration of the proposed method in testing time. We use a shared feature encoder to learn deep feature maps in dual-path, corresponding to two feature scales: $32\times32$ and $64\times64$. The flows of operations are the same for these two paths, while we modularize the operations of the 2nd path and drawn them in gray to save space. In each path, we first learn one foreground prototype $\textbf{p}$ from the support features. Next we compute the similarity map between each query feature vector and the prototype. Then we predict the initial segmentation mask $\tilde{\textbf{m}}^q_{32}$ via anomaly detection performed on the similarity map with threshold $T$, which is learned from query image feature by two fully-connected layers. Then we refine the prototype by repeating the following two operations for a fixed number of times: (1) replacing the foreground feature vectors with the prototype; (2) minimizing a reconstruction loss. In training time, the prototype refinement module is turned off.   -->

### Dependencies
Please install following essential dependencies:
```
dcm2nii
json5==0.8.5
jupyter==1.0.0
nibabel==2.5.1
numpy==1.22.0
opencv-python==4.5.5.62
Pillow>=8.1.1
sacred==0.8.2
scikit-image==0.18.3
SimpleITK==1.2.3
torch==1.10.2
torchvision=0.11.2
tqdm==4.62.3
```

### Datasets and pre-processing
Download:  
1. **Abdominal MRI**  [Combined Healthy Abdominal Organ Segmentation dataset](https://chaos.grand-challenge.org/)  
2. **Cardiac MRI** [Multi-sequence Cardiac MRI Segmentation dataset (bSSFP fold)](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mscmrseg/)  

**Pre-processing** is performed according to [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation.git) and we follow the procedure on their github repository.  
We put the pre-processed images and their corresponding labels in `./data/CHAOST2/chaos_MR_T2_normalized` folder for Abdominal MRI and `./data/CMR/cmr_MR_normalized` folder for Cardiac MRI.  

**Supervoxel segmentation** is performed according to [Hansen et al.](https://github.com/sha168/ADNet.git) and we follow the procedure on their github repository.  
We also put the package `supervoxels` in `./data`, run our modified file `./data./supervoxels/generate_supervoxels.py` to implement pseudolabel generation. The generated supervoxels for `CHAOST2` and `CMR` datasets are put in `./data/CHAOST2/supervoxels_5000` folder and `./data/CMR/supervoxels_1000` folder, respectively.  

### Training  
1. Download pre-trained ResNet-101 weights [vanilla version](https://download.pytorch.org/models/resnet101-63fe2227.pth) or [deeplabv3 version](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) and put your checkpoints folder, then replace the absolute path in the code `./models/encoder.py`.  
2. Run `bash scripts/train_<abd,cmr>_mr.sh`  
#### Note:  
The alpha coefficient for dual-scale features in the code `./models/fewshot.py` should be manually modified.  
For setting 1, the alpha = [0.9, 0.1]  
For setting 2, the alpha = [0.6, 0.4]  

### Testing
Run `bash scripts/val.sh`

### Acknowledgement
This code is based on [SSL-ALPNet](https://arxiv.org/abs/2007.09886v2) (ECCV'20) by [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation.git) and [ADNet](https://www.sciencedirect.com/science/article/pii/S1361841522000378) by [Hansen et al.](https://github.com/sha168/ADNet.git). 
