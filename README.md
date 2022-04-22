# HFT: Lifting Perspective Representations via Hybrid Feature Transformation
This repositary contains Pytorch implementation for [HFT](https://arxiv.org/abs/2204.05068)
![image](https://user-images.githubusercontent.com/77472466/162715638-145897ba-2c35-4734-b6a7-b30048ab80f8.png)
## Introduction
Autonomous driving requires accurate and detailed Bird's Eye View (BEV) semantic segmentation for decision making, which is one of the most challenging tasks for high-level scene perception. Feature transformation from frontal view to BEV is the pivotal technology for BEV semantic segmentation. Existing works can be roughly classified into two categories, i.e., Camera model-Based Feature Transformation (CBFT) and Camera model-Free Feature Transformation (CFFT). In this paper, we empirically analyze the vital differences between CBFT and CFFT. The former transforms features based on the flat-world assumption, which may cause distortion of regions lying above the ground plane. The latter is limited in the segmentation performance due to the absence of geometric priors and time-consuming computation. In order to reap the benefits and avoid the drawbacks of CBFT and CFFT, we propose a novel framework with a Hybrid Feature Transformation module (HFT). Specifically, we decouple the feature maps produced by HFT for estimating the layout of outdoor scenes in BEV. Furthermore, we design a mutual learning scheme to augment hybrid transformation by applying feature mimicking. Notably, extensive experiments demonstrate that with negligible extra overhead, HFT achieves a relative improvement of 13.3% on the Argoverse dataset and 16.8% on the KITTI 3D Object datasets compared to the best-performing existing method.

## Requirements
For more requirements, please see [requirements.txt](https://github.com/JiayuZou2020/HFT/blob/main/HFT/requirements.txt) for details.
## Install

## Data Preparation
We conduct experiments of nuScenes, Argoverse, Kitti Raw, Kitti Odometry, and Kitti 3D Object datasets. The datasets' structures look like: 
### nuScenes

### Argoverse

### Kitti Raw

### Kitti Odometry

### Kitti 3D Object

## Training

## Evaluation

## Visulization

## Citation
If you find our work useful in your research, please cite our work:
```
@article{zou2022hft,
  title={HFT: Lifting Perspective Representations via Hybrid Feature Transformation},
  author={Zou, Jiayu and Xiao, Junrui and Zhu, Zheng and Huang, Junjie and Huang, Guan and Du, Dalong and Wang, Xingang},
  journal={arXiv preprint arXiv:2204.05068},
  year={2022}
}
```
