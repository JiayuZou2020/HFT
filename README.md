# HFT: Lifting Perspective Representations via Hybrid Feature Transformation
This repositary contains Pytorch implementation for [HFT](https://arxiv.org/abs/2204.05068)
![image](https://user-images.githubusercontent.com/77472466/162715638-145897ba-2c35-4734-b6a7-b30048ab80f8.png)
## Introduction
Autonomous driving requires accurate and detailed Bird's Eye View (BEV) semantic segmentation for decision making, which is one of the most challenging tasks for high-level scene perception. Feature transformation from frontal view to BEV is the pivotal technology for BEV semantic segmentation. Existing works can be roughly classified into two categories, i.e., Camera model-Based Feature Transformation (CBFT) and Camera model-Free Feature Transformation (CFFT). In this paper, we empirically analyze the vital differences between CBFT and CFFT. The former transforms features based on the flat-world assumption, which may cause distortion of regions lying above the ground plane. The latter is limited in the segmentation performance due to the absence of geometric priors and time-consuming computation. In order to reap the benefits and avoid the drawbacks of CBFT and CFFT, we propose a novel framework with a Hybrid Feature Transformation module (HFT). Specifically, we decouple the feature maps produced by HFT for estimating the layout of outdoor scenes in BEV. Furthermore, we design a mutual learning scheme to augment hybrid transformation by applying feature mimicking. Notably, extensive experiments demonstrate that with negligible extra overhead, HFT achieves a relative improvement of 13.3% on the Argoverse dataset and 16.8% on the KITTI 3D Object datasets compared to the best-performing existing method.

## Install
To use our code, please install the following dependencies:
* torch==1.9.1
* torchvison==0.10.1
* mmcv-full==1.3.15
* CUDA 9.2+

For more requirements, please see [requirements.txt](https://github.com/JiayuZou2020/HFT/blob/main/HFT/requirements.txt) for details. You can refer to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation) to install the environment correctly.
## Data Preparation
We conduct experiments of nuScenes, Argoverse, Kitti Raw, Kitti Odometry, and Kitti 3D Object datasets. The datasets' structures look like: 
### [nuScenes](https://www.nuscenes.org/download)
```

```
### [Argoverse](https://www.argoverse.org/)
```

```
### [Kitti Raw](http://www.cvlibs.net/datasets/kitti/raw_data.php)
```

```
### [Kitti Odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
```

```
### [Kitti 3D Object](http://www.cvlibs.net/datasets/kitti/eval_3dobject.php)
```

```
## Training
Take Argoverse as an example. To train a semantic segmentation model under a specific configuration, run:
```
cd HFT
python -m torch.distributed.launch --nproc_per_node ${NUM_GPU} --master_port ${PORT} tools/train.py ${CONFIG} --work-dir ${WORK_DIR} --launcher pytorch
```
For instance, to train Argoverse under [this config](https://github.com/JiayuZou2020/HFT/blob/main/HFT/configs/pyva/pyva_swin_argoverse.py), run:
```
cd HFT
python -m torch.distributed.launch --nproc_per_node 4 --master_port 14300 tools/train.py ./configs/pyva/pyva_swin_argoverse.py --work-dir ./models_dir/pyva_swin_argoverse --launcher pytorch
```
## Evaluation
To evaluate the performance, run the following command:
```
cd HFT
python -m torch.distributed.launch --nproc_per_node ${NUM_GPU} --master_port ${PORT} tools/test.py ${CONFIG} ${MODEL_PATH} --out ${SAVE_RESULT_PATH} --eval ${METRIC} --launcher pytorch
```
For example, we evaluate the mIoU on Argoverse under [this config](https://github.com/JiayuZou2020/HFT/blob/main/HFT/configs/pyva/pyva_swin_argoverse.py) by running:
```
cd HFT
python -m torch.distributed.launch --nproc_per_node 4 --master_port 14300 tools/test.py ./configs/pyva/pyva_swin_argoverse.py ./models_dir/pyva_swin_argoverse/iter_20000.pth  --out ./results/pyva/pyva_20k.pkl --eval mIoU --launcher pytorch
```
## Visulization
To get the visulization results of the model, we first change the output_type from 'iou' to 'seg' in the testing process. Take [this config](https://github.com/JiayuZou2020/HFT/blob/main/HFT/configs/pyva/pyva_swin_argoverse.py) as an example.
```
model = dict(
    decode_head=dict(
        type='PyramidHeadArgoverse',
        num_classes=8,
        align_corners=True),
    # change the output_type from 'iou' to 'seg'
    test_cfg=dict(mode='whole',output_type='seg',positive_thred=0.5)
)
```
And then, we can generate the visualization results by running the following command:
```
python -m torch.distributed.launch --nproc_per_node 4 --master_port 14300 tools/test.py ./configs/pyva/pyva_swin_argoverse.py ./models_dir/pyva_swin_argoverse/iter_20000.pth --format-only --eval-options "imgfile_prefix=./models_dir/pyva_swin_argoverse" --launcher pytorch
```
## Acknowledgement
Our work is partially based on [mmseg](https://github.com/open-mmlab/mmsegmentation). Thanks for their contributions to the research community.
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
