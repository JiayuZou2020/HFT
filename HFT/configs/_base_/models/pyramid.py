# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='BEVSegmentor',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        num_outs=5),
    transformer=dict(type='TransformerPyramid'),
    decode_head=dict(
        type='PyramidHead',
        num_classes=14,
        align_corners=True),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
