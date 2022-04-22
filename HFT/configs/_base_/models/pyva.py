# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='origin_pyva_BEVSegmentor',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 1, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='PyvaHead'),
    transformer = dict(type='origin_Pyva_transformer',size = (32,32)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole',output_type='iou',positive_thred=0.5))
