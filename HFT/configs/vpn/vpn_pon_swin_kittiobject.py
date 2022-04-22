norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='kd_pon_vpn_BEVSegmentor',
    pretrained=
    '/models/swin_tiny_patch4_window7_224.pth',
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        pretrain_style='official',
        output_missing_index_as_none=True),
    transformer=dict(
        type='Pon_combine_vpn_simple_fpn_force_transformer', use_light=True, size=(32, 32)),
    decode_head=dict(
        type='PyramidHeadKitti',
        num_classes=1,
        align_corners=True,
        priors=[0.04]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole', output_type='iou', positive_thred=0.5),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=1,
        num_outs=5))
dataset_type = 'KittiObjectDataset'
data_root = '/data/kitti_processed/kitti_object/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        reduce_zero_label=False,
        with_calib_kittiobject=True,
        imdecode_backend='pyramid'),
    dict(
        type='Resize',
        img_scale=(1024, 1024),
        resize_gt=False,
        keep_ratio=False),
    dict(type='RandomFlipKitti', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'img_norm_cfg', 'calib'))
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        reduce_zero_label=False,
        with_calib_kittiobject=True,
        imdecode_backend='pyramid'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False, resize_gt=False),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor', 'flip',
                           'flip_direction', 'img_norm_cfg', 'gt_semantic_seg',
                           'calib'))
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type='KittiObjectDataset',
        data_root=
        '/data/kitti_processed/kitti_object/',
        img_dir='img_dir/train',
        ann_dir='ann_bev_dir/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations',
                reduce_zero_label=False,
                with_calib_kittiobject=True,
                imdecode_backend='pyramid'),
            dict(
                type='Resize',
                img_scale=(1024, 1024),
                resize_gt=False,
                keep_ratio=False),
            dict(type='RandomFlipKitti', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_semantic_seg'],
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor',
                           'img_norm_cfg', 'calib'))
        ]),
    val=dict(
        type='KittiObjectDataset',
        data_root=
        '/data/kitti_processed/kitti_object/',
        img_dir='img_dir/val',
        ann_dir='ann_bev_dir/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations',
                reduce_zero_label=False,
                with_calib_kittiobject=True,
                imdecode_backend='pyramid'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False, resize_gt=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=('filename', 'ori_filename', 'ori_shape',
                                   'img_shape', 'pad_shape', 'scale_factor',
                                   'flip', 'flip_direction', 'img_norm_cfg',
                                   'gt_semantic_seg', 'calib'))
                ])
        ]),
    test=dict(
        type='KittiObjectDataset',
        data_root=
        '/data/kitti_processed/kitti_object/',
        img_dir='img_dir/val',
        ann_dir='ann_bev_dir/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations',
                reduce_zero_label=False,
                with_calib_kittiobject=True,
                imdecode_backend='pyramid'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False, resize_gt=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=('filename', 'ori_filename', 'ori_shape',
                                   'img_shape', 'pad_shape', 'scale_factor',
                                   'flip', 'flip_direction', 'img_norm_cfg',
                                   'gt_semantic_seg', 'calib'))
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=0.00018,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=4600)
checkpoint_config = dict(by_epoch=False, interval=100)
evaluation = dict(interval=4600, metric='mIoU')
backbone_norm_cfg = dict(type='LN', requires_grad=True)
