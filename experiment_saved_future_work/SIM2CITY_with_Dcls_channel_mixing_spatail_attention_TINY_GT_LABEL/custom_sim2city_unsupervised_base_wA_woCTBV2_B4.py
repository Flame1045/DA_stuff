dataset_type = 'CocoDataset'
data_root = 'data/coco/'
classes = ('car', )
eval_peroid = 100
saving_peroid = 100
epochs = 15
batch_size = 4
target_images = 2975
source_images = 10000
numbers_of_images = 21900
total_iters = 82125
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    persistent_workers=True,
    train=[
        dict(
            type='CocoDataset',
            ann_file='data/coco/Sim2Real_source/sim10k_train.json',
            img_prefix='data/coco/Sim2Real_source/JPEGImages/',
            classes=('car', ),
            filter_empty_gt=False,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='Resize',
                    img_scale=(1024, 1024),
                    ratio_range=(0.1, 2.0),
                    multiscale_mode='range',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(1024, 1024),
                    recompute_bbox=True,
                    allow_negative_crop=True),
                dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Pad',
                    size=(1024, 1024),
                    pad_val=dict(img=(114, 114, 114))),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]),
        dict(
            type='CocoDataset',
            ann_file=
            'data/coco/Sim2Real_target/annotations/instances_train2017.json',
            img_prefix='data/coco/Sim2Real_target/train2017/',
            classes=('car', ),
            filter_empty_gt=False,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='Resize',
                    img_scale=(1024, 1024),
                    ratio_range=(0.1, 2.0),
                    multiscale_mode='range',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(1024, 1024),
                    recompute_bbox=True,
                    allow_negative_crop=True),
                dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Pad',
                    size=(1024, 1024),
                    pad_val=dict(img=(114, 114, 114))),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]),
        dict(
            type='CocoDataset',
            ann_file=
            'data/coco/Sim2Real_target/annotations/instances_train2017.json',
            img_prefix='data/coco/Sim2Real_target/train2017/',
            classes=('car', ),
            filter_empty_gt=False,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='Resize',
                    img_scale=(1024, 1024),
                    ratio_range=(0.1, 2.0),
                    multiscale_mode='range',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(1024, 1024),
                    recompute_bbox=True,
                    allow_negative_crop=True),
                dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Pad',
                    size=(1024, 1024),
                    pad_val=dict(img=(114, 114, 114))),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]),
        dict(
            type='CocoDataset',
            ann_file=
            'data/coco/Sim2Real_target/annotations/instances_train2017.json',
            img_prefix='data/coco/Sim2Real_target/train2017/',
            classes=('car', ),
            filter_empty_gt=False,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='Resize',
                    img_scale=(1024, 1024),
                    ratio_range=(0.1, 2.0),
                    multiscale_mode='range',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(1024, 1024),
                    recompute_bbox=True,
                    allow_negative_crop=True),
                dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Pad',
                    size=(1024, 1024),
                    pad_val=dict(img=(114, 114, 114))),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]),
        dict(
            type='CocoDataset',
            ann_file=
            'data/coco/Sim2Real_target/annotations/instances_train2017.json',
            img_prefix='data/coco/Sim2Real_target/train2017/',
            classes=('car', ),
            filter_empty_gt=False,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='Resize',
                    img_scale=(1024, 1024),
                    ratio_range=(0.1, 2.0),
                    multiscale_mode='range',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(1024, 1024),
                    recompute_bbox=True,
                    allow_negative_crop=True),
                dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Pad',
                    size=(1024, 1024),
                    pad_val=dict(img=(114, 114, 114))),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ])
    ],
    val=dict(
        type='CocoDataset',
        ann_file='data/coco/Sim2Real_target/annotations/instances_val2017.json',
        img_prefix='data/coco/Sim2Real_target/val2017/',
        classes=('car', ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Pad',
                        size=(1024, 1024),
                        pad_val=dict(img=(114, 114, 114))),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file='data/coco/Sim2Real_target/annotations/instances_val2017.json',
        img_prefix='data/coco/Sim2Real_target/val2017/',
        classes=('car', ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Pad',
                        size=(1024, 1024),
                        pad_val=dict(img=(114, 114, 114))),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=100, metric='bbox', classwise=True, iou_thrs=[0.5])
checkpoint_config = dict(interval=100, by_epoch=False)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'experiment_saved/SIM2CITY_with_Dcls_channel_mixing_spatail_attention/pretrained/SIM2CITY.pth'
resume_from = None
workflow = [('train', 100)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
num_dec_layer = 6
lambda_2 = 2.0
model = dict(
    type='CoDETR',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=12.0),
        loss_bbox=dict(type='L1Loss', loss_weight=12.0)),
    query_head=dict(
        type='CoDeformDETRHead',
        num_query=300,
        num_classes=1,
        in_channels=2048,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=True,
        mixed_selection=True,
        transformer=dict(
            type='CoDeformableDetrTransformer',
            num_co_heads=2,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=[
                    dict(
                        type='BaseTransformerLayer_',
                        attn_cfgs=dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            dropout=0.0),
                        feedforward_channels=2048,
                        ffn_dropout=0.0,
                        operation_order=('self_attn', 'norm', 'ffn', 'adapter',
                                         'norm')),
                    dict(
                        type='BaseTransformerLayer_',
                        attn_cfgs=dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            dropout=0.0),
                        feedforward_channels=2048,
                        ffn_dropout=0.0,
                        operation_order=('self_attn', 'norm', 'ffn', 'adapter',
                                         'norm')),
                    dict(
                        type='BaseTransformerLayer_',
                        attn_cfgs=dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            dropout=0.0),
                        feedforward_channels=2048,
                        ffn_dropout=0.0,
                        operation_order=('self_attn', 'norm', 'ffn', 'adapter',
                                         'norm')),
                    dict(
                        type='BaseTransformerLayer_',
                        attn_cfgs=dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            dropout=0.0),
                        feedforward_channels=2048,
                        ffn_dropout=0.0,
                        operation_order=('self_attn', 'norm', 'ffn', 'adapter',
                                         'norm')),
                    dict(
                        type='BaseTransformerLayer_',
                        attn_cfgs=dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            dropout=0.0),
                        feedforward_channels=2048,
                        ffn_dropout=0.0,
                        operation_order=('self_attn', 'norm', 'ffn', 'adapter',
                                         'norm')),
                    dict(
                        type='BaseTransformerLayer_',
                        attn_cfgs=dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            dropout=0.0),
                        feedforward_channels=2048,
                        ffn_dropout=0.0,
                        operation_order=('self_attn', 'norm', 'ffn', 'adapter',
                                         'norm'))
                ]),
            decoder=dict(
                type='CoDeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                look_forward_twice=True,
                transformerlayers=[
                    dict(
                        type='DetrTransformerDecoderLayer_',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.0),
                            dict(
                                type='MultiScaleDeformableAttention',
                                embed_dims=256,
                                dropout=0.0)
                        ],
                        feedforward_channels=2048,
                        ffn_dropout=0.0,
                        operation_order=('self_attn',
                                         'cross_attn_seq_adapterV25x5_slide8',
                                         'norm', 'ffn', 'adapter', 'norm')),
                    dict(
                        type='DetrTransformerDecoderLayer_',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.0),
                            dict(
                                type='MultiScaleDeformableAttention',
                                embed_dims=256,
                                dropout=0.0)
                        ],
                        feedforward_channels=2048,
                        ffn_dropout=0.0,
                        operation_order=('self_attn',
                                         'cross_attn_seq_adapterV25x5_slide8',
                                         'norm', 'ffn', 'adapter', 'norm')),
                    dict(
                        type='DetrTransformerDecoderLayer_',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.0),
                            dict(
                                type='MultiScaleDeformableAttention',
                                embed_dims=256,
                                dropout=0.0)
                        ],
                        feedforward_channels=2048,
                        ffn_dropout=0.0,
                        operation_order=('self_attn',
                                         'cross_attn_seq_adapterV25x5_slide8',
                                         'norm', 'ffn', 'adapter', 'norm')),
                    dict(
                        type='DetrTransformerDecoderLayer_',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.0),
                            dict(
                                type='MultiScaleDeformableAttention',
                                embed_dims=256,
                                dropout=0.0)
                        ],
                        feedforward_channels=2048,
                        ffn_dropout=0.0,
                        operation_order=('self_attn',
                                         'cross_attn_seq_adapterV25x5_slide8',
                                         'norm', 'ffn', 'adapter', 'norm')),
                    dict(
                        type='DetrTransformerDecoderLayer_',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.0),
                            dict(
                                type='MultiScaleDeformableAttention',
                                embed_dims=256,
                                dropout=0.0)
                        ],
                        feedforward_channels=2048,
                        ffn_dropout=0.0,
                        operation_order=('self_attn',
                                         'cross_attn_seq_adapterV25x5_slide8',
                                         'norm', 'ffn', 'adapter', 'norm')),
                    dict(
                        type='DetrTransformerDecoderLayer_',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.0),
                            dict(
                                type='MultiScaleDeformableAttention',
                                embed_dims=256,
                                dropout=0.0)
                        ],
                        feedforward_channels=2048,
                        ffn_dropout=0.0,
                        operation_order=('self_attn',
                                         'cross_attn_seq_adapterV25x5_slide8',
                                         'norm', 'ffn', 'adapter', 'norm'))
                ])),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    roi_head=[
        dict(
            type='CoStandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[8, 16, 32, 64],
                finest_scale=112),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.2),
                loss_bbox=dict(type='GIoULoss', loss_weight=12.0)))
    ],
    isSAP=True,
    isARoiLoss=False,
    gamma=0.5,
    aroiweight=1.0,
    train_cfg=[
        dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(
                    type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
        dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=4000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False))
    ],
    test_cfg=[
        dict(max_per_img=100),
        dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.0,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100)),
        dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.0,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=100)
    ])
optimizer = dict(
    type='AdamW',
    lr=0.001,
    weight_decay=1e-07,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            sampling_offsets=dict(lr_mult=0.1),
            reference_points=dict(lr_mult=0.1))))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='step', step=[5000])
runner = dict(type='IterBasedRunner', max_iters=82125)
work_dir = 'outputs/SIM2CITY_with_Dcls_channel_mixing_spatail_attention_TINY_GT_LABEL'
adapter = True
adapter_choose = [
    'slideatten', 'SAP', 'adapter', 'scalar', 'da_head', 'cls_branches',
    'reg_branches', 'label_embedding', 'rpn_head', 'roi_head', 'bbox_head'
]
da_head = False
grad_cam = False
auto_resume = False
gpu_ids = [0]
pseudo_label_flag = False
ORACLE = False
TINY_GT_LABEL = True
