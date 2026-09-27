"""RTMDet-Ins-m comparison on the data-v2 train/val split."""

_base_ = '/root/autodl-tmp/mmdetection/configs/rtmdet/rtmdet-ins_m_8xb32-300e_coco.py'

classes = ('Rice leaffolder', 'Rice stemborers')
metainfo = dict(classes=classes)
data_root = '/root/yolo_data/'
annotation_root = '/root/yolo_plus/data/table5/annotations/'

model = dict(
    bbox_head=dict(num_classes=2),
    test_cfg=dict(score_thr=0.001),
)

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        ann_file=annotation_root + 'train.json',
        data_prefix=dict(img='images/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
    ),
)
val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        ann_file=annotation_root + 'val.json',
        data_prefix=dict(img='images/'),
        metainfo=metainfo,
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(
    ann_file=annotation_root + 'val.json',
    metric=['bbox', 'segm'],
)
test_evaluator = val_evaluator

train_cfg = dict(max_epochs=300, val_interval=1, dynamic_intervals=[(280, 1)])
optim_wrapper = dict(
    type='AmpOptimWrapper',
    accumulative_counts=2,
    optimizer=dict(lr=0.00025),
)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000,
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0000125,
        begin=150,
        end=300,
        T_max=150,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]
default_hooks = dict(
    checkpoint=dict(
        interval=10,
        save_best='coco/segm_mAP',
        rule='greater',
        max_keep_ckpts=2,
    ),
)
randomness = dict(seed=42)
load_from = (
    '/root/autodl-tmp/table5-weights/'
    'rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth'
)
