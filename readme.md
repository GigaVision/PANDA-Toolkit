# Installation
1. Environment: **Python 3**
2. Get PANDA from [download page](https://tianchi.aliyun.com/competition/entrance/531855/information)
3. Install dependencies
```
    pip install -r requirements.txt
```
4. We use MMDetection to train baseline model, please refer to [this page](https://github.com/open-mmlab/mmdetection) for MMDetection installation and usage.

**NOTE**

 You can either finetune the Faster RCNN model and save the checkpoints according to the following [training tutorial](#train-on-panda-image-with-MMDetection), or download the pretrained weights from [here](https://pan.baidu.com/s/143REv-mb_H-CXWDwU3y33w) (code: c6uc) and inference on the PANDA-IMAGE test set without training. For details, please refer to [Test and inference](#test-and-inference).



# Train on PANDA-IMAGE with MMDetection

In this note, you will know how to finetune Faster RCNN model with PANDA-IMAGE dataset. 

The basic steps are as below:

1. [Prepare the PANDA-IMAGE dataset](#prepare-the-panda-image-dataset)
2. [Prepare config files of MMDetection](#prepare-config-files-of-MMDetection)
3. [Download COCO pre-trained model](#download-coco-pre-trained-model)
4. [Train on the PANDA IMAGE dataset](#train-on-the-panda-image-dataset)

## Prepare the PANDA-IMAGE dataset
We need to reorganize the dataset into COCO format. Please refer to `Task1_utils.py`.

## Prepare config files of MMDetection

The second step is to prepare config files thus the dataset could be successfully loaded. Assume that we want to use Faster RCNN, the config to train the detector on PANDA IMAGE dataset is as below. 

The **first config** need to be overwritten is the file `configs/_base_/datasets/coco_detection.py`, the config is as below.

```python
CATE_ID = '1'

classes_dict = {'1': 'visible body', '2': 'full body', '3': 'head', '4': 'vehicle'}
json_pre_dict = {'1': 'person_visible', '2': 'person_full', '3': 'person_head', '4':'vehicle'}

data_root = 'YOUR/PATH/split_' + json_pre_dict[CATE_ID].split('_')[0] +'_train/'
anno_root = 'YOUR/PATH/coco_format_json/'

classes = (classes_dict[CATE_ID],)
json_pre = json_pre_dict[CATE_ID]

dataset_type = 'CocoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=anno_root + json_pre + '_train.json',
        img_prefix=data_root + 'image_train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=anno_root + json_pre + '_val.json',
        img_prefix=data_root + 'image_train',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=anno_root + json_pre + '_val.json',
        img_prefix=data_root + 'image_train',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

# We can use the pre-trained Faster RCNN model to obtain higher performance
# load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
```

We train only one category of box during one training process.

- Change the **CATE_ID** in the above config to select target category for training
- Change the **data_root** and the **anno_root** to your training data path

The **second config** need to be overwritten is the file `configs/_base_/models/faster_rcnn_r50_fpn.py`, the config is as below.

```python
model = dict(
    type='FasterRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
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
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=1000,
            max_num=1000,
            nms_thr=0.7,
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
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_num=1000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=500)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

```

The **third config** need to be modified is the file `configs/_base_/default_runtime.py`, you need to set the key  **load_from** as `'./checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'`

## Download COCO pre-trained model

Download  COCO pre-trained model from [here](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth) and put it to `checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth`.

## Train on the PANDA IMAGE dataset

To train a model with the new config, you can simply run

```shell
python tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py [optional arguments]
```

For more detailed usages, please refer to the [Case 1](https://github.com/open-mmlab/mmdetection/blob/master/docs/1_exist_data_model.md).


## Test and inference
1. Copy `Task1_test.py` to MMdetection root folder.
2. Finetune the Faster RCNN model and save the checkpoints according to the tutorial above, or download the pretrained weights from [here](https://pan.baidu.com/s/143REv-mb_H-CXWDwU3y33w) (code: c6uc) and update the PATH setting in `Task1_test.py`
3. To test the model, navigate to MMdetection root folder, and then you can simply run the following code to generate a result file to submit.
```shell
python Task1_test.py
```
**NOTE**

Try to use various combinations of WIDTH and HEIGHT, for example:(1000,500),(2000,1000)... We get the following baseline results with window size of (1000,500) for category 1、2、4, and window size of (500,250) for category 3.

# Baseline results

| Matrices                | Details                                           | Score |
| ----------------------- | ------------------------------------------------- | ----- |
| Average Precision  (AP) | @[ IoU=0.50:0.95 \| area =  all \| maxDets=500 ]  | 0.312 |
| Average Precision  (AP) | @[ IoU=0.50    \| area =  all \| maxDets=500 ]    | 0.482 |
| Average Precision  (AP) | @[ IoU=0.75    \| area =  all \| maxDets=500 ]    | 0.352 |
| Average Precision  (AP) | @[ IoU=0.50:0.95 \| area = small \| maxDets=500 ] | 0.203 |
| Average Precision  (AP) | @[ IoU=0.50:0.95 \| area =medium \| maxDets=500 ] | 0.376 |
| Average Precision  (AP) | @[ IoU=0.50:0.95 \| area = large \| maxDets=500 ] | 0.352 |
| Average Recall   (AR)   | @[ IoU=0.50:0.95 \| area =  all \| maxDets= 10 ]  | 0.061 |
| Average Recall   (AR)   | @[ IoU=0.50:0.95 \| area =  all \| maxDets=100 ]  | 0.266 |
| Average Recall   (AR)   | @[ IoU=0.50:0.95 \| area =  all \| maxDets=500 ]  | 0.347 |
| Average Recall   (AR)   | @[ IoU=0.50:0.95 \| area = small \| maxDets=500 ] | 0.229 |
| Average Recall   (AR)   | @[ IoU=0.50:0.95 \| area =medium \| maxDets=500 ] | 0.413 |
| Average Recall   (AR)   | @[ IoU=0.50:0.95 \| area = large \| maxDets=500 ] | 0.379 |

