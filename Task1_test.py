'''
Date: 2021/3/8
Author: Wang Xueyang
Function: APIs to run detection on Giga-pixel images by sliding window in mmdetection.
Methods:
    detector_prepare: initiate the detector, load config and checkpoint
    fine_det_full: detect on image by sliding window with overlap, return all box results w/o NMS
    nms_after_det: non maximum suppress on detection result boxes
    show_after_nms: show the detection result and save as images
Note: This script should be run under the mmdetection environment.
'''
'''
The dataset file tree should be organized as follows:
|--IMAGE_ROOT
    |--image_train
    |--image_annos
    |--image_test
'''
import numpy as np
import time
import torch
import mmcv
import json
import time
import os
import os.path as pt
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.core import multiclass_nms
from mmdet.core import get_classes
from mmdet.apis import init_detector, inference_detector


CATE = {'1': 'person_visible', '2': 'person_full', '3': 'person_head', '4': 'vehicle'}
IMAGE_ROOT = 'YOUR/PATH/PANDA_IMAGE/' 
FASTER_RCNN_CONFIG = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py' # model configs
CKPT_PATH = 'YOUR/PATH/pts' # model checkpoints
RESULT_PATH = 'YOUR/PATH/' # save the detection results
SCORE_THRES = 0.85 

# TODO try to use various combinations of WIDTH and HEIGHT, for example:(1000,500),(2000,1000)...
HEIGHT = 500 # Sliding Window Height
WIDTH = 1000 # Sliding Window Width


if not pt.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)


def detector_prepare(ckpt_path,
                     cfg_path=FASTER_RCNN_CONFIG,
                     is_pretrained=False):
    # read config file
    cfg = mmcv.Config.fromfile(cfg_path)
    if not is_pretrained:
        cfg.model.pretrained = None
    model = init_detector(cfg_path, ckpt_path, device='cuda:0')

    return cfg, model


def fine_det_full(img_path,
                  detector_cfg,
                  detector_model,
                  WIDTH = 2048,
                  HEIGHT = 1024,
                  overlap=0.1,
                  scales=[0.1, 0.3],
                  edge_margin=10,
                  score_thres=SCORE_THRES):
    '''
    functions:
        fine detect on an image by using sliding window with overlap
        return the list of bboxes and labels
    parameters:
        overlap: overlap ratio of sliding windows
        scales: the downsample scales of raw input image before detecting
        edge_margin: delete detected result boxes if they are to close to the sliding window edge, edge_margin
                    is the threshold of distance between boxes and window edge
        score_thres: only keep result boxes whose confidence is more than this threshold
    '''
    raw_img = mmcv.imread(img_path)
    raw_height, raw_width = raw_img.shape[:2]

    # detect on images
    labels_list = []
    bboxes_list = []
    x_step = int(WIDTH * (1 - overlap))
    y_step = int(HEIGHT * (1 - overlap))
    for scale in scales:
        n = 0
        src_img = mmcv.imresize(
            raw_img, (int(raw_width * scale), int(raw_height * scale)),
            interpolation='area')
        src_height, src_width = src_img.shape[:2]
        # sub image generate
        ini_y = 0
        while ini_y < src_height:
            ini_x = 0
            while ini_x < src_width:
                tic = time.time()
                sub_img = src_img[ini_y:ini_y + HEIGHT, ini_x:ini_x + WIDTH]
                # result = inference_detector(detector_model, sub_img, detector_cfg)
                result = inference_detector(detector_model, sub_img)
                # process result
                labels = [
                    np.full(bbox.shape[0], i, dtype=np.int32)
                    for i, bbox in enumerate(result)
                ]
                labels = np.concatenate(labels)
                bboxes = np.vstack(result)
                for i, bbox in enumerate(bboxes):
                    score = bbox[4]
                    if score > score_thres:
                        x1 = bbox[0]
                        y1 = bbox[1]
                        x2 = bbox[2]
                        y2 = bbox[3]
                        if x1 > edge_margin and y1 > edge_margin and x2 < WIDTH - edge_margin and y2 < HEIGHT - edge_margin:
                            bboxes_list.append([(x1 + ini_x) / scale,
                                                (y1 + ini_y) / scale,
                                                (x2 + ini_x) / scale,
                                                (y2 + ini_y) / scale, score])
                            labels_list.append(labels[i])
                n += 1
                toc = time.time()
                # print('finish: scale: ' + str(scale) + ' num: ' + str(n) +
                #       ' cost time: ' + str(toc - tic))
                if ini_x == src_width - WIDTH:
                    break
                ini_x += x_step
                if ini_x + WIDTH > src_width:
                    ini_x = src_width - WIDTH

            if ini_y == src_height - HEIGHT:
                break
            ini_y += y_step
            if ini_y + HEIGHT > src_height:
                ini_y = src_height - HEIGHT

    return bboxes_list, labels_list


def nms_after_det(bboxes_list,
                  labels_list,
                  is_pretrained=False,
                  class_names=get_classes('coco'),
                  is_rcnn=True):
    '''
    non maximum suppress on detection result boxes
    input bboxes list and labels list, return ndarray of nms result
    result format: det_bboxes: [[x1, y1, x2, y2, score],...]  det_labels: [0 0 0 0 1 1 1 2 2 ...]
    '''
    # read config file
    if is_rcnn:
        cfg_path = FASTER_RCNN_CONFIG
    cfg = mmcv.Config.fromfile(cfg_path)
    if not is_pretrained:
        cfg.model.pretrained = None

    # NMS
    multi_bboxes = []
    multi_scores = []
    for i, bbox in enumerate(bboxes_list):
        # only show vehicles
        # if 2 <= (labels_list[i] + 1) <= 8:  # vehicles
        if 0 <= labels_list[
                i] <= 7:  # choose what to keep, now keep person and all vehicles
            multi_bboxes.append(bbox[0:4])
            # temp = [0 for _ in range(len(class_names))]
            temp = [0 for _ in range(3)]
            temp[labels_list[i] + 1] = bbox[4]
            multi_scores.append(temp)

    # if result is null
    if not multi_scores:
        return np.array([]), np.array([])

    if is_rcnn:
        det_bboxes, det_labels = multiclass_nms(
            torch.from_numpy(np.array(multi_bboxes).astype(np.float32)),
            torch.from_numpy(np.array(multi_scores).astype(np.float32)),
            cfg.model.test_cfg.rcnn.score_thr, cfg.model.test_cfg.rcnn.nms,
            cfg.model.test_cfg.rcnn.max_per_img)
    else:
        det_bboxes, det_labels = multiclass_nms(
            torch.from_numpy(np.array(multi_bboxes).astype(np.float32)),
            torch.from_numpy(np.array(multi_scores).astype(np.float32)),
            cfg.test_cfg.score_thr, cfg.test_cfg.nms, cfg.test_cfg.max_per_img)

    return det_bboxes.numpy(), det_labels.numpy()


def show_after_nms(
        img_path,
        det_bboxes,
        det_labels,
        target_class,
        save_dir,
        show_scale=0.05,
        score_thres=SCORE_THRES):
    # show the detection result and save it
    # load full image
    full_img = mmcv.imread(img_path)
    full_height, full_width = full_img.shape[:2]
    full_img = mmcv.imresize(
        full_img,
        (int(full_width * show_scale), int(full_height * show_scale)))

    # transfer scale of detection results
    det_bboxes[:, 0:4] *= show_scale

    # save result after NMS
    mmcv.imshow_det_bboxes(
        full_img.copy(),
        det_bboxes,
        det_labels,
        class_names=['unsure', target_class],
        score_thr=score_thres,
        out_file=save_dir,
        show=False,
        wait_time=0,
    )
    return None


def merge_all_results(dirs, savepath):
    # merge 4 categories detect results
    results = []
    for file_dir in dirs:
        with open(file_dir, 'r') as f:
            result = json.load(f)
            results.extend(result)
    with open(savepath , 'w') as f:
        json.dump(results, f, indent=4)

    print('merge results done')


def main(cate_id, faster_rcnn_url, result_path):
    #  run detection on all the test images
    RESULTS = []
    with open(pt.join(IMAGE_ROOT, 'image_annos', 'person_bbox_test_A.json')) as f:
        test_annos = json.load(f)
    
    img_paths = []
    img_ids = []
    cfg, model = detector_prepare(faster_rcnn_url)  
    for root, dirs, files in os.walk(pt.join(IMAGE_ROOT, 'image_test')): 
        for f in files:
            img_path = os.path.join(root, f)
            img_paths.append(img_path)
            img_key = '{}/{}'.format(img_path.split('/')[-2], img_path.split('/')[-1])
            if img_key not in test_annos:
                print('img key error')
                raise InterruptedError
            img_id = test_annos[img_key]['image id']
            img_ids.append(img_id)

            
    for i, img_path in enumerate(img_paths):
        print('processing img {} in {}...'.format(i+1, img_path))
        bboxes_list, labels_list = fine_det_full(img_path, cfg, model, WIDTH, HEIGHT)
        det_bboxes, det_labels = nms_after_det(bboxes_list, labels_list)

        for idx in range(det_bboxes.shape[0]):
            item = {
                "image_id": int(img_ids[i]),
                "category_id": int(cate_id),
                "score": round(float(det_bboxes[idx][4]), 5),
                "bbox_left": round(float(det_bboxes[idx][0]), 5),
                "bbox_top": round(float(det_bboxes[idx][1]), 5),
                "bbox_width": round(float(det_bboxes[idx][2] - det_bboxes[idx][0]), 5),
                "bbox_height": round(float(det_bboxes[idx][3] - det_bboxes[idx][1]), 5)
            }
            RESULTS.append(item)

        # show_after_nms(img_path, det_bboxes, det_labels, CATE[cate_id], pt.join('./visualize_results', img_path.split('/')[-1]))

    with open(result_path, 'w') as f:
        json.dump(RESULTS, f, indent=4)


if __name__ == '__main__':
    # inference for each category
    for i in range(1,5):
        cate_id = str(i)
        print('processing category {}...'.format(CATE[cate_id]))
        print('sliding window size = ({},{})'.format(WIDTH, HEIGHT))
        result_path = pt.join(RESULT_PATH, '{}.json'.format(CATE[cate_id]))
        faster_rcnn_url = pt.join(CKPT_PATH, '{}.pth'.format(CATE[cate_id]))       # model urls
        main(cate_id, faster_rcnn_url, result_path)

    # merge the detection results of each category
    result_paths = []
    for i in range(1,5):
        cate_id = str(i)
        result_path = pt.join(RESULT_PATH, '{}.json'.format(CATE[cate_id]))
        result_paths.append(result_path)
    
    savepath = pt.join(RESULT_PATH, 'merge.json')
    merge_all_results(result_paths,savepath)
    
