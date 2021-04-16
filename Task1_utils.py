# --------------------------------------------------------
# The pipeline of preparing the training data for mmdetection(https://github.com/open-mmlab/mmdetection)
# Written by Wang Xueyang (wangxuey19@mails.tsinghua.edu.cn), Version 20210301
# Inspired from DOTA dataset devkit (https://github.com/CAPTAIN-WHU/DOTA_devkit)
# --------------------------------------------------------

import numpy as np
import argparse
import os
import os.path as pt
import json
import funcy
import panda_utils as util
from PANDA import PANDA_IMAGE, PANDA_VIDEO
from ImgSplit import ImgSplit
from ResultMerge import DetResMerge
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.model_selection import train_test_split

'''
The dataset file tree should be organized as follows:
|--IMAGE_ROOT
    |--image_train
    |--image_annos
    |--image_test
'''

CATE = {'1': 'person_visible', '2': 'person_full', '3': 'person_head', '4': 'vehicle'}
IMAGE_ROOT = 'YOUR/PATH/PANDA_IMAGE/'
OUT_PERSON_PATH = 'YOUR/PATH/split_person_train/'
OUT_VEHICLE_PATH = 'YOUR/PATH/split_vehicle_train/'
COCO_FORMAT_JSON_PATH = 'YOUR/PATH/coco_format_json'

if not pt.exists(OUT_PERSON_PATH):
    os.makedirs(OUT_PERSON_PATH)
if not pt.exists(OUT_VEHICLE_PATH):
    os.makedirs(OUT_VEHICLE_PATH)
if not pt.exists(COCO_FORMAT_JSON_PATH):
    os.makedirs(COCO_FORMAT_JSON_PATH)

def main():
    '''1. split image and annotations'''
    print('procsess 1.....')
    # Note: The most common runtime error is file path error. 
    person_anno_file = 'person_bbox_train.json'
    vehicle_anno_file = 'vehicle_bbox_train.json'
    
    # We provide the scale param before split the images and labels.
    split = ImgSplit(IMAGE_ROOT, person_anno_file, 'person', OUT_PERSON_PATH, person_anno_file)
    split.splitdata(1)
    split = ImgSplit(IMAGE_ROOT, vehicle_anno_file, 'vehicle', OUT_VEHICLE_PATH, vehicle_anno_file)
    split.splitdata(1)

    '''2. transfer annotations to COCO format'''
    print('procsess 2.....')
    src_person_file  = pt.join(OUT_PERSON_PATH, 'image_annos', person_anno_file) 
    src_vehicle_file = pt.join(OUT_VEHICLE_PATH, 'image_annos', vehicle_anno_file) 

    tgt_person_file = pt.join(COCO_FORMAT_JSON_PATH, person_anno_file)
    tgt_vehicle_file = pt.join(COCO_FORMAT_JSON_PATH, vehicle_anno_file)
    
    util.generate_coco_anno_person(src_person_file, tgt_person_file)
    util.generate_coco_anno_vehicle(src_vehicle_file, tgt_vehicle_file)

    '''3. save each kind of boxes in a single file'''
    print('procsess 3.....')
    split_person_cate(tgt_person_file,COCO_FORMAT_JSON_PATH)
    split_vehicle_cate(tgt_vehicle_file,COCO_FORMAT_JSON_PATH)


    '''4. split train and val'''
    print('procsess 4.....')
    for cid in range(1,5):
        annotations = pt.join(COCO_FORMAT_JSON_PATH, CATE[str(cid)] + '_train_val.json')
        trainpath = pt.join(COCO_FORMAT_JSON_PATH, CATE[str(cid)] + '_train.json')
        valpath = pt.join(COCO_FORMAT_JSON_PATH, CATE[str(cid)] + '_val.json')
        split_train_val(annotations, trainpath, valpath)


def split_train_val(annotations, trainpath, valpath, splitrate=0.7):
    with open(annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        # number_of_images = len(images)
        train, val = train_test_split(images, train_size=splitrate)

        save_coco(trainpath, train, filter_annotations(annotations, train),
                  categories)
        save_coco(valpath, val, filter_annotations(annotations, val),
                  categories)

        print("Saved {} entries in {} and {} in {}".format(
            len(train), trainpath, len(val), valpath))


def save_coco(file, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump(
            {
                'images': images,
                'annotations': annotations,
                'categories': categories
            },
            coco,
            indent=4,
            sort_keys=True)


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids,
                         annotations)


def split_person_cate(srcpath, savepath):
    # split 3 categories of person boxes and change category id
    for cid in range(1,4):
        with open(srcpath,'r') as f:
            data = json.load(f)

        newdata = {}
        newdata['categories'] = []
        newdata['images'] = []
        newdata['annotations'] = []
        newdata['type'] = data['type']

        ids = []

        for item in data['categories']:
            if int(item['id']) == cid:
                item['id'] = 1
                newdata['categories'].append(item)

        for item in data['annotations']:
            if int(item["category_id"]) == cid:
                item["category_id"] = 1
                newdata['annotations'].append(item)
                ids.append(item['image_id'])

        for item in data['images']:
            if item['id'] in ids:
                newdata['images'].append(item)

        with open(pt.join(savepath, CATE[str(cid)] + '_train_val.json'), 'w') as f:
            json.dump(newdata, f, indent=4)  


def split_vehicle_cate(srcpath, savepath):
    # change vehicle category id
    with open(srcpath,'r') as f:
        data = json.load(f)

    data['categories'][0]['id'] = 1

    for item in data['annotations']:
        if int(item["category_id"]) == 4:
            item["category_id"] = 1

    with open(pt.join(savepath, CATE['4'] + '_train_val.json'),'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    main()