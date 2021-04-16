# --------------------------------------------------------
# Basic functions which are useful for process PANDA data
# Written by Wang Xueyang  (wangxuey19@mails.tsinghua.edu.cn), Version 20200321
# Inspired from DOTA dataset devkit (https://github.com/CAPTAIN-WHU/DOTA_devkit)
# --------------------------------------------------------

import os
import json
import glob
import random

CATEGORY = {
    'visible body': 1,
    'full body': 2,
    'head': 3,
    'vehicle': 4
}


def custombasename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])


def GetFileFromThisRootDir(dir, ext=None):
    allfiles = []
    needExtFilter = (ext != None)
    for root, dirs, files in os.walk(dir):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]
            if needExtFilter and extension in ext:
                allfiles.append(filepath)
            elif not needExtFilter:
                allfiles.append(filepath)
    return allfiles


def restrain_between_0_1(values_list):
    return_list = []
    for value in values_list:
        if value < 0:
            new_value = 0
        elif value > 1:
            new_value = 1
        else:
            new_value = value
        return_list.append(new_value)

    return return_list


def RectDict2List(rectdict, imgwidth, imgheight, scale, mode='tlbr'):
    x1, y1, x2, y2 = restrain_between_0_1([rectdict['tl']['x'], rectdict['tl']['y'],
                                           rectdict['br']['x'], rectdict['br']['y']])
    xmin = int(x1 * imgwidth * scale)
    ymin = int(y1 * imgheight * scale)
    xmax = int(x2 * imgwidth * scale)
    ymax = int(y2 * imgheight * scale)

    if mode == 'tlbr':
        return xmin, ymin, xmax, ymax
    elif mode == 'tlwh':
        return xmin, ymin, xmax - xmin, ymax - ymin


def List2RectDict(bbox, imgwidth, imgheight, scale, mode='tlwh'):
    if mode == 'tlbr':
        xmin, ymin, xmax, ymax = bbox
    elif mode == 'tlwh':
        xmin, ymin, w, h = bbox
        xmax, ymax = xmin + w, ymin + h

    rectdict = {
        'tl': {
            'x': xmin / imgwidth / scale,
            'y': ymin / imgheight / scale
        },
        'br': {
            'x': xmax / imgwidth / scale,
            'y': ymax / imgheight / scale
        }
    }

    return rectdict


def parse_panda_rect(annopath, annomode, showwidth):
    """
        parse the panda ground truth for visualization
    """
    images = {}
    print('Loading annotation json file: {}'.format(annopath))
    with open(annopath, 'r') as load_f:
        annodict = json.load(load_f)

    if annomode == 'person':
        for (imagename, imagedict) in annodict.items():
            parsed = []
            imgwidth = imagedict['image size']['width']
            imgheight = imagedict['image size']['height']
            scale = showwidth / imgwidth
            for object_dict in imagedict['objects list']:
                objcate = object_dict['category']
                if objcate == 'person':
                    personpose = object_dict['riding type'] if object_dict['pose'] == 'riding' else object_dict['pose']
                    fullrect = RectDict2List(object_dict['rects']['full body'], imgwidth, imgheight, scale)
                    visiblerect = RectDict2List(object_dict['rects']['visible body'], imgwidth, imgheight, scale)
                    headrect = RectDict2List(object_dict['rects']['head'], imgwidth, imgheight, scale)
                    parsed.append({
                        'ignore': False,
                        'cate': personpose,
                        'fullrect': fullrect,
                        'visiblerect': visiblerect,
                        'headrect': headrect
                    })
                else:
                    rect = RectDict2List(object_dict['rect'], imgwidth, imgheight, scale)
                    parsed.append({
                        'ignore': True,
                        'cate': objcate,
                        'rect': rect,
                    })
            images[imagename] = parsed

    elif annomode == 'vehicle':
        for (imagename, imagedict) in annodict.items():
            parsed = []
            imgwidth = imagedict['image size']['width']
            imgheight = imagedict['image size']['height']
            scale = showwidth / imgwidth
            for object_dict in imagedict['objects list']:
                objcate = object_dict['category']
                rect = RectDict2List(object_dict['rect'], imgwidth, imgheight, scale)
                if objcate == 'vehicles':
                    parsed.append({
                        'ignore': True,
                        'cate': objcate,
                        'rect': rect,
                    })
                else:
                    parsed.append({
                        'ignore': False,
                        'cate': objcate,
                        'rect': rect,
                    })
            images[imagename] = parsed

    elif annomode == 'headbbox':
        for (imagename, imagedict) in annodict.items():
            parsed = []
            imgwidth = imagedict['image size']['width']
            imgheight = imagedict['image size']['height']
            scale = showwidth / imgwidth
            for object_dict in imagedict['objects list']:
                rect = RectDict2List(object_dict['rect'], imgwidth, imgheight, scale)
                parsed.append(rect)
            images[imagename] = parsed

    elif annomode == 'headpoint':
        for (imagename, imagedict) in annodict.items():
            parsed = []
            imgwidth = imagedict['image size']['width']
            imgheight = imagedict['image size']['height']
            scale = showwidth / imgwidth
            for object_dict in imagedict['objects list']:
                x = int(object_dict['rect']['x'] * imgwidth * scale)
                y = int(object_dict['rect']['y'] * imgheight * scale)
                parsed.append((x, y))
            images[imagename] = parsed

    return images


def GT2DetRes(gtpath, outdetpath):
    """
        transfer format: groundtruth to detection results
    :param gtpath: the path to groundtruth json file
    :param outdetpath:the path to output detection result json file
    :return:
    """
    print('Loading groundtruth json file: {}'.format(gtpath))
    with open(gtpath, 'r') as load_f:
        gt = json.load(load_f)
    outputlist = []
    for (imgname, imgdict) in gt.items():
        imageid = imgdict['image id']
        imgwidth = imgdict['image size']['width']
        imgheight = imgdict['image size']['height']
        for obj in imgdict['objects list']:
            if obj['category'] == 'person':
                rect = RectDict2List(obj['rects']['visible body'], imgwidth, imgheight, 1, mode='tlwh')
            else:
                rect = RectDict2List(obj['rect'], imgwidth, imgheight, 1, mode='tlwh')
            outputlist.append({
                "image_id": imageid,
                "category_id": 1,
                "bbox": rect,
                "score": 1
            })
    with open(outdetpath, 'w', encoding='utf-8') as f:
        dict_str = json.dumps(outputlist, indent=2)
        f.write(dict_str)


def DetRes2GT(detrespath, outgtpath, gtannopath):
    """
        transfer format: detection results to groundtruth
    :param detrespath: the path to input detection result json file
    :param outgtpath: the path to output groundtruth json file
    :param gtannopath: source annotation json file path for image data
    :return:
    """
    print('Loading source groundtruth json file: {}'.format(gtannopath))
    with open(gtannopath, 'r') as load_f:
        gtanno = json.load(load_f)
    print('Loading detection result json file: {}'.format(detrespath))
    with open(detrespath, 'r') as load_f:
        detres = json.load(load_f)

    outgt = {}
    for (imgname, imgdict) in gtanno.items():
        outgt[imgname] = {
            "image id": imgdict['image id'],
            "image size": imgdict['image size'],
            "objects list": []
        }

    for detdict in detres:
        imageid = detdict["image_id"]
        bbox = detdict["bbox"]
        for imgname in outgt.keys():
            if outgt[imgname]['image id'] == imageid:
                outgt[imgname]['objects list'].append({
                    "category": "person",
                    "rect": List2RectDict(bbox, outgt[imgname]['image size']['width'],
                                          outgt[imgname]['image size']['height'], 1)
                })

    with open(outgtpath, 'w', encoding='utf-8') as f:
        dict_str = json.dumps(outgt, indent=2)
        f.write(dict_str)


def generate_coco_anno(personsrcfile, vehiclesrcfile, tgtfile, keywords=None):
    """
    transfer ground truth to COCO format
    :param personsrcfile: person ground truth file path
    :param vehiclesrcfile: vehicle ground truth file path
    :param tgtfile: generated file save path
    :param keywords: list of str, only keep image with keyword in image name
    :return:
    """
    attrDict = dict()
    attrDict["categories"] = [
        {"supercategory": "none", "id": 1, "name": 'visible body'},
        {"supercategory": "none", "id": 2, "name": 'full body'},
        {"supercategory": "none", "id": 3, "name": 'head'},
        {"supercategory": "none", "id": 4, "name": 'vehicle'}
    ]
    with open(personsrcfile, 'r') as load_f:
        person_anno_dict = json.load(load_f)
    with open(vehiclesrcfile, 'r') as load_f:
        vehicle_anno_dict = json.load(load_f)

    images = list()
    annotations = list()
    imageids = list()

    objid = 1
    for (imagename, imagedict) in person_anno_dict.items():
        if keywords:
            flag = False
            for kw in keywords:
                if kw in imagename:
                    flag = True
            if not flag:
                continue
        image = dict()
        image['file_name'] = imagename
        imgid = imagedict['image id']
        imageids.append(imgid)
        imgwidth = imagedict['image size']['width']
        imgheight = imagedict['image size']['height']
        image['height'] = imgheight
        image['width'] = imgwidth
        image['id'] = imgid
        images.append(image)
        for objdict in imagedict['objects list']:
            cate = objdict['category']
            if cate == 'person':
                for label in ['visible body', 'full body', 'head']:
                    rect = objdict['rects'][label]
                    annotation = dict()
                    annotation["image_id"] = imgid
                    annotation["ignore"] = 0
                    annotation["iscrowd"] = 0
                    x, y, w, h = RectDict2List(rect, imgwidth, imgheight, scale=1, mode='tlwh')
                    annotation["bbox"] = [x, y, w, h]
                    annotation["area"] = float(w * h)
                    annotation["category_id"] = CATEGORY[label]
                    annotation["id"] = objid
                    objid += 1
                    annotation["segmentation"] = [[x, y, x, (y + h), (x + w), (y + h), (x + w), y]]
                    annotations.append(annotation)
            else:
                annotation = dict()
                if cate == 'crowd':
                    annotation["iscrowd"] = 1
                else:
                    annotation["iscrowd"] = 0
                rect = objdict['rect']
                annotation["image_id"] = imgid
                annotation["ignore"] = 1
                x, y, w, h = RectDict2List(rect, imgwidth, imgheight, scale=1, mode='tlwh')
                annotation["bbox"] = [x, y, w, h]
                annotation["area"] = float(w * h)
                annotation["category_id"] = CATEGORY['visible body']
                annotation["id"] = objid
                objid += 1
                annotation["segmentation"] = [[x, y, x, (y + h), (x + w), (y + h), (x + w), y]]
                annotations.append(annotation)

        for objdict in vehicle_anno_dict[imagename]['objects list']:
            cate = objdict['category']
            if cate in ['vehicles', 'fake', 'ignore']:
                annotation = dict()
                rect = objdict['rect']
                annotation["image_id"] = imgid
                annotation["iscrowd"] = 1
                annotation["ignore"] = 1
                x, y, w, h = RectDict2List(rect, imgwidth, imgheight, scale=1, mode='tlwh')
                annotation["bbox"] = [x, y, w, h]
                annotation["area"] = float(w * h)
                annotation["category_id"] = CATEGORY['vehicle']
                annotation["id"] = objid
                objid += 1
                annotation["segmentation"] = [[x, y, x, (y + h), (x + w), (y + h), (x + w), y]]
                annotations.append(annotation)
            else:
                annotation = dict()
                rect = objdict['rect']
                annotation["image_id"] = imgid
                annotation["ignore"] = 0
                annotation["iscrowd"] = 0
                x, y, w, h = RectDict2List(rect, imgwidth, imgheight, scale=1, mode='tlwh')
                annotation["bbox"] = [x, y, w, h]
                annotation["area"] = float(w * h)
                annotation["category_id"] = CATEGORY['vehicle']
                annotation["id"] = objid
                objid += 1
                annotation["segmentation"] = [[x, y, x, (y + h), (x + w), (y + h), (x + w), y]]
                annotations.append(annotation)

    attrDict["images"] = images
    attrDict["annotations"] = annotations
    attrDict["type"] = "instances"

    # print attrDict
    jsonString = json.dumps(attrDict, indent=2)
    with open(tgtfile, "w") as f:
        f.write(jsonString)

    return imageids


def generate_res_from_gt(personsrcfile, vehiclesrcfile, resFile, keywords=None):
    with open(personsrcfile, 'r') as load_f:
        person_anno_dict = json.load(load_f)
    with open(vehiclesrcfile, 'r') as load_f:
        vehicle_anno_dict = json.load(load_f)
    annotations = list()

    for (imagename, imagedict) in person_anno_dict.items():
        if keywords:
            flag = False
            for kw in keywords:
                if kw in imagename:
                    flag = True
            if not flag:
                continue
        imgid = imagedict['image id']
        imgwidth = imagedict['image size']['width']
        imgheight = imagedict['image size']['height']
        for objdict in imagedict['objects list']:
            cate = objdict['category']
            if cate == 'person':
                for label in ['visible body', 'full body', 'head']:
                    rect = objdict['rects'][label]
                    annotation = dict()
                    annotation["image_id"] = imgid
                    x, y, w, h = RectDict2List(rect, imgwidth, imgheight, scale=1, mode='tlwh')
                    annotation["bbox"] = [x, y, w, h]
                    annotation["category_id"] = CATEGORY[label]
                    annotation["score"] = 0.999
                    annotations.append(annotation)
        for objdict in vehicle_anno_dict[imagename]['objects list']:
            cate = objdict['category']
            if cate != 'vehicles':
                rect = objdict['rect']
                annotation = dict()
                annotation["image_id"] = imgid
                x, y, w, h = RectDict2List(rect, imgwidth, imgheight, scale=1, mode='tlwh')
                annotation["bbox"] = [x, y, w, h]
                annotation["category_id"] = CATEGORY['vehicle']
                annotation["score"] = 0.999
                annotations.append(annotation)

    # print attrDict
    jsonString = json.dumps(annotations, indent=2)
    with open(resFile, "w") as f:
        f.write(jsonString)

def generate_coco_anno_person(personsrcfile, tgtfile, keywords=None):
    """
    transfer ground truth (only person annotations) to COCO format
    :param personsrcfile: person ground truth file path
    :param tgtfile: generated file save path
    :param keywords: list of str, only keep image with keyword in image name
    :return:
    """
    attrDict = dict()
    attrDict["categories"] = [
        {"supercategory": "none", "id": 1, "name": 'visible body'},
        {"supercategory": "none", "id": 2, "name": 'full body'},
        {"supercategory": "none", "id": 3, "name": 'head'}
    ]
    with open(personsrcfile, 'r') as load_f:
        person_anno_dict = json.load(load_f)

    images = list()
    annotations = list()
    imageids = list()

    objid = 1
    for (imagename, imagedict) in person_anno_dict.items():
        if keywords:
            flag = False
            for kw in keywords:
                if kw in imagename:
                    flag = True
            if not flag:
                continue
        image = dict()
        image['file_name'] = imagename
        imgid = imagedict['image id']
        imageids.append(imgid)
        imgwidth = imagedict['image size']['width']
        imgheight = imagedict['image size']['height']
        image['height'] = imgheight
        image['width'] = imgwidth
        image['id'] = imgid
        images.append(image)
        for objdict in imagedict['objects list']:
            cate = objdict['category']
            if cate == 'person':
                for label in ['visible body', 'full body', 'head']:
                    rect = objdict['rects'][label]
                    annotation = dict()
                    annotation["image_id"] = imgid
                    annotation["ignore"] = 0
                    annotation["iscrowd"] = 0
                    x, y, w, h = RectDict2List(rect, imgwidth, imgheight, scale=1, mode='tlwh')
                    annotation["bbox"] = [x, y, w, h]
                    annotation["area"] = float(w * h)
                    annotation["category_id"] = CATEGORY[label]
                    annotation["id"] = objid
                    objid += 1
                    annotation["segmentation"] = [[x, y, x, (y + h), (x + w), (y + h), (x + w), y]]
                    annotations.append(annotation)
            else:
                annotation = dict()
                if cate == 'crowd':
                    annotation["iscrowd"] = 1
                else:
                    annotation["iscrowd"] = 0
                rect = objdict['rect']
                annotation["image_id"] = imgid
                annotation["ignore"] = 1
                x, y, w, h = RectDict2List(rect, imgwidth, imgheight, scale=1, mode='tlwh')
                annotation["bbox"] = [x, y, w, h]
                annotation["area"] = float(w * h)
                annotation["category_id"] = CATEGORY['visible body']
                annotation["id"] = objid
                objid += 1
                annotation["segmentation"] = [[x, y, x, (y + h), (x + w), (y + h), (x + w), y]]
                annotations.append(annotation)

    attrDict["images"] = images
    attrDict["annotations"] = annotations
    attrDict["type"] = "instances"

    # print attrDict
    jsonString = json.dumps(attrDict, indent=2)
    with open(tgtfile, "w") as f:
        f.write(jsonString)

    return imageids


def generate_coco_anno_vehicle(vehiclesrcfile, tgtfile, keywords=None):
    """
     transfer ground truth (only vehicle annotations) to COCO format
    :param vehiclesrcfile: vehicle ground truth file path
    :param tgtfile: generated file save path
    :param keywords: list of str, only keep image with keyword in image name
    :return:
    """
    attrDict = dict()
    attrDict["categories"] = [
        {"supercategory": "none", "id": 4, "name": 'vehicle'}
    ]
    with open(vehiclesrcfile, 'r') as load_f:
        vehicle_anno_dict = json.load(load_f)

    images = list()
    annotations = list()
    imageids = list()

    objid = 1
    for (imagename, imagedict) in vehicle_anno_dict.items():
        if keywords:
            flag = False
            for kw in keywords:
                if kw in imagename:
                    flag = True
            if not flag:
                continue
        image = dict()
        image['file_name'] = imagename
        imgid = imagedict['image id']
        imageids.append(imgid)
        imgwidth = imagedict['image size']['width']
        imgheight = imagedict['image size']['height']
        image['height'] = imgheight
        image['width'] = imgwidth
        image['id'] = imgid
        images.append(image)
        

        for objdict in imagedict['objects list']:
            cate = objdict['category']
            if cate in ['vehicles', 'fake', 'ignore']:
                annotation = dict()
                rect = objdict['rect']
                annotation["image_id"] = imgid
                annotation["iscrowd"] = 1
                annotation["ignore"] = 1
                x, y, w, h = RectDict2List(rect, imgwidth, imgheight, scale=1, mode='tlwh')
                annotation["bbox"] = [x, y, w, h]
                annotation["area"] = float(w * h)
                annotation["category_id"] = CATEGORY['vehicle']
                annotation["id"] = objid
                objid += 1
                annotation["segmentation"] = [[x, y, x, (y + h), (x + w), (y + h), (x + w), y]]
                annotations.append(annotation)
            else:
                annotation = dict()
                rect = objdict['rect']
                annotation["image_id"] = imgid
                annotation["ignore"] = 0
                annotation["iscrowd"] = 0
                x, y, w, h = RectDict2List(rect, imgwidth, imgheight, scale=1, mode='tlwh')
                annotation["bbox"] = [x, y, w, h]
                annotation["area"] = float(w * h)
                annotation["category_id"] = CATEGORY['vehicle']
                annotation["id"] = objid
                objid += 1
                annotation["segmentation"] = [[x, y, x, (y + h), (x + w), (y + h), (x + w), y]]
                annotations.append(annotation)

    attrDict["images"] = images
    attrDict["annotations"] = annotations
    attrDict["type"] = "instances"

    # print attrDict
    jsonString = json.dumps(attrDict, indent=2)
    with open(tgtfile, "w") as f:
        f.write(jsonString)

    return imageids

    
def generate_mot_anno(srcdir, tgtdir):
    """
    transfer ground truth to MOTChallenge format
    :param srcdir: root directory to source gt json file
    :param tgtdir: target directory
    :return:
    """
    print('transferring file format.')
    if not os.path.exists(tgtdir):
        os.makedirs(tgtdir)

    gtdirs = glob.glob(os.path.join(srcdir, '*/'))
    for gtdir in gtdirs:
        tracksfile = os.path.join(gtdir, 'tracks.json')
        seqinfofile = os.path.join(gtdir, 'seqinfo.json')
        if not os.path.exists(tracksfile):
            continue
        with open(tracksfile, 'r') as load_f:
            tracks_list = json.load(load_f)
        with open(seqinfofile, 'r') as load_f:
            seqinfo_dict = json.load(load_f)
        seqname = seqinfo_dict["name"]
        width = seqinfo_dict["imWidth"]
        height = seqinfo_dict["imHeight"]

        with open(os.path.join(tgtdir, seqname + '.txt'), 'w') as f:
            for track_dict in tracks_list:
                track_id = track_dict["track id"]
                for frame_dict in track_dict["frames"]:
                    frame_id = frame_dict["frame id"]
                    rect = frame_dict["rect"]
                    occ = frame_dict["occlusion"]
                    x, y, w, h = RectDict2List(rect, width, height, scale=1, mode='tlwh')
                    if occ == 'normal':
                        visible_ratio = 1
                    elif occ == 'hide':
                        visible_ratio = 0.66667
                    elif occ == 'serious_hide':
                        visible_ratio = 0.33333
                    else:
                        visible_ratio = 0
                    # save MOT file:
                    f.writelines([str(frame_id), ',', str(track_id), ',', str(x), ',', str(y), ',', str(w), ',',
                                  str(h), ',', '1', ',', '1', ',', str(visible_ratio), '\n'])


def generate_mot_res(srcdir, tgtdir):
    """
    generate results in MOTChallenge format from ground truth
    :param srcdir: root directory to source gt json file
    :param tgtdir: target directory
    :return:
    """
    print('generating results from gt.')
    if not os.path.exists(tgtdir):
        os.makedirs(tgtdir)

    gtdirs = glob.glob(os.path.join(srcdir, '*/'))
    for gtdir in gtdirs:
        tracksfile = os.path.join(gtdir, 'tracks.json')
        seqinfofile = os.path.join(gtdir, 'seqinfo.json')
        if not os.path.exists(tracksfile):
            continue
        with open(tracksfile, 'r') as load_f:
            tracks_list = json.load(load_f)
        with open(seqinfofile, 'r') as load_f:
            seqinfo_dict = json.load(load_f)
        seqname = seqinfo_dict["name"]
        width = seqinfo_dict["imWidth"]
        height = seqinfo_dict["imHeight"]
        seqLength = seqinfo_dict['seqLength']

        with open(os.path.join(tgtdir, seqname + '.txt'), 'w') as f:
            for i in range(seqLength):
                frame_id = i + 1
                for track_dict in tracks_list:
                    track_id = track_dict["track id"]
                    for frame_dict in track_dict["frames"]:
                        if frame_id == frame_dict["frame id"]:
                            rect = frame_dict["rect"]
                            x, y, w, h = RectDict2List(rect, width, height, scale=1, mode='tlwh')
                            x += random.gauss(0, 10)
                            y += random.gauss(0, 10)
                            w += random.gauss(0, 5)
                            h += random.gauss(0, 5)
                            # save MOT result file:
                            f.writelines([str(frame_id), ',', str(track_id), ',', str(x), ',', str(y), ',', str(w), ',',
                                          str(h), ',', '1', ',', '-1', ',', '-1', ',', '-1', '\n'])
