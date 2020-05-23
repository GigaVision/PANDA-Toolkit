# --------------------------------------------------------
# Result merge modules for PANDA
# Written by Wang Xueyang  (wangxuey19@mails.tsinghua.edu.cn), Version 20200523
# Inspired from DOTA dataset devkit (https://github.com/CAPTAIN-WHU/DOTA_devkit)
# --------------------------------------------------------

import os
import numpy as np
import panda_utils as util
import json
from collections import defaultdict


class DetResMerge():
    def __init__(self,
                 basepath,
                 resfile,
                 splitannofile,
                 srcannofile,
                 outpath,
                 outfile,
                 imgext='.jpg',
                 code='utf-8',
                 ):
        """
        :param basepath: base directory for panda image data and annotations
        :param resfile: detection result file path
        :param splitannofile: generated split annotation file
        :param srcannofile: source annotation file
        :param resmode: detection result mode, which can be 'person', 'vehicle', 'headbbox' or 'headpoint'
        :param outpath: output base path for merged result file
        :param outfile: name for merged result file
        :param imgext: ext for the split image format
        """
        self.basepath = basepath
        self.resfile = resfile
        self.splitannofile = splitannofile
        self.srcannofile = srcannofile
        self.outpath = outpath
        self.outfile = outfile
        self.imgext = imgext
        self.code = code
        self.imgpath = os.path.join(self.basepath, 'PANDA_IMAGE')
        self.respath = os.path.join(self.basepath, 'results', resfile)
        self.splitannopath = os.path.join(self.basepath, 'image_annos', splitannofile)
        self.srcannopath = os.path.join(self.basepath, 'image_annos', srcannofile)
        self.imagepaths = util.GetFileFromThisRootDir(self.imgpath, ext='jpg')
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)
        self.results = defaultdict(list)
        self.indexResults()

    def indexResults(self):
        print('Loading result json file: {}'.format(self.respath))
        with open(self.respath, 'r') as load_f:
            reslist = json.load(load_f)
        print('Loading split annotation json file: {}'.format(self.splitannopath))
        with open(self.splitannopath, 'r') as load_f:
            splitanno = json.load(load_f)
        indexedresults = defaultdict(list)
        for (filename, annodict) in splitanno.items():
            imageid = annodict['image id']
            for resdict in reslist:
                resimageid = resdict['image_id']
                if resimageid == imageid:
                    indexedresults[filename].append(resdict)
        self.results = indexedresults

    def mergeResults(self, is_nms=True, nms_thresh=0.5):
        """
        :param is_nms: do non-maximum suppression on after merge
        :param nms_thresh: non-maximum suppression IoU threshold
        :return:
        """
        print('Loading source annotation json file: {}'.format(self.srcannopath))
        with open(self.srcannopath, 'r') as load_f:
            srcanno = json.load(load_f)
        mergedresults = defaultdict(list)
        for (filename, objlist) in self.results.items():
            srcfile, paras = filename.split('___')
            srcfile = srcfile.replace('_IMG', '/IMG') + self.imgext
            srcimageid = srcanno[srcfile]['image id']
            scale, left, up = paras.replace(self.imgext, '').split('__')
            for objdict in objlist:
                mergedresults[srcimageid].append([*recttransfer(objdict['bbox'], float(scale), int(left), int(up)),
                      objdict['score'], objdict['category_id']])
        if is_nms:
            for (imageid, objlist) in mergedresults.items():
                keep = py_cpu_nms(np.array(objlist), nms_thresh)
                outdets = []
                for index in keep:
                    outdets.append(objlist[index])
                mergedresults[imageid] = outdets
        savelist = []
        for (imageid, objlist) in mergedresults.items():
            for obj in objlist:
                savelist.append({
                    "image_id": imageid,
                    "category_id": obj[5],
                    "bbox": tlbr2tlwh(obj[:4]),
                    "score": obj[4]
                })
        with open(os.path.join(self.outpath, self.outfile), 'w', encoding=self.code) as f:
            dict_str = json.dumps(savelist, indent=2)
            f.write(dict_str)


def recttransfer(rect, scale, left, up):
    xmin, ymin, w, h = rect
    xmax, ymax = xmin + w, ymin + h
    return [int(temp / scale) for temp in [xmin + left, ymin + up, xmax + left, ymax + up]]


def tlbr2tlwh(rect):
    xmin, ymin, xmax, ymax = rect
    w, h = xmax - xmin, ymax - ymin
    return [xmin, ymin, w, h]


def py_cpu_nms(dets, thresh):
    #print('dets:', dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    ## index for dets
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
