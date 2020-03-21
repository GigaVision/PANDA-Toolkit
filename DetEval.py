# --------------------------------------------------------
# Compute metrics for detectors using ground-truth data
# Written by Wang Xueyang (wangxuey19@mails.tsinghua.edu.cn), Version 20200321
# Based on pycocotools (https://github.com/cocodataset/cocoapi/)
# --------------------------------------------------------

import numpy as np
import argparse
from panda_utils import generate_coco_anno
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="""
Compute metrics for detectors using ground-truth data.

Files
-----
All result files have to comply with the COCO format described in
http://cocodataset.org/#format-results
Structure
---------

Layout for ground truth data
    <GT_ROOT>/anno.json'

Layout for test data
    <TEST_ROOT>/results.json
""", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('personfile', type=str, help='File path to person annotation json file')
    parser.add_argument('vehiclefile', type=str, help='File path to vehicle annotation json file')
    parser.add_argument('result', type=str, help='File path to result json file.')
    parser.add_argument('--transfered', type=str, help='Directory containing transfered gt files', default='transfered.json')
    parser.add_argument('--annType', type=str, help='annotation type', default='bbox')
    parser.add_argument('--maxDets', type=list, help='[10, 100, 500] M = 3 thresholds on max detections per image',
                        default=[10, 100, 500])
    parser.add_argument('--areaRng', type=list, help='[...] A = 4 object area ranges for evaluation',
                        default=[0, 200, 400, 1e5])
    return parser.parse_args()


def main():
    args = parse_args()

    # transfer ground truth to COCO format
    generate_coco_anno(args.personfile, args.vehiclefile, args.transfered)

    # initialize COCO ground truth api
    cocoGt = COCO(args.transfered)

    # initialize COCO detections api
    cocoDt = cocoGt.loadRes(args.result)

    imgIds = sorted(cocoGt.getImgIds())
    # random.shuffle(imgIds)

    cocoEval = COCOeval(cocoGt, cocoDt, args.annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.params.maxDets = args.maxDets
    thres0, thres1, thres2, thres3 = args.areaRng
    cocoEval.params.areaRng = [[thres0 ** 2, thres3 ** 2],
                               [thres0 ** 2, thres1 ** 2],
                               [thres1 ** 2, thres2 ** 2],
                               [thres2 ** 2, thres3 ** 2]]

    cocoEval.evaluate()
    cocoEval.accumulate()
    summarize(cocoEval)

    '''we use AP, AP_{IOU = 0.50}, AP_{IOU = 0.75}, AR_{max = 10}, AR_{max = 100}, AR_{max = 500} to evaluate'''
    AP, AP_05, AP_075, _, _, _, AR_10, AR_100, AR_500, _, _, _ = cocoEval.stats
    print(AP, AP_05, AP_075, AR_10, AR_100, AR_500)


def summarize(self):
    '''
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    '''
    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap==1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,:,aind,mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,aind,mind]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
        return mean_s
    def _summarizeDets():
        stats = np.zeros((12,))
        stats[0] = _summarize(1, maxDets=self.params.maxDets[2])
        stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
        stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
        stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
        stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
        stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
        stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
        stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
        stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
        stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
        return stats
    if not self.eval:
        raise Exception('Please run accumulate() first')
    iouType = self.params.iouType
    if iouType == 'bbox':
        summarize = _summarizeDets
    self.stats = summarize()


if __name__ == '__main__':
    main()
