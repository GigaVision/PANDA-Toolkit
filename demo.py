# --------------------------------------------------------
# Tool kit function demonstration
# Written by Wang Xueyang (wangxuey19@mails.tsinghua.edu.cn), Version 20200321
# Inspired from DOTA dataset devkit (https://github.com/CAPTAIN-WHU/DOTA_devkit)
# --------------------------------------------------------

from PANDA import PANDA_IMAGE, PANDA_VIDEO
import panda_utils as util
from ImgSplit import ImgSplit
from ResultMerge import DetResMerge

if __name__ == '__main__':
    image_root = 'G:/PANDA/PANDA_IMAGE'
    person_anno_file = 'human_bbox_train.json'
    annomode = 'person'
    example = PANDA_IMAGE(image_root, person_anno_file, annomode='person')

    '''1. show images'''
    example.showImgs()

    '''2. show annotations'''
    example.showAnns(range=50, shuffle=True)

    '''
    3. Split Image And Label
    We provide the scale param before split the images and labels.
    '''
    outpath = 'split'
    outannofile = 'split.json'
    split = ImgSplit(image_root, person_anno_file, annomode, outpath, outannofile)
    split.splitdata(0.5)

    '''
    4. Merge patches
    Now, we will merge these patches to see if they can be restored in the initial large images
    '''
    util.GT2DetRes('split/annoJSONs/split.json', 'split/resJSONs/res.json')
    merge = DetResMerge('split', 'res.json', 'split.json', 'human_bbox_all.json', 'results', 'mergetest.json')
    merge.mergeResults(is_nms=False)
    util.DetRes2GT('results/mergetest.json', 'G:/annoJSONs/mergegt.json', 'G:/annoJSONs/human_bbox_all.json')

    '''show merged results'''
    example = PANDA_IMAGE(image_root, 'mergegt.json', annomode='vehicle')
    example.showAnns()

    '''5. PANDA video visualization test'''
    video_root = 'G:/PANDA/PANDA_VIDEO'
    video_savepath = 'results'
    request = ['02_OCT_Habour']
    example = PANDA_VIDEO(video_root, video_savepath)

    '''save video'''
    example.saveVideo(videorequest=request, maxframe=100)
