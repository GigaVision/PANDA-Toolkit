# --------------------------------------------------------
# Visualization modules for PANDA
# Written by Wang Xueyang  (wangxuey19@mails.tsinghua.edu.cn), Version 20200523
# Inspired from DOTA dataset devkit (https://github.com/CAPTAIN-WHU/DOTA_devkit)
# --------------------------------------------------------

import os
import cv2
import json
import random
from collections import defaultdict

import panda_utils as util

IMAGE_ANNO_MODE = ('person', 'vehicle', 'person&vehicle', 'headbbox', 'headpoint')


class PANDA_IMAGE:
    def __init__(self, basepath, annofile, annomode, extraannofile=None, showwidth=1280):
        """
        :param basepath: base directory for panda image data and annotations
        :param annofile: annotation file path
        :param annomode: the type of annotation, which can be 'person', 'vehicle', 'person&vehicle', 'headbbox' or 'headpoint'
        :param extraannofile: if you want to show person and vehicle annotations simultaneously,
                                choose annomode 'person&vehicle' and send vehicle annotation file path into extraannofile
        :param showwidth: the width of visualized image
        """
        assert annomode in IMAGE_ANNO_MODE, 'Annotation mode must be \'person\', \'vehicle\', \'person&vehicle\', \'headbox\' or \'headpoint\''
        self.annomode = annomode
        self.basepath = basepath
        self.annofile = annofile
        self.extraannofile = extraannofile
        self.showwidth = showwidth
        self.imagepath = os.path.join(basepath, 'image_train')
        self.annopath = os.path.join(basepath, 'image_annos', annofile)
        if extraannofile:
            self.extraannopath = os.path.join(basepath, 'image_annos', extraannofile)
        self.imgpaths = util.GetFileFromThisRootDir(self.imagepath, ext='jpg')
        self.annos = defaultdict(list)
        self.extraannos = defaultdict(list)
        self.createIndex()

    def createIndex(self):
        if self.annomode == 'person&vehicle':
            annos = util.parse_panda_rect(self.annopath, 'person', self.showwidth)
            extraannos = util.parse_panda_rect(self.extraannopath, 'vehicle', self.showwidth)
            self.annos = annos
            self.extraannos = extraannos
        else:
            annos = util.parse_panda_rect(self.annopath, self.annomode, self.showwidth)
            self.annos = annos

    def showImgs(self, imgrequest=None, range=10, imgfilters=[], shuffle=True):
        """
        :param imgrequest: list, images names you want to request, eg. ['1-HIT_canteen/IMG_1_4.jpg', ...]
        :param range: number of image to show
        :param imgfilters: essential keywords in image name
        :param shuffle: shuffle all image
        :return:
        """
        if imgrequest is None or not isinstance(imgrequest, list):
            allnames = list(self.annos.keys())
            imgnames = [] if imgfilters else allnames
            if imgfilters:
                for imgname in allnames:
                    iskeep = False
                    for imgfilter in imgfilters:
                        if imgfilter in imgname:
                            iskeep = True
                    if iskeep:
                        imgnames.append(imgname)
            if shuffle:
                random.shuffle(imgnames)
            if range:
                if isinstance(range, int) and range <= len(imgnames):
                    imgnames = imgnames[:range]
        else:
            imgnames = imgrequest

        for imgname in imgnames:
            imgpath = os.path.join(self.imagepath, imgname)
            img = self.loadImg(imgpath)
            if img is None:
                continue
            cv2.putText(img, 'Press any button to continue', (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(util.custombasename(imgname), img)
            cv2.waitKey(0)

    def showAnns(self, imgrequest=None, range=10, imgfilters=[], shuffle=True, saveimg=False):
        """
        :param imgrequest: list, images names you want to request, eg. ['1-HIT_canteen/IMG_1_4.jpg', ...]
        :param range: number of image to show
        :param imgfilters: essential keywords in image name
        :param shuffle: shuffle all image
        :return:
        """
        savedir = 'results/image'
        if saveimg and not os.path.exists(savedir):
            os.makedirs(savedir)

        if imgrequest is None or not isinstance(imgrequest, list):
            allnames = list(self.annos.keys())
            imgnames = [] if imgfilters else allnames
            if imgfilters:
                for imgname in allnames:
                    iskeep = False
                    for imgfilter in imgfilters:
                        if imgfilter in imgname:
                            iskeep = True
                    if iskeep:
                        imgnames.append(imgname)
            if shuffle:
                random.shuffle(imgnames)
            if range:
                if isinstance(range, int) and range <= len(imgnames):
                    imgnames = imgnames[:range]
        else:
            imgnames = imgrequest

        for imgname in imgnames:
            imgpath = os.path.join(self.imagepath, imgname)
            img = self.loadImg(imgpath)
            if img is None:
                continue
            if self.annomode == 'person':
                imgwithann = self._addPersonAnns(imgname, img)
            elif self.annomode == 'vehicle':
                imgwithann = self._addVehicleAnns(imgname, img)
            elif self.annomode == 'person&vehicle':
                imgwithann = self._addPersonVehicleAnns(imgname, img)
            elif self.annomode == 'headbbox':
                imgwithann = self._addHeadbboxAnns(imgname, img)
            elif self.annomode == 'headpoint':
                imgwithann = self._addHeadpointAnns(imgname, img)

            if saveimg:
                cv2.imwrite(os.path.join(savedir, util.custombasename(imgname) + '.jpg'), imgwithann)
            cv2.putText(img, 'Press any button to continue', (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('image_with_anno', imgwithann)  # image_with_anno --> util.custombasename(imgname)
            cv2.waitKey(0)

    def loadImg(self, imgpath):
        """
        :param imgpath: the path of image to load
        :return: loaded img object
        """
        print('filename:', imgpath)
        if not os.path.exists(imgpath):
            print('Can not find {}, please check local dataset!'.format(imgpath))
            return None
        img = cv2.imread(imgpath)
        imgheight, imgwidth = img.shape[:2]
        scale = self.showwidth / imgwidth
        img = cv2.resize(img, (int(imgwidth * scale), int(imgheight * scale)))

        return img

    def _addPersonAnns(self, imgname, img, showcate=False):
        objlist = self.annos[imgname]
        for objdict in objlist:
            cate = objdict['cate']
            if objdict['ignore']:
                xmin, ymin, xmax, ymax = objdict['rect']
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                cv2.line(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                cv2.line(img, (xmin, ymax), (xmax, ymin), (0, 0, 255), 1)
                if showcate:
                    cv2.putText(img, cate, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                b = random.randint(0, 255)
                g = random.randint(0, 255)
                r = random.randint(0, 255)
                for rect in [objdict['fullrect'], objdict['visiblerect'], objdict['headrect']]:
                    xmin, ymin, xmax, ymax = rect
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (b, g, r), 1)
                xmin, ymin, _, _ = objdict['fullrect']
                if showcate:
                    cv2.putText(img, cate, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 1)
        return img

    def _addVehicleAnns(self, imgname, img, showcate=False):
        objlist = self.annos[imgname]
        for objdict in objlist:
            cate = objdict['cate']
            xmin, ymin, xmax, ymax = objdict['rect']
            b = random.randint(0, 255)
            g = random.randint(0, 255)
            r = random.randint(0, 255)
            if objdict['ignore']:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                cv2.line(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                cv2.line(img, (xmin, ymax), (xmax, ymin), (0, 0, 255), 1)
            else:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (b, g, r), 1)
                if showcate:
                    cv2.putText(img, cate, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 1)
        return img

    def _addPersonVehicleAnns(self, imgname, img, showcate=False):
        personobjlist = self.annos[imgname]
        vehicleobjlist = self.extraannos[imgname]
        for objdict in personobjlist:
            cate = objdict['cate']
            if objdict['ignore']:
                xmin, ymin, xmax, ymax = objdict['rect']
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                cv2.line(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                cv2.line(img, (xmin, ymax), (xmax, ymin), (0, 0, 255), 1)
                if showcate:
                    cv2.putText(img, cate, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                b = random.randint(0, 255)
                g = random.randint(0, 255)
                r = random.randint(0, 255)
                for rect in [objdict['fullrect'], objdict['visiblerect'], objdict['headrect']]:
                    xmin, ymin, xmax, ymax = rect
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (b, g, r), 1)
                xmin, ymin, _, _ = objdict['fullrect']
                if showcate:
                    cv2.putText(img, cate, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 1)
        for objdict in vehicleobjlist:
            cate = objdict['cate']
            xmin, ymin, xmax, ymax = objdict['rect']
            b = random.randint(0, 255)
            g = random.randint(0, 255)
            r = random.randint(0, 255)
            if objdict['ignore']:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                cv2.line(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                cv2.line(img, (xmin, ymax), (xmax, ymin), (0, 0, 255), 1)
            else:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (b, g, r), 1)
                if showcate:
                    cv2.putText(img, cate, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 1)
        return img

    def _addHeadbboxAnns(self, imgname, img):
        objlist = self.annos[imgname]
        for rect in objlist:
            xmin, ymin, xmax, ymax = rect
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
        return img

    def _addHeadpointAnns(self, imgname, img):
        objlist = self.annos[imgname]
        for point in objlist:
            x, y = point
            cv2.circle(img, (x, y), radius=1, color=(0, 0, 255), thickness=-1)
        return img


PANDA_VIDEO_SEQS = [
    '1-HIT_Canteen_frames',
    '2-OCT_Bay_frames',
    '7-Shenzhennorth_Station_frames',
    '3-Xili_Crossroad_frames',
    '4-Nanshan_I_Park_frames',
    '8-Xili_Pedestrian_Street_frames',
    '5-Primary_School_frames',
    '9-Tsinghuasz_Basketball_Court_frames',
    '10-Xinzhongguan_frames',
    '12-Tsinghua_Zhulou_frames',
    '13-Tsinghua_Xicao_frames',
    '11-Tsinghua_ceremony_frames',
    '16-Xili_frames'
]


class PANDA_VIDEO:
    def __init__(self, basepath, savepath, videowidth=1280):
        """
        :param basepath: base directory for panda video data and annotations
        :param savepath: video save root path
        :param videowidth: the width of visualized video
        """
        self.basepath = basepath
        self.savepath = savepath
        self.videowidth = videowidth
        self.seqspath = os.path.join(basepath, 'video_test')
        self.annopath = os.path.join(basepath, 'video_annos')
        self.annofile = 'tracks.json'
        self.seqinfofile = 'seqinfo.json'
        self.seqnames = PANDA_VIDEO_SEQS

    def saveVideo(self, videorequest=None, withanno=True, maxframe=None):
        """
        :param maxframe: maximum frame number for each video
        :param withanno: add annotation on video to save or not
        :param videorequest: list, sequence names you want to request, eg. ['1-HIT_Canteen_frames', ...]
        :return:
        """
        if videorequest is None or not isinstance(videorequest, list):
            seqnames = self.seqnames
        else:
            seqnames = videorequest

        for seqname in seqnames:
            framespath = os.path.join(self.seqspath, seqname)
            annopath = os.path.join(self.annopath, seqname, self.annofile)
            seqinfopath = os.path.join(self.annopath, seqname, self.seqinfofile)

            print('Loading annotation json file: {}'.format(annopath))
            with open(annopath, 'r') as load_f:
                anno = json.load(load_f)
            with open(seqinfopath, 'r') as load_f:
                seqinfo = json.load(load_f)
            framerate = seqinfo["frameRate"]
            width = seqinfo["imWidth"]
            height = seqinfo["imHeight"]
            frames = seqinfo["imUrls"]

            save_height = int(self.videowidth / width * height)

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(os.path.join(self.savepath, seqname + '.avi'), fourcc, framerate,
                                  (self.videowidth, save_height), True)

            for i, frame in enumerate(frames):
                print('writing frame {} to video.'.format(frame))
                imgpath = os.path.join(framespath, frame)
                img = cv2.imread(imgpath)
                img = cv2.resize(img, (self.videowidth, save_height))
                # TODO: add tracking results
                if withanno:
                    img = self.addanno(img, i + 1, anno, (self.videowidth, save_height))
                cv2.imwrite(os.path.join(self.savepath, str(i) + '.jpg'), img)
                out.write(img)
                if isinstance(maxframe, int) and i + 1 == maxframe:
                    break
            out.release()

    def addanno(self, img, frameid, anno, savesize, showpid=True):
        savewidth, saveheight = savesize
        for track in anno:
            for frame in track['frames']:
                if frame["frame id"] == frameid:
                    pid = track['track id']
                    color = genColorByPid(pid)
                    xmin = int(frame["rect"]["tl"]["x"] * savewidth)
                    ymin = int(frame["rect"]["tl"]["y"] * saveheight)
                    xmax = int(frame["rect"]["br"]["x"] * savewidth)
                    ymax = int(frame["rect"]["br"]["y"] * saveheight)
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 1)
                    if showpid:
                        cv2.putText(img, str(pid), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img


COLORS_10 =[(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),(210,105, 30),(220, 20, 60),
            (192,192,192),(255,228,196),( 50,205, 50),(139,  0,139),(100,149,237),(138, 43,226),(238,130,238),
            (255,  0,255),(  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),(255,239,213),
            (199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),(176,196,222),( 65,105,225),(173,255, 47),
            (255, 20,147),(219,112,147),(186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
            (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),(255,255,224),(128,128,128),
            (105,105,105),( 64,224,208),(205,133, 63),(  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),
            (250,240,230),(152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),(  0,250,154),
            (245,255,250),(240,230,140),(245,222,179),(  0,139,139),(143,188,143),(255,  0,  0),(240,128,128),
            (102,205,170),( 60,179,113),( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
            (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]


def genColorByPid(pid):
    return COLORS_10[pid % len(COLORS_10)]