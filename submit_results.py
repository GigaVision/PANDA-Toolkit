'''
依赖: pip install ai-hub
测试用例:
model 为 y=2*x
请求数据为 json:{"img":3}
#(version>=0.1.7)
-----------
post请求:
curl localhost:8080/tccapi -X POST -d '{"img":3}'
返回结果
'''
from ai_hub import inferServer
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import argparse
import os
from tracker.multitracker import JDETracker, JDE_PANDA_Tracker
from utils.datasets import letterbox
import logging
from utils.log import logger


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)

    print('save results to {}'.format(filename))


def sliding_window(src_giga_img):
    width = 1088
    height = 608
    overlap = 0.2
    x_step = round(width * (1 - overlap))
    y_step = round(height * (1 - overlap))
    scales = (0.2, 0.4)

    sub_img_list = []

    raw_height, raw_width = src_giga_img.shape[:2]
    for scale in scales:
        n = 0
        src_img = cv2.resize(src_giga_img, (int(raw_width * scale), int(raw_height * scale)))
        src_height, src_width = src_img.shape[:2]
        # sub image generate
        ini_y = 0
        while ini_y < src_height:
            ini_x = 0
            while ini_x < src_width:
                img0 = src_img[ini_y : ini_y + height, ini_x : ini_x + width]
                loc_info = (scale, ini_x, ini_y)
                # Padded resize
                img, _, _, _ = letterbox(img0, height=height, width=width)

                # Normalize RGB
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img, dtype=np.float32)
                img /= 255.0

                sub_img_list.append((img0, img, loc_info))

                n += 1
                if ini_x == src_width - width:
                    break
                ini_x += x_step
                if ini_x + width > src_width:
                    ini_x = src_width - width
            if ini_y == src_height - height:
                break
            ini_y += y_step
            if ini_y + height > src_height:
                ini_y = src_height - height

    return sub_img_list


class myInfer(inferServer):
    def __init__(self, model):
        super().__init__(model)
        self.opt = model
        self.model = model
        # self.opt = opt
        print("init_myInfer")
        # 数据前处理

    def pre_process(self, data):
        print("------------1-my_pre_process------------")
        flag = int(data.files['flag'].read().decode('utf-8'))
        dataset = data.files['dataset'].read().decode('utf-8')
        print(flag)

        if flag == 1:
            dataset = data.files['dataset'].read().decode('utf-8')
            img_id = data.files['id'].read().decode('utf-8')
            img_rb = data.files['img'].read()

            nparr = np.frombuffer(img_rb, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1

            print(img_id)
            print(img_np.shape)

            return (flag, dataset, img_id, img_np)
        else:
            return (flag, dataset, 0, 0)
        # 数据后处理

    def predict(self, data):
        print("----------2-predict----------")
        flag, dataset, img_id, img_np = data

        # ret = self.model(img_np)
        if flag == 0:
            print('2.1-----initial Tracker')
            self.tracker = JDE_PANDA_Tracker(self.opt, frame_rate=30)
            self.result_filename = os.path.join('./submit_results', dataset + '.txt')
            self.results = []
            self.frame_id = 0

            filedict = {'flag': flag}

        if flag == 1:
            print('2.2-------tracking....')
            filedict = {'flag': flag}

            print('Processing imgid = {}'.format(img_id))
            sub_img_list = sliding_window(img_np)
            online_targets = self.tracker.update(sub_img_list)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            self.results.append((self.frame_id + 1, online_tlwhs, online_ids))
            self.frame_id += 1

        if flag == 2:
            print('2.3------return tracking results')
            write_results(self.result_filename, self.results)

            with open(self.result_filename, 'r') as f:
                txt_results = f.readlines()
                filedict = {'flag': flag, 'results': txt_results}

        # filedict = {'flag': 1, 'results': 2}
        filedict = json.dumps(filedict)
        return filedict

    def post_process(self, data):
        print("--------3-post_process-------------")
        processed_data = data
        return processed_data


# class mymodel(nn.Module):
#     def __init__(self):
#         self.model = lambda x: x * 2

# def forward(self, x):
#     y = self.model(x)
#     print('y = {}'.format(y))
#     return y


if __name__ == "__main__":

    # mymodel = lambda x: x * 2
    # my_infer = myInfer(mymodel)
    # my_infer.run("127.0.0.1", 1234, debuge=True)  #
    # 默认为 ("127.0.0.1",80),可自定义端口

    parser = argparse.ArgumentParser(prog='track.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3_1088x608.cfg', help='cfg file path')
    parser.add_argument(
        '--weights', type=str, default='weights/jde.1088x608.uncertainty.pt', help='path to weights file'
    )
    parser.add_argument('--img-size', type=int, default=(1088, 608), help='size of each image dimension')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
    parser.add_argument('--min-box-area', type=float, default=50, help='filter out tiny boxes')
    parser.add_argument('--track-buffer', type=int, default=300, help='tracking buffer')
    parser.add_argument('--test-mot16', action='store_true', help='tracking buffer')
    opt = parser.parse_args()

    # mymodel = opt
    logger.setLevel(logging.INFO)
    my_infer = myInfer(opt)
    my_infer.run("127.0.0.1", 1234, debuge=True)  #
