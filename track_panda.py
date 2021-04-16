import os
import os.path as osp
import cv2
import logging
import argparse

from tracker.multitracker import JDETracker, JDE_PANDA_Tracker
from utils import visualization as vis
from utils.log import logger
from utils.timer import Timer
from utils.evaluation import Evaluator
import utils.datasets as datasets
import torch
from utils.utils import *
import zipfile


def make_zip(source_dir, output_filename):
    zipf = zipfile.ZipFile(output_filename, 'w')
    pre_len = len(os.path.dirname(source_dir))
    for parent, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)  # 相对路径
            zipf.write(pathfile, arcname)
    zipf.close()
    print('save in {}'.format(output_filename))


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_panda_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30):
    if save_dir:
        mkdir_if_missing(save_dir)

    tracker = JDE_PANDA_Tracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    for path, src_giga_img, sub_img_list in dataloader:
        if frame_id % 5 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1.0 / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        online_targets = tracker.update(sub_img_list)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        timer.toc()
        print('Total cost time: {} s'.format(timer.interval()))
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking_panda(
                src_giga_img, online_tlwhs, online_ids, frame_id=frame_id, fps=1.0 / timer.average_time, scales=0.2
            )
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main_for_panda(
    opt,
    data_root='test_frames',
    det_root=None,
    seqs=('IMG',),
    exp_name='demo',
    save_images=False,
    save_videos=False,
    show_image=True,
):
    logger.setLevel(logging.INFO)
    # result_root = os.path.join('results', exp_name)
    result_root = 'results'
    output_root = 'vis_results'

    mkdir_if_missing(result_root)
    mkdir_if_missing(output_root)

    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []

    logger.info('start seq: {}'.format(exp_name))

    dataloader = datasets.LoadPandaImages(data_root, opt.img_size)
    result_filename = os.path.join(result_root, '{}.txt'.format(exp_name))
    output_dir = os.path.join(output_root, exp_name) if save_images or save_videos else None

    frame_rate = 30

    nf, ta, tc = eval_panda_seq(
        opt,
        dataloader,
        data_type,
        result_filename,
        save_dir=output_dir,
        show_image=show_image,
        frame_rate=frame_rate,
    )
    n_frame += nf
    timer_avgs.append(ta)
    timer_calls.append(tc)

    if save_videos:
        output_video_path = osp.join(output_dir, '{}.mp4'.format(exp_name))
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
        os.system(cmd_str)

    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))


if __name__ == '__main__':
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
    print(opt, end='\n\n')

    '''change the root setting here for offline test'''
    # root = 'YOUR/PATH'
    root = '/tcdata'

    scenes = [
        'panda_round2_test_20210331_A_part1/11_Train_Station_Square',
        'panda_round2_test_20210331_A_part2/12_Nanshan_i_Park',
        'panda_round2_test_20210331_A_part3/13_University_Playground',
    ]

    for scene in scenes:
        data_root = os.path.join(root, scene)
        main_for_panda(
            opt,
            data_root=data_root,
            exp_name=data_root.split('/')[-1],
            show_image=False,
            save_images=False,
            save_videos=False,
        )
    
    # 打包为提交格式results.zip
    make_zip('results','results.zip')

