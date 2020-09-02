from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import imagezmq
import torch
import numpy as np
from glob import glob
from pathlib import Path

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str, help='videos or image files')
parser.add_argument('--rotate', default=False, type=bool, help='rotate video')
parser.add_argument('--save_result', default=False, type=bool, help='save tracking result')
args = parser.parse_args()


def get_frames(video_name, rotate=False):
    if not video_name:
        image_hub = imagezmq.ImageHub(open_port='tcp://*:6006')
        while True:
            rpi_name, frame = image_hub.recv_image()
            image_hub.send_reply(b'OK')
            yield frame
    elif video_name.endswith('avi') or video_name.endswith('mp4')  or video_name.endswith('MP4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                if rotate:
                    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE) 
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)

    # mk result dir
    if args.save_result:
        dir_path = Path(args.video_name)
        Path(dir_path.parents[0]).joinpath('result').mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        v_path = str(dir_path.parents[0])+'/result.avi'
        videoWriter = cv2.VideoWriter(v_path, fourcc, 29, (720, 1280))

    c = 1
    first_frame = True
    for frame in get_frames(args.video_name, args.rotate):
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                continue
            tracker.init(frame, init_rect)
            first_frame = False
            if args.save_result:
                x, y, w, h = init_rect
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 8)
                font = cv2.FONT_HERSHEY_SIMPLEX 
                org = (50, 50) 
                fontScale = 2
                color = (0, 0, 255) 
                thickness = 3
                cv2.putText(frame, 'Initialization', org, font,  fontScale, color, thickness, cv2.LINE_AA) 
                for i in range(10):
                    videoWriter.write(frame)
        else:
            outputs = tracker.track(frame)
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 4)
            cv2.imshow(video_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('r'):
                first_frame = True
                print('re-initialize at frame:', c)

        c = c + 1
        if args.save_result:
            f_path = str(dir_path.parents[0])+'/result/' + '{0:05d}'.format(c) + '.jpg'
            out = cv2.imwrite(f_path, frame)
            videoWriter.write(frame)
            print(out, c, f_path)
            

    if args.save_result:
        videoWriter.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()