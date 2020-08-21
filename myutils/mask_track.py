import glob
import argparse
import json
import torch
import cv2
import numpy as np
from os.path import join, isdir, isfile

from utils.config_helper import load_config
from utils.load_helper import load_pretrain
from utils.tracker_config import TrackerConfig

from tracker import Tracker, tracker_init, tracker_track
from myutils.pysot_track import get_frames


parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')
parser.add_argument('--video_name', default='', help='datasets')
args = parser.parse_args()

if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup tracker config
    resume = "../SiamMask/experiments/siammask_sharp/SiamMask_DAVIS.pth"
    config_f = "../SiamMask/experiments/siammask_sharp/config_davis.json"
    config = json.load(open(config_f))
    p = TrackerConfig()
    p.renew()
    smooth_lr = p.lr

    # create model
    siammask = Tracker(p=p, anchors=config['anchors'])

    # load model
    assert isfile(resume), 'Please download {} first.'.format(resume)
    siammask = load_pretrain(siammask, resume)
    siammask.eval().to(device)

    # Tracking
    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    cv2.namedWindow('tracking', cv2.WND_PROP_FULLSCREEN)
    for im in get_frames(args.video_name):
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, im, False, False)
                x, y, w, h = init_rect
            except:
                continue
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = tracker_init(im, target_pos, target_sz, siammask, device=device)  # init tracker
            state['device'] = device
            first_frame = False
        else:
            state = tracker_track(state, im, siammask, device=device)  # track
            target_pos, target_sz =state['target_pos'], state['target_sz']
            x, y = (target_pos - target_sz/2).astype(int)
            x2, y2 = (target_pos + target_sz/2).astype(int)
            cv2.rectangle(im, (x, y), (x2, y2), (0, 255, 0), 4)
            cv2.imshow('tracking', im)
            key = cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('r'):
                first_frame = True
            if key == ord('q'):
                break
            elif key == ord('r'):
                first_frame = True
