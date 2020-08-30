import glob
import argparse
import json
import torch
import cv2
import numpy as np

from os.path import join, isdir, isfile
from pathlib import Path

from utils.config_helper import load_config
from utils.load_helper import load_pretrain
from utils.tracker_config import TrackerConfig

from tracker import Tracker, tracker_init, tracker_track
from myutils.pysot_track import get_frames


parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')
parser.add_argument('--video_name', default='', help='datasets')
parser.add_argument('--rotate', default=False, type=bool, help='rotate video')
parser.add_argument('--save_result', default=False, type=bool, help='save tracking result')
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

    # mk result dir
    if args.save_result:
        dir_path = Path(args.video_name)
        Path(dir_path.parents[0]).joinpath('result').mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        v_path = str(dir_path.parents[0])+'/result.avi'
        videoWriter = cv2.VideoWriter(v_path, fourcc, 29, (720, 1280))

    # Tracking
    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    cv2.namedWindow('tracking', cv2.WND_PROP_FULLSCREEN)
    c = 1
    for im in get_frames(args.video_name, args.rotate):
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
            if args.save_result:
                cv2.rectangle(im, (x, y), (x+w, y+h), (0, 0, 255), 8)
                font = cv2.FONT_HERSHEY_SIMPLEX 
                org = (50, 50) 
                fontScale = 2
                color = (0, 0, 255) 
                thickness = 3
                cv2.putText(im, 'Initialization', org, font,  fontScale, color, thickness, cv2.LINE_AA) 
                for i in range(10):
                    videoWriter.write(im)
        else:
            state = tracker_track(state, im, siammask, device=device)  # track
            target_pos, target_sz =state['target_pos'], state['target_sz']
            x, y = (target_pos - target_sz/2).astype(int)
            x2, y2 = (target_pos + target_sz/2).astype(int)
            cv2.rectangle(im, (x, y), (x2, y2), (0, 255, 0), 8)
            cv2.imshow('tracking', im)
            key = cv2.waitKey(1)
            if key == ord('r'):
                first_frame = True
            elif key == ord('q'):
                break


        c = c + 1
        if args.save_result:
            f_path = str(dir_path.parents[0])+'/result/' + '{0:05d}'.format(c) + '.jpg'
            out = cv2.imwrite(f_path, im)
            videoWriter.write(im)
            print(out, c, f_path)
            

    if args.save_result:
        videoWriter.release()
    cv2.destroyAllWindows()
    
