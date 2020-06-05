import glob
import argparse
import torch
import cv2
import numpy as np
from os.path import join, isdir, isfile

from utils.config_helper import load_config
from utils.load_helper import load_pretrain
from utils.tracker_config import TrackerConfig

from tracker import Tracker, tracker_init, tracker_track


parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')
parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
parser.add_argument('--gt_file', default=None, type=str, help='ground truth txt file')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    p = TrackerConfig()
    p.renew()
    siammask = Tracker(p=p, anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)
    siammask.eval().to(device)

    # Parse Image file
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    ims = [cv2.imread(imf) for imf in img_files]

    # Select ROI
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    x, y, w, h = 305, 112, 163, 253
    if args.gt_file:
        with open(args.base_path + '/../' + args.gt_file, "r") as f:
            gts = f.readlines()
            split_flag = ',' if ',' in gts[0] else '\t'
            gts = list(map(lambda x: list(map(int, x.rstrip().split(split_flag))), gts))
            x, y, w, h = gts[0]
    else:
        try:
            init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
            gts = None
            x, y, w, h = init_rect
        except:
            exit()

    toc = 0
    for f, im in enumerate(ims):
        tic = cv2.getTickCount()
        if f == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = tracker_init(im, target_pos, target_sz, siammask, device=device)  # init tracker
            state['gts'] = gts
            state['device'] = device
        elif f > 0:  # tracking
            state = tracker_track(state, im, siammask, device=device)  # track
            target_pos, target_sz =state['target_pos'], state['target_sz']
            x, y = (target_pos - target_sz/2).astype(int)
            x2, y2 = (target_pos + target_sz/2).astype(int)
            cv2.rectangle(im, (x, y), (x2, y2), (0, 255, 0), 4)
            cv2.imshow('SiamMask', im)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
