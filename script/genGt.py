# generate gt file based on siammask tracking result

import glob
import json
import torch
import cv2
import numpy as np
from os.path import join, isfile

from utils.load_helper import load_pretrain
from utils.tracker_config import TrackerConfig

from tracker import Tracker, tracker_init, tracker_track

def main(video_path):
    # setup config
    resume = "../SiamMask/experiments/siammask_sharp/SiamMask_DAVIS.pth"
    config_f = "../SiamMask/experiments/siammask_sharp/config_davis.json"
    config = json.load(open(config_f))
    p = TrackerConfig()
    p.renew()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    siammask = Tracker(p=p, anchors=config['anchors'])
    assert isfile(resume), 'Please download {} first.'.format(resume)
    siammask = load_pretrain(siammask, resume)
    siammask.eval().to(device)

    # Parse Image file
    img_files = sorted(glob.glob(join(join(video_path, 'img'), '*.jp*')))
    ims = [cv2.imread(imf) for imf in img_files]

    # Select ROI
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    try:
        init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
        gts = None
        x, y, w, h = init_rect
    except:
        exit()

    file1 = open(join(video_path, 'groundtruth.txt'), 'w') 
    file1.write('{0:d},{1:d},{2:d},{3:d}\n'.format(x, y, w, h))
    print('Gt file to: ', join(video_path, 'groundtruth.txt'))

    file2 = open(join(video_path, 'full_occlusion.txt'), 'w') 
    content = '0,' * (len(img_files)-1) + '0'
    file2.write(content)
    file2.close()
    print('Gt file to: ', join(video_path, 'full_occlusion.txt'))

    file3 = open(join(video_path, 'out_of_view.txt'), 'w') 
    content = '0,' * (len(img_files)-1) + '0'
    file3.write(content)
    file3.close()
    print('Gt file to: ', join(video_path, 'out_of_view.txt'))

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
            file1.write('{0:d},{1:d},{2:d},{3:d}\n'.format(x, y, x2-x, y2-y))
        toc += cv2.getTickCount() - tic
    file1.close() 

    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))

if __name__ == "__main__":
    import fire
    fire.Fire(main)
