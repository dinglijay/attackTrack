import glob
import argparse
import torch
import cv2
import kornia
import numpy as np
from os.path import join, isdir, isfile
from torch.utils.data import DataLoader

from utils.config_helper import load_config
from utils.load_helper import load_pretrain
from utils.tracker_config import TrackerConfig

from tracker import Tracker, tracker_init, tracker_track
from attack_dataset import AttackDataset


parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')
parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
parser.add_argument('--gt_file', default=None, type=str, help='ground truth txt file')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()


def bbox2center_sz(bbox):
    x, y, w, h = bbox
    pos = np.array([x + w / 2, y + h / 2])
    sz = np.array([w, h])
    return pos, sz

def track(model, p, template_img, template_bbox, search_img, search_bbox):
    pos_z, size_z = bbox2center_sz(template_bbox)
    pos_x, size_x = bbox2center_sz(search_bbox)

    model.template(kornia.image_to_tensor(template_img).to(device).float(), 
                   torch.from_numpy(pos_z).to(device),
                   torch.from_numpy(size_z).to(device))

    pscore, delta, pscore_size = model.track(kornia.image_to_tensor(search_img).to(device).float(),
                                                torch.from_numpy(pos_x).to(device),
                                                torch.from_numpy(size_x).to(device))

    wc_x = size_x[1] + p.context_amount * sum(size_x)
    hc_x = size_x[0] + p.context_amount * sum(size_x)
    scale_x = p.exemplar_size / np.sqrt(wc_x * hc_x)

    best_pscore_id = np.argmax(pscore.squeeze().detach().cpu().numpy())

    pred_in_img = delta.squeeze().detach().cpu().numpy()[:, best_pscore_id] / scale_x
    lr = pscore_size.squeeze().detach().cpu().numpy()[best_pscore_id] * p.lr  # lr for OTB

    res_x = pred_in_img[0] + pos_x[0]
    res_y = pred_in_img[1] + pos_x[1]
    res_w = size_x[0] * (1 - lr) + pred_in_img[2] * lr
    res_h = size_x[1] * (1 - lr) + pred_in_img[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])

    im_h, im_w = template_img.shape[0], template_img.shape[1]
    target_pos[0] = max(0, min(im_w, target_pos[0]))
    target_pos[1] = max(0, min(im_h, target_pos[1]))
    target_sz[0] = max(10, min(im_w, target_sz[0]))
    target_sz[1] = max(10, min(im_h, target_sz[1]))

    x, y, w, h = template_bbox
    x2, y2 = x+w, y+h
    cv2.rectangle(template_img, (x, y), (x2, y2), (0, 255, 0), 4)
    cv2.imshow('template', template_img)

    x, y = (target_pos - target_sz/2).astype(int)
    x2, y2 = (target_pos + target_sz/2).astype(int)
    cv2.rectangle(search_img, (x, y), (x2, y2), (0, 255, 0), 8)
    cv2.imshow('SiamMask', search_img)
    key = cv2.waitKey(1)

    return x, y, x2-x, y2-y

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
    model = siammask

    # Setup Dataset
    dataset = AttackDataset()

    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    cv2.namedWindow("template", cv2.WND_PROP_FULLSCREEN)

    bbox = None
    for data in dataset:
        template_img, template_bbox, search_img, search_bbox = data
        if bbox: search_bbox = bbox
        bbox = track(model, p, template_img, template_bbox, search_img, search_bbox)

    