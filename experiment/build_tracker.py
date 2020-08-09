
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

import kornia
from siamrpn_tracker import SiamRPNTracker
from tracker import bbox2center_sz


torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
args = parser.parse_args()

def track(model, template_img, template_bbox, search_img, search_bbox):
    pos_z, size_z = bbox2center_sz(template_bbox)
    pos_x, size_x = bbox2center_sz(search_bbox)

    model.template(template_img.to(device),
                   pos_z.to(device),
                   size_z.to(device))

    pscore, delta, pscore_size = model.track(search_img.to(device),
                                             pos_x.to(device),
                                             size_x.to(device))

    scale_x = model.penalty.get_scale_x(size_x)

    assert pscore.shape[0]==1
    tuple(map(lambda x: x.squeeze_().numpy(), [pos_x, size_x, template_bbox, search_bbox]))
    template_img = np.ascontiguousarray(kornia.tensor_to_image(template_img.byte()))
    search_img = np.ascontiguousarray(kornia.tensor_to_image(search_img.byte()))

    best_pscore_id = np.argmax(pscore.squeeze().detach().cpu().numpy())
    pred_in_img = delta.squeeze().detach().cpu().numpy()[:, best_pscore_id] / scale_x
    lr = pscore_size.squeeze().detach().cpu().numpy()[best_pscore_id] * cfg.TRACK.LR  # lr for OTB

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

    from dataset.attack_dataset import AttackDataset
    from torch.utils.data import DataLoader

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
    model = SiamRPNTracker(model)
    model.get_subwindow.to(device)
    model.penalty.to(device)

    # Setup Dataset
    dataset = AttackDataset(root_dir='data/lasot/car/car-2', step=1, test=True)
    dataloader = DataLoader(dataset, batch_size=100, num_workers=1)


    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    cv2.namedWindow("template", cv2.WND_PROP_FULLSCREEN)

    bbox = None
    for data in dataloader:
        data = list(map(lambda x: x.split(1), data))
        for template_img, template_bbox, search_img, search_bbox in zip(*data):
            track_bbox = torch.tensor(bbox).unsqueeze_(dim=0) if bbox else template_bbox
            bbox = track(model, template_img, template_bbox, search_img, track_bbox)
