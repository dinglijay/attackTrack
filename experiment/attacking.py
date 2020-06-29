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

from tracker import Tracker, bbox2center_sz
from attack_dataset import AttackDataset
from masks import warp_patch, scale_bbox, get_bbox_mask_tv


parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')
args = parser.parse_args()


def track(model, p, template_img, template_bbox, search_img, search_bbox):
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
    tuple(map(lambda x: x.squeeze_().cpu().numpy(), [pos_x, size_x, template_bbox, search_bbox]))
    template_img = np.ascontiguousarray(kornia.tensor_to_image(template_img.byte()))
    search_img = np.ascontiguousarray(kornia.tensor_to_image(search_img.byte()))

    best_pscore_id = np.argmax(pscore.squeeze().detach().cpu().numpy())
    pred_in_img = delta.squeeze().detach().cpu().numpy()[:, best_pscore_id] / scale_x.cpu().numpy()
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

    global i, save_img
    if save_img:
        i += 1
        status =  cv2.imwrite('./results/res_{:03d}.jpg'.format(i), cv2.resize(search_img, (384, 216)))
        print(status, 'results//res_{:03d}.jpg'.format(i))

    return x, y, x2-x, y2-y

if __name__ == '__main__':

    # Setup cf and model file
    args.resume = "../SiamMask/experiments/siammask_sharp/SiamMask_DAVIS.pth"
    args.config = "../SiamMask/experiments/siammask_sharp/config_davis.json"

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
    dataloader = DataLoader(AttackDataset(root_dir='data/Human1', step=1, test=True), batch_size=100)

    # Load Patch
    patch = cv2.imread('patch_sm.png')
    patch = kornia.image_to_tensor(patch).to(torch.float) # (3, H, W)
    patch = patch.clone().detach().requires_grad_(True) # (3, H, W)

    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    cv2.namedWindow("template", cv2.WND_PROP_FULLSCREEN)
    
    # For save tracking result as img file
    i = 0
    save_img = False

    bbox = None
    for data in dataloader:
        data = list(map(lambda x: x.split(1), data))
        for template_img, template_bbox, search_img, search_bbox in zip(*data):
            # Move tensor to device
            data_slice = (template_img, template_bbox, search_img, search_bbox, patch)
            template_img, template_bbox, search_img, search_bbox, patch = tuple(map(lambda x: x.to(device), data_slice))

            # Generate masks
            pert_sz_ratio = (0.6, 0.3)
            im_shape = template_img.shape[2:]
            patch_pos_temp = scale_bbox(template_bbox, pert_sz_ratio)
            patch_pos_search = scale_bbox(search_bbox, pert_sz_ratio)
            mask_template = get_bbox_mask_tv(shape=im_shape, bbox=patch_pos_temp)
            mask_search = get_bbox_mask_tv(shape=im_shape, bbox=patch_pos_search)

            # Apply patch
            patch_warped_template = warp_patch(patch, template_img, patch_pos_temp)
            patch_warped_search = warp_patch(patch, search_img, patch_pos_search)
            patch_template = torch.where(mask_template==1, patch_warped_template, template_img)
            patch_search = torch.where(mask_search==1, patch_warped_search, search_img)

            # Tracking
            track_bbox = torch.tensor(bbox).unsqueeze_(dim=0) if bbox else template_bbox
            bbox = track(model, p, patch_template, template_bbox, patch_search, track_bbox)

    