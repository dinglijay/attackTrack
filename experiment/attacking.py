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
from dataset.attack_dataset import AttackDataset
from masks import warp_patch, scale_bbox, get_bbox_mask_tv


parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')
args = parser.parse_args()

def init(model, template_img, template_bbox):
    pos_z, size_z = bbox2center_sz(template_bbox)
    model.template(template_img.to(device),
                   pos_z.to(device),
                   size_z.to(device))
    
    template_img = np.ascontiguousarray(kornia.tensor_to_image(template_img.byte()))
    
    x, y, w, h = template_bbox.squeeze().to(int).cpu().numpy()
    x2, y2 = x+w, y+h
    cv2.rectangle(template_img, (x, y), (x2, y2), (0, 255, 0), 4)
    cv2.imshow('template', template_img)


def track(model, p, search_img, search_bbox):
    pos_x, size_x = bbox2center_sz(search_bbox)
    pscore, delta, pscore_size = model.track(search_img.to(device),
                                             pos_x.to(device),
                                             size_x.to(device))
    scale_x = model.penalty.get_scale_x(size_x)

    assert pscore.shape[0]==1
    tuple(map(lambda x: x.squeeze_().cpu().numpy(), [pos_x, size_x, search_bbox]))
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

    im_h, im_w = search_img.shape[0], search_img.shape[1]
    target_pos[0] = max(0, min(im_w, target_pos[0]))
    target_pos[1] = max(0, min(im_h, target_pos[1]))
    target_sz[0] = max(10, min(im_w, target_sz[0]))
    target_sz[1] = max(10, min(im_h, target_sz[1]))

    x, y = (target_pos - target_sz/2).astype(int)
    x2, y2 = (target_pos + target_sz/2).astype(int)
    cv2.rectangle(search_img, (x, y), (x2, y2), (0, 255, 0), 8)
    cv2.imshow('SiamMask', search_img)
    key = cv2.waitKey(1)

    global i, save_img
    i += 1
    if save_img:
        status =  cv2.imwrite('./results/res_{:03d}.jpg'.format(i), cv2.resize(search_img, (384, 216)))
        print(status, 'results//res_{:03d}.jpg'.format(i))

    return x, y, x2-x, y2-y

if __name__ == '__main__':

    # Setup cf and model file
    args.resume = "../SiamMask/experiments/siammask_sharp/SiamMask_DAVIS.pth"
    args.config = "../SiamMask/experiments/siammask_sharp/config_davis.json"

    cv2.namedWindow("template", cv2.WND_PROP_FULLSCREEN)
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)

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
    dataset = AttackDataset(root_dir='data/lasot/person/person-1', step=1, test=True)
    dataloader = DataLoader(dataset, batch_size=100, num_workers=1)

    # Load Patch
    pert_sz_ratio = (0.5, 0.5)
    # patch = cv2.imread('data/styleTrans/tennis_object.jpg')
    patch = cv2.imread('patch_sm.png')
    patch = kornia.image_to_tensor(patch).to(torch.float) # (3, H, W)
    patch = patch.clone().detach().requires_grad_(False) # (3, H, W)

    # For save tracking result as img file
    i = 0
    save_img = False

    # Random Transformation
    # para_trans_color = {'brightness': 0.2, 'contrast': 0.1, 'saturation': 0.0, 'hue': 0.0}
    # para_trans_affine = {'degrees': 2, 'translate': [0.01, 0.01], 'scale': [0.95, 1.05], 'shear': [-2, 2] }
    # para_trans_affine_t = {'degrees': 2, 'translate': [0.01, 0.01], 'scale': [0.95, 1.05], 'shear': [-2, 2] }
    para_trans_color = {'brightness': 0, 'contrast': 0, 'saturation': 0, 'hue': 0.0}
    para_trans_affine = {'degrees': 0}
    para_trans_affine_t = {'degrees': 0}
    para_gauss = {'kernel_size': (9, 9), 'sigma': (5,5)}

    # Transformation Aug
    trans_color = kornia.augmentation.ColorJitter(**para_trans_color)
    trans_affine = kornia.augmentation.RandomAffine(**para_trans_affine)
    trans_affine_t = kornia.augmentation.RandomAffine(**para_trans_affine_t)
    avg_filter = kornia.filters.GaussianBlur2d(**para_gauss)


    bbox = None
    for data in dataloader:
        data = list(map(lambda x: x.split(1), data))
        for template_img, template_bbox, search_img, search_bbox in zip(*data):
            # Move tensor to device
            data_slice = (template_img, template_bbox, search_img, search_bbox, patch)
            template_img, template_bbox, search_img, search_bbox, patch = tuple(map(lambda x: x.to(device), data_slice))

            # Generate masks
            patch_pos_temp = scale_bbox(template_bbox, pert_sz_ratio)
            patch_pos_search = scale_bbox(search_bbox, pert_sz_ratio)

            # Transformation on patch
            patch_c = patch.expand(template_img.shape[0], -1, -1, -1)
            patch_c = trans_color(patch_c / 255.0) * 255.0
            patch_c = patch_c.clamp(0.01, 255)
            patch_warpped_t = warp_patch(patch_c, template_img, patch_pos_temp)
            patch_warpped_s = warp_patch(patch_c, search_img, patch_pos_search)
            patch_warpped_t = trans_affine_t(patch_warpped_t)
            patch_warpped_s = trans_affine(patch_warpped_s)
            patch_template = torch.where(patch_warpped_t==0, template_img, patch_warpped_t)
            patch_search = torch.where(patch_warpped_s==0, search_img, patch_warpped_s)

            # Init template
            if i==0: 
                init(model, patch_template, template_bbox )

            # Tracking
            track_bbox = torch.tensor(bbox).unsqueeze_(dim=0) if bbox else template_bbox
            bbox = track(model, p, patch_search, track_bbox)

    