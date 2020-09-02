import configparser
import glob
import json
import torch
import cv2
import time
import kornia
import numpy as np
import os
from os.path import join, isdir, isfile
from torch.utils.data import DataLoader
from pathlib import Path

from utils.load_helper import load_pretrain
from utils.tracker_config import TrackerConfig
from pysot.core.config import cfg

from tracker import Tracker, bbox2center_sz
from siamrpn_tracker import SiamRPNModel, SiamRPNTracker
from dataset.attack_dataset import AttackDataset
from masks import warp_patch, scale_bbox, scale_bbox_keep_ar, get_bbox_mask_tv


def init(model, template_img, template_bbox):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pos_z, size_z = bbox2center_sz(template_bbox)
    model.template(template_img.to(device),
                   pos_z.to(device),
                   size_z.to(device))
    
    template_img = np.ascontiguousarray(kornia.tensor_to_image(template_img.byte()))
    
    x, y, w, h = template_bbox.squeeze().to(int).cpu().numpy()
    x2, y2 = x+w, y+h
    # cv2.rectangle(template_img, (x, y), (x2, y2), (0, 255, 0), 4)
    # cv2.imshow('template', template_img)

def track(model, smooth_lr, search_img, search_bbox, victim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    lr = pscore_size.squeeze().detach().cpu().numpy()[best_pscore_id] * smooth_lr  # lr for OTB

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
    # cv2.rectangle(search_img, (x, y), (x2, y2), (0, 255, 0), 8)
    # cv2.imshow(victim, search_img)

    return x, y, x2-x, y2-y

def setup_siammask(device):
    # load config
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

    return siammask, smooth_lr

def setup_siamrpn(device, config):
    # load config
    base_path = "../pysot/experiments"
    snapshot = join(base_path, config.get('attack', 'victim_nn'), 'model.pth')
    nnConfig = join(base_path, config.get('attack', 'victim_nn'), 'config.yaml')
    cfg.merge_from_file(nnConfig)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    smooth_lr = cfg.TRACK.LR

    # create and load model
    model = SiamRPNModel()
    model.load_state_dict(torch.load(snapshot, map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    model = SiamRPNTracker(model)
    model.get_subwindow.to(device)
    model.penalty.to(device)

    return model, smooth_lr

def main(victim='siammask', display=False, save_result=False, config_f='config/test_config.ini'):

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup tracker
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(config_f)
    datadb = config['attack']['datadb']
    victim = config['attack']['victim']
    patch_save_p = config.get('attack', 'patch_save_f')
    pert_sz_ratio = eval(config['attack']['pert_sz_ratio'])
    pert_pos_delta = eval(config['attack']['pert_pos_delta'])
    get_bbox = scale_bbox_keep_ar if config.getboolean('attack', 'scale_bbox_keep_ar') else scale_bbox

    print('victim: ', victim)
    print(patch_save_p)

    if display:
        cv2.namedWindow("template", cv2.WND_PROP_FULLSCREEN)
        cv2.namedWindow(victim, cv2.WND_PROP_FULLSCREEN)

    # Setup Model
    if victim == 'siammask':
        model, smooth_lr = setup_siammask(device)
    elif victim == 'siamrpn':
        model, smooth_lr = setup_siamrpn(device, config)
    
    # mk result dir
    if save_result:
        dir_path = Path(video)
        Path(dir_path).joinpath('result_sm').mkdir(parents=True, exist_ok=True)
        bbox_file = open(join(Path(dir_path), 'attacked_bbox_sm.txt'), 'w') 
        print('Gt file to: ', Path(dir_path), 'result and groundtruth.txt')

    # Load Patch
    patch = cv2.imread(patch_save_p)
    patch = kornia.image_to_tensor(patch).to(torch.float) # (3, H, W)
    patch = patch.clone().detach().requires_grad_(False) # (3, H, W)
 
    # result saving path
    # if save_result:
    video_trainedOn = os.path.split(os.path.split(patch_save_p)[0])[-1]
    patch_name = os.path.splitext(os.path.split(patch_save_p)[-1])[0]
    model_name = video_trainedOn+'_'+patch_name + '_on-' + victim
    model_path = os.path.join('results', datadb.replace('LaSoT', "LaSOT"), model_name)
    print('Save results to:', model_path)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    # Loop on datadb
    with open(join('data/lasot', datadb.split('-')[-1], 'anno.json'), 'r') as f:
        videos = json.load(f).keys()
    for idx, video_name in enumerate(videos):
        toc = time.time()
        result_path = os.path.join(model_path, '{}.txt'.format(video_name))

        # Setup Dataset
        video = join('data/lasot', datadb.split('-')[-1], video_name)
        dataset = AttackDataset(root_dir=video, test=True)
        dataloader = DataLoader(dataset, batch_size=50, num_workers=4)

        i, first_frame, bbox = 0, True, None
        pred_bboxes = []
        for data in dataloader:
            data = list(map(lambda x: x.split(1), data))
            
            for template_img, template_bbox, search_img, search_bbox in zip(*data):
                if first_frame:
                    template_img, template_bbox = search_img, search_bbox
                # Move tensor to device
                data_slice = (template_img, template_bbox, search_img, search_bbox, patch)
                template_img, template_bbox, search_img, search_bbox, patch = tuple(map(lambda x: x.to(device), data_slice))

                # Generate masks
                aspect = patch.shape[-2] / patch.shape[-1]
                patch_pos_temp = get_bbox(template_bbox, pert_sz_ratio, aspect, pert_pos_delta)
                patch_pos_search = get_bbox(search_bbox, pert_sz_ratio, aspect, pert_pos_delta)

                # Transformation on patch
                patch_c = patch.expand(template_img.shape[0], -1, -1, -1).clamp(0.01, 255)
                patch_warpped_t = warp_patch(patch_c, template_img, patch_pos_temp)
                patch_warpped_s = warp_patch(patch_c, search_img, patch_pos_search)
                patch_template = torch.where(patch_warpped_t==0, template_img, patch_warpped_t)
                patch_search = torch.where(patch_warpped_s==0, search_img, patch_warpped_s)

                # Init template
                if first_frame: 
                    init(model, patch_template, template_bbox)
                    # init(model, template_img, template_bbox )
                    first_frame = False
                    bbox = search_bbox.cpu().squeeze().numpy()

                    if save_result:
                        frame = np.ascontiguousarray(kornia.tensor_to_image(patch_template.byte()))
                        x, y, w, h = template_bbox.squeeze().to(int).cpu().numpy()
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 8)
                        font = cv2.FONT_HERSHEY_SIMPLEX 
                        org = (50, 50) 
                        fontScale = 2
                        color = (0, 0, 255) 
                        thickness = 3
                        cv2.putText(frame, 'Initialization', org, font,  fontScale, color, thickness, cv2.LINE_AA) 
                # Tracking
                else:
                    track_bbox = torch.tensor(bbox).unsqueeze_(dim=0)
                    bbox = track(model, smooth_lr, patch_search, track_bbox, victim)
                    # bbox = track(model, smooth_lr, search_img, track_bbox)

                    if cv2.waitKey(1) & 0xFF == ord('r'):
                        first_frame, bbox = True, None
                        print('reset at =======================================')

                    # if i%1000==0: # change this number to 200?
                    #     first_frame, bbox = True, None
                    #     print('reset at =======================================')
                    if save_result and not first_frame:
                        frame = np.ascontiguousarray(kornia.tensor_to_image(patch_search.byte()))
                        x, y, w, h = bbox
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 8)
                i += 1
                pred_bboxes.append(list(bbox))
                if save_result:
                    f_path = str(dir_path)+'/result_sm/' + '{0:05d}'.format(i) + '.jpg'
                    out = cv2.imwrite(f_path, frame)
                    bbox_file.write('{0:d},{1:d},{2:d},{3:d}\n'.format(x, y, w, h))
                    # videoWriter.write(frame)
                    print(out, i, f_path)

        with open(result_path, 'w') as f:
            for bbox in pred_bboxes:
                x, y, w, h = bbox
                f.write('{0:f},{1:f},{2:f},{3:f}\n'.format(x, y, w, h))
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                idx+1, video_name, (time.time()-toc), i / (time.time()-toc)))
        if save_result:
            bbox_file.close()   


if __name__ == '__main__':
    import fire
    fire.Fire(main)
