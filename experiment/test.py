# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np
import kornia

from torchvision import transforms
from PIL import Image
from masks import scale_bbox_keep_ar, warp_patch, warp

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.utils.region import vot_overlap, vot_float2str

from dataset.DatasetFactory import DatasetFactory

parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', type=str, help='datasets')
parser.add_argument('--config', default='', type=str, help='config file')
parser.add_argument('--snapshot', default='', type=str, help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str, help='eval one special video')
parser.add_argument('--vis', action='store_true', help='whether visualzie result')
parser.add_argument('--patch', type=str, help='patch file')
args = parser.parse_args()

torch.set_num_threads(1)

class Patch_applier(object):
    def __init__(self):
        super(Patch_applier, self).__init__()
        para_trans_color = {'brightness': 0.2, 'contrast': 0.1, 'saturation': 0.0, 'hue': 0.0}
        para_trans_affine = {'degrees': 2, 'translate': [0.01, 0.01], 'scale': [0.95, 1.05], 'shear': [-2, 2] }
        self.pert_sz_ratio = (0.5, 0.4)
    
        # patch_name = 'patches/random.jpg'
        patch_name = args.patch
        self.load_patch(patch_name)
        self.setup_trans(para_trans_color, para_trans_affine)

    def load_patch(self, patch_name, mode='pil'):
        self.patch_cv2 = cv2.imread(patch_name)
        self.patch_aspect = self.patch_cv2.shape[0]/self.patch_cv2.shape[1]
        if mode == 'pil':
            self.patch = Image.open(patch_name)
        else:
            self.patch = kornia.image_to_tensor(self.patch_cv2).to(torch.float).clone() 

    def setup_trans(self, para_trans_color, para_trans_affine, mode='pil'):
        if mode == 'pil':
            self.trans_color = transforms.ColorJitter(**para_trans_color)
            self.trans_affine = transforms.RandomAffine(**para_trans_affine)
        else:
            self.trans_color = kornia.augmentation.ColorJitter(**para_trans_color)
            self.trans_affine = kornia.augmentation.RandomAffine(**para_trans_affine)

    def warp_patch_cv2(self, patch, img, bbox_dest):
        x, y, w, h = 0, 0, patch.shape[1], patch.shape[0]
        points_src = np.array([[x, y], [x+w, y], [x, y+h], [x+w, y+h]], dtype='float32')
        x, y, w, h = bbox_dest
        points_dst = np.array([[x, y], [x+w, y], [x, y+h], [x+w, y+h]], dtype='float32')

        M = cv2.getPerspectiveTransform(points_src, points_dst)
        patch_warped  = cv2.warpPerspective(patch, M, (img.shape[1], img.shape[0]))
        return patch_warped

    def apply_patch_cv2(self, img, gt_bbox):
        # patch_pos = scale_bbox(gt_bbox, self.pert_sz_ratio)
        patch_pos = scale_bbox_keep_ar(gt_bbox, self.pert_sz_ratio, self.patch_aspect)
        patch_c = self.trans_color(self.patch)
        patch_warpped = self.warp_patch_cv2(np.array(patch_c), img, patch_pos)
        patch_warpped = self.trans_affine(Image.fromarray(patch_warpped))
        patch_warpped = np.array(patch_warpped)[:,:,::-1]

        mask = patch_warpped.sum(axis=2)==0
        mask = np.repeat(np.expand_dims(mask, axis=2), 3, axis=2)
        # mask = np.expand_dims(mask, axis=2)
        # mask = np.broadcast_to(mask, (mask.shape[0], mask.shape[1], 3))
        patch_img = np.where(mask, img, patch_warpped)

        # cv2.namedWindow("img", cv2.WND_PROP_FULLSCREEN)
        # cv2.imshow('patch', patch_img)
        # cv2.waitKey(0)
        return patch_img

    def apply_patch_without_trans(self, img, gt_bbox):
        # patch_pos = scale_bbox(gt_bbox, self.pert_sz_ratio)
        patch_pos = scale_bbox_keep_ar(gt_bbox, self.pert_sz_ratio, self.patch_aspect)
        patch_warpped = self.warp_patch_cv2(self.patch_cv2, img, patch_pos)

        mask = patch_warpped.sum(axis=2)==0
        mask = np.repeat(np.expand_dims(mask, axis=2), 3, axis=2)
        patch_img = np.where(mask, img, patch_warpped)
        return patch_img

    def apply_patch(self, img, gt_bbox):
        patch_pos = scale_bbox(gt_bbox, self.pert_sz_ratio)
        patch_pos = torch.tensor(patch_pos).unsqueeze_(dim=0)

        patch_c = self.trans_color(self.patch / 255.0) * 255.0
        patch_c = patch_c.clamp(0.01, 255)

        img = kornia.image_to_tensor(img).unsqueeze_(dim=0)
        patch_warpped = warp_patch_cv2(patch_c, img, patch_pos)
        patch_warpped = self.trans_affine(patch_warpped)
        patch_img = torch.where(patch_warpped==0, img, patch_warpped.byte())
        patch_img = kornia.tensor_to_image(patch_img.squeeze().byte())
        return patch_img

def main():
    # load config
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = build_tracker(model)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    # setup applier
    applier = Patch_applier()

    # result saving path
    video_trained = os.path.split(os.path.split(args.patch)[0])[-1]
    patch_name = os.path.splitext(os.path.split(args.patch)[-1])[0]
    model_name = video_trained+'_'+patch_name
    print('Save results to:', model_name)
                                    
    # model_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()

                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                # img = applier.apply_patch_without_trans(img, gt_bbox_)

                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    if cfg.MASK.MASK:
                        pred_bbox = outputs['polygon']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5 # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                            True, (0, 255, 0), 3)
                    if cfg.MASK.MASK:
                        cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
                                True, (0, 255, 255), 3)
                    else:
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                      (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join('results', args.dataset, model_name,
                    'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))
    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                img = applier.apply_patch_without_trans(img, gt_bbox)
                # img = applier.apply_patch_cv2(img, gt_bbox)
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.namedWindow(video.name, cv2.WND_PROP_FULLSCREEN)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name,
                        'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                        '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                model_path = os.path.join('results', args.dataset, model_name)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':
    main()
