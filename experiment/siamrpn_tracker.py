# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker
from pysot.models.model_builder import ModelBuilder

import torch
from tracker import SubWindow, PenaltyLayer
from utils.tracker_config import TrackerConfig

class SiamRPNModel(ModelBuilder):
    def __init__(self):
        super(SiamRPNModel, self).__init__()
    
    def template(self, z):
        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf
        return zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.rpn_head(self.zf, xf)
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
                'xf_all': xf,
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None
               }


class SiamRPNTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamRPNTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()

        # Dylan
        self.get_subwindow = SubWindow()
        self.all_anchors = torch.from_numpy(self.anchors).float()
        self.p = self.set_p(cfg=cfg)
        self.penalty = PenaltyLayer(anchor=self.all_anchors, p=self.p)

    def set_p(self, cfg):
        p = TrackerConfig()
        p.windowing = 'cosine'  # to penalize large displacements [cosine/uniform]
        p.penalty_k = cfg.TRACK.PENALTY_K
        p.window_influence = cfg.TRACK.WINDOW_INFLUENCE
        p.score_size = self.score_size
        p.anchor_num = self.anchor_num
        p.context_amount =  cfg.TRACK.CONTEXT_AMOUNT
        p.exemplar_size = cfg.TRACK.EXEMPLAR_SIZE
        return p

    def get_crop_sz(self, target_sz, is_search=False):
        device = target_sz.device
        target_sz = target_sz.cpu().numpy()

        if len(target_sz.shape) == 1:
            target_sz = np.expand_dims(target_sz, axis=0)

        wc = target_sz[:,0] + self.p.context_amount * target_sz.sum(axis=1)
        hc = target_sz[:,1] + self.p.context_amount * target_sz.sum(axis=1)
        crop_sz = np.sqrt(wc * hc).round() 

        if is_search:
            scale_x = self.p.exemplar_size / crop_sz
            d_search = (self.p.instance_size - self.p.exemplar_size) / 2
            pad = d_search / scale_x
            crop_sz = crop_sz + 2 * pad

        return torch.from_numpy(crop_sz).requires_grad_(False).to(device)

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def template(self, template_img, template_pos, template_sz):
        crop_sz = self.get_crop_sz(template_sz)
        self.template_cropped = self.get_subwindow(template_img, template_pos, crop_sz, out_size=self.p.exemplar_size)
        self.zf_all = self.model.template(self.template_cropped)

    def track(self, search_img, target_pos, target_sz):
        crop_sz = self.get_crop_sz(target_sz, is_search=True)
        self.search_cropped = self.get_subwindow(search_img, target_pos, crop_sz, out_size=self.p.instance_size)
        outputs = self.model.track(self.search_cropped)
        self.xf_all, self.rpn_pred_cls, self.rpn_pred_loc = outputs['xf_all'], outputs['cls'], outputs['loc']
        pscore, delta, pscore_size = self.penalty(outputs['cls'], outputs['loc'], target_sz)
        return pscore, delta, pscore_size
