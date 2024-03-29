from datasets.siam_rpn_dataset import AnchorTargetLayer
from utils.anchors import Anchors
from utils.tracker_config import TrackerConfig
from utils.bbox_helper import IoU, corner2center, center2corner
from models.siammask_sharp import select_cross_entropy_loss, get_cls_loss, weight_l1_loss
from masks import get_bbox_mask, get_circle_mask, scale_bbox, warp, warp_patch
import test

from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import torch

import cv2
import kornia
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from collections import namedtuple
Corner = namedtuple('Corner', 'x1 y1 x2 y2')
Center = namedtuple('Center', 'x y w h')


class AttackWrapper(object):

    def __init__(self, x_crop, state, scale_x, s_x):

        self.x_crop = x_crop
        self.s_x = s_x

        self.state = state
        self.model = state['net']
        self.scale_x = scale_x

    def attack(self):

        if self.state['n_frame'] == 1:
            return self.gen_template()
        else:
            return self.gen_xcrop()
        
    def gen_template(self):
        num_iters = 500
        adam_lr = 10
        mu, sigma = 127, 5
        label_thr_iou = 0.2
        pert_sz_ratio = (0.6, 0.3)

        # Load state
        state = self.state
        device = state['device']
        p = state['p']
        s_z = state['s_z']
        
        # Get imgs and mask tensor
        im_shape = state['im'].shape[0:2]
        bbox_pert_temp = scale_bbox(state['gts'][0], pert_sz_ratio)
        bbox_pert_xcrop = scale_bbox(state['gts'][state['n_frame']], pert_sz_ratio)
        mask_template = get_bbox_mask(shape=im_shape, bbox=bbox_pert_temp, mode='tensor').to(device)
        mask_xcrop = get_bbox_mask(shape=im_shape, bbox=bbox_pert_xcrop, mode='tensor').to(device)
        im_template = kornia.image_to_tensor(state['first_im']).to(device)
        im_xcrop = kornia.image_to_tensor(state['im']).to(device)

        # Get Label
        track_res, score_res, pscore_res = self.get_tracking_result(state['template'], self.x_crop)
        labels = self.get_label(track_res, thr_iou=label_thr_iou, need_iou=True)

        im_template = im_template.unsqueeze(dim=0).to(torch.float)
        im_xcrop = im_xcrop.unsqueeze(dim=0).to(torch.float)
        bbox_pert_temp = torch.tensor(bbox_pert_temp).unsqueeze(dim=0).to(device)
        bbox_pert_xcrop = torch.tensor(bbox_pert_xcrop).unsqueeze(dim=0).to(device)

        pert_sz = (100, 75)
        # pert = (mu + sigma * torch.randn(3, pert_sz[0], pert_sz[1])).clamp(0,255)
        pert = cv2.resize(cv2.imread('data/patchnew0.jpg'), (pert_sz[1], pert_sz[0])) # W, H
        pert = kornia.image_to_tensor(pert).to(torch.float)
        pert = pert.clone().detach().to(device).requires_grad_(True).to(im_template.device) # (3, H, W)
        optimizer = torch.optim.Adam([pert], lr=adam_lr)
        for i in range(num_iters):
            patch_warped_template = warp_patch(pert, im_template, bbox_pert_temp)
            patch_warped_search = warp_patch(pert, im_xcrop, bbox_pert_xcrop)
            patch_template = torch.where(mask_template==1, patch_warped_template, im_template)
            patch_search = torch.where(mask_xcrop==1, patch_warped_search, im_xcrop)
    
            template = test.get_subwindow_tracking_(patch_template, state['pos_z'], p.exemplar_size, round(state['s_z']), 0)
            x_crop = test.get_subwindow_tracking_(patch_search, state['target_pos'], p.instance_size, round(self.s_x), 0)
           
            score, delta = self.model.track(x_crop, template)

            ###################  Show Loss and Delta Change ######################
            # if i%10==0:
            #     score_data = F.softmax(score.view(score.shape[0], 2, -1), dim=1)[:,1]
            #     delta_data = delta.view(delta.shape[0], 4, -1).data
            #     res_cx, res_cy, res_w, res_h = track_res
            #     track_res_data = (res_cx-res_w/2, res_cy-res_h/2, res_w, res_h)
            #     self.show_pscore_delta(score_data, delta_data, track_res_data)
            #     self.show_attacking(track_res, score_res, pscore_res, template, x_crop)

            loss = self.loss2(score, delta, pscore_res, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pert.data = (pert.data).clamp(0, 255)

            # fig, ax = plt.subplots(1,2,num='x_crop & template')
            # ax[0].set_title('template')
            # ax[0].imshow(kornia.tensor_to_image(template.byte()))
            # ax[1].set_title('x_crop')
            # ax[1].imshow(kornia.tensor_to_image(x_crop.byte()))
            # plt.pause(0.01)

        state['pert'] = pert.detach()
        state['pert_template'] = template.detach()
        state['pert_sz_ratio'] = pert_sz_ratio
       
        # self.show_label(labels, track_res)
        # self.show_attacking(track_res, score_res, pscore_res, template, x_crop)
        plt.show()

        return template, x_crop

    def gen_xcrop(self):
        # state = self.state

        # im_pert_template = state['im_pert_template']
        # pert_sz_ratio = state['pert_sz_ratio']
        # p = state['p']

        # im_shape = state['im'].shape[0:2]
        # bbox_pert_xcrop = scale_bbox(state['gts'][state['n_frame']], pert_sz_ratio)
        # mask_xcrop = get_bbox_mask(shape=im_shape, bbox=bbox_pert_xcrop, mode='tensor').to(state['device'])

        # bbox_pert_temp = scale_bbox(state['gts'][0], pert_sz_ratio)
        # bbox_pert_xcrop = scale_bbox(state['gts'][state['n_frame']], pert_sz_ratio)
        # im_pert_warped = warp(im_pert_template, bbox_pert_temp, bbox_pert_xcrop)
        # im_xcrop = kornia.image_to_tensor(state['im']).to(state['device'])
        # im_pert_xcrop = im_xcrop * (1-mask_xcrop) + im_pert_warped * mask_xcrop

        # x_crop = test.get_subwindow_tracking_(im_pert_xcrop, state['target_pos'], p.instance_size, round(self.s_x), 0)


        state = self.state
        pert = state['pert']
        pert_sz_ratio = state['pert_sz_ratio']
        p = state['p']
        device = state['device']

        im_shape = state['im'].shape[0:2]
        bbox_pert_xcrop = scale_bbox(state['gts'][state['n_frame']], pert_sz_ratio)
        mask_xcrop = get_bbox_mask(shape=im_shape, bbox=bbox_pert_xcrop, mode='tensor').to(device)

        bbox_pert_xcrop = torch.tensor(bbox_pert_xcrop).unsqueeze(dim=0).to(device)
        im_xcrop = kornia.image_to_tensor(state['im']).to(torch.float).unsqueeze(dim=0).to(device)
        patch_warped_search = warp_patch(pert, im_xcrop, bbox_pert_xcrop)
        patch_search = torch.where(mask_xcrop==1, patch_warped_search, im_xcrop)
        x_crop = test.get_subwindow_tracking_(patch_search, state['target_pos'], p.instance_size, round(self.s_x), 0)

        cv2.imshow('template', kornia.tensor_to_image(state['pert_template'].byte()))
        cv2.imshow('x_crop', kornia.tensor_to_image(x_crop.byte()))
        cv2.waitKey(1)
        
        return state['pert_template'], x_crop

    def get_crop_box(self):
        state = self.state
        p = state['p']
        x, y, w, h = state['gts'][state['n_frame']]
        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])
        wc_x = target_sz[1] + p.context_amount * sum(target_sz)
        hc_x = target_sz[0] + p.context_amount * sum(target_sz)
        s_x = np.sqrt(wc_x * hc_x)
        scale_x = p.exemplar_size / s_x
        d_search = (p.instance_size - p.exemplar_size) / 2
        pad = d_search / scale_x
        s_x = s_x + 2 * pad
        crop_box = [target_pos[0] - round(s_x) / 2, target_pos[1] - round(s_x) / 2, round(s_x), round(s_x)]
        return tuple(map(int, crop_box))

    def get_tracking_result(self, template, x_crop=None):
        p = self.state['p']
        window = self.state['window']
        target_pos = self.state['target_pos']
        target_sz = self.state['target_sz']

        if x_crop is None: x_crop = self.x_crop
        score_, delta = self.model.track(x_crop, template)
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
        score = F.softmax(score_.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0), dim=1).data[:, 1].cpu().numpy()

        delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
        delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

        # size penalty
        target_sz_in_crop = target_sz*self.scale_x
        s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz_in_crop)))  # scale penalty
        r_c = change((target_sz_in_crop[0] / target_sz_in_crop[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty
        penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
        pscore = penalty * score

        # cos window (motion model)
        pscore = pscore * (1 - p.window_influence) + window * p.window_influence
        best_pscore_id = np.argmax(pscore)
        pred_in_crop = delta[:, best_pscore_id] # / scale_x
        lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr  # lr for OTB

        res_cx = int(pred_in_crop[0] + 127)
        res_cy = int(pred_in_crop[1] + 127)
        res_w = int(target_sz_in_crop[0] * (1 - lr) + pred_in_crop[2] * lr)
        res_h = int(target_sz_in_crop[1] * (1 - lr) + pred_in_crop[3] * lr)

        return (res_cx, res_cy, res_w, res_h), score, pscore
    
    def get_label(self, track_res, thr_iou=0.2, need_iou=False):

        anchors = self.state['p'].anchor
        anchor_num = anchors.shape[0]
        clss = np.zeros((anchor_num,), dtype=np.int64)
        delta = np.zeros((4, anchor_num), dtype=np.float32)

        tcx, tcy, tw, th = track_res
        cx, cy, w, h = anchors[:,0]+127, anchors[:,1]+127, anchors[:,2], anchors[:,3] 
        x1, y1, x2, y2 = center2corner(np.array((cx,cy,w,h)))

        # delta
        delta[0] = (tcx - cx) / w
        delta[1] = (tcy - cy) / h
        delta[2] = np.log(tw / w)
        delta[3] = np.log(th / h)

        # IoU
        overlap = IoU([x1, y1, x2, y2], center2corner(track_res))
        pos = np.where(overlap > thr_iou)
        clss[pos] = 1

        if not need_iou:
            return clss, delta
        else:
            return clss, delta, overlap

    def loss(self, score, delta, pscore_res, labels):

        clss_label, deltas_label, ious = labels
        clss_label = torch.from_numpy(clss_label).cuda()
        deltas_lable =  torch.tensor(deltas_label, device='cuda').reshape((1, 4, 5, 25, 25))

        clss_pred = self.model.softmax(score).view(-1, 2)
        pos = Variable(clss_label.data.eq(1).nonzero().squeeze()).cuda()
        neg = Variable(clss_label.data.eq(0).nonzero().squeeze()).cuda()
        loss_clss_pos = get_cls_loss(clss_pred, clss_label, pos)
        loss_clss_neg = get_cls_loss(clss_pred, clss_label, neg)
        
        deltas_pred = delta
        loss_weight = torch.Tensor(pscore_res).cuda().reshape(1, 5, 25, 25)
        loss_delta = weight_l1_loss(deltas_pred, deltas_lable, loss_weight)

        loss = loss_clss_pos + loss_clss_neg + loss_delta

        print('Loss -> clss_pos: {:.2f}, clss_neg: {:.2f}, delta: {:.2f}'\
                .format(loss_clss_pos.cpu().data.numpy(),\
                        loss_clss_neg.cpu().data.numpy(),\
                        loss_delta.cpu().data.numpy()))

        return loss
  
    def loss2(self, score, delta, pscore_res, labels):

        clss_label, deltas_label, ious = labels

        # clss_label = torch.from_numpy(clss_label).cuda()
        # pos = clss_label.data.eq(1).nonzero().squeeze().cuda()
        # neg = clss_label.data.eq(0).nonzero().squeeze().cuda()
        # deltas_label = torch.tensor(deltas_label, device='cuda')

        # ######################################
        # b, a2, h, w = score.size()
        # assert b==1
        # score = score.view(b, 2, a2//2, h, w).permute(0, 2, 3, 4, 1).contiguous()
        # # clss_pred = F.softmax(score, dim=4).view(-1,2)[...,1]
        # clss_pred = F.log_softmax(score, dim=4).view(-1,2)
        # loss_clss = F.nll_loss(clss_pred, clss_label)
        
        # pred_pos = torch.index_select(clss_pred, 0, pos)
        # pred_neg = torch.index_select(clss_pred, 0, neg)
        # # loss_clss = torch.max(pred_neg) - torch.max(pred_pos)

        # ######################################   
        # deltas_pred = delta.view(4,-1)
        # diff = (deltas_pred - deltas_label).abs().sum(dim=0)
        # loss_delta = torch.index_select(diff, 0, pos).mean()

        # print('Loss -> pred_pos: {:.2f}, pred_neg: {:.2f}, clss: {:.2f}, delta: {:.5f}'\
        #         .format(torch.max(pred_pos).cpu().data.numpy(),\
        #                 torch.max(pred_neg).cpu().data.numpy(),\
        #                 loss_clss.cpu().data.numpy(),\
        #                 loss_delta.cpu().data.numpy()))


        target = np.array([1, 1])
        deltas_pred = delta.view(-1,4)
        diff = deltas_pred[..., 2:] - torch.from_numpy(target).cuda()
        diff = diff.abs().mean(dim=1) # (3125)
        pscore_res = torch.from_numpy(pscore_res).cuda()
        idx = torch.topk(pscore_res, k=10, dim=0)[1]
        loss_delta = diff.take(idx).mean()

        print('loss_delta: {:.5f} '.format(loss_delta.cpu().data.numpy()))
        
        return loss_delta

        
    def show_label(self, labels, gt_bbox):
        clss, _, ious = labels

        p = self.state['p']
        res_cx, res_cy, res_w, res_h = gt_bbox

        fig = plt.figure('Label')
        ax = fig.add_subplot(121)
        plt.imshow(np.sum(ious.reshape(5,25,25), axis=0))

        ax = fig.add_subplot(122)
        ax.imshow(self.x_crop.data.squeeze().cpu().numpy().transpose(1,2,0).astype(int))
        rect = patches.Rectangle((res_cx-res_w/2, res_cy-res_h/2), res_w, res_h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        for i in range(p.anchor.shape[0]):
            cx, cy, w, h = p.anchor[i,0], p.anchor[i,1], p.anchor[i,2], p.anchor[i,3]
            bb_center = patches.Circle((cx+127, cy+127), color='b', radius=0.5)
            ax.add_patch(bb_center)
        for i in range(p.anchor.shape[0]):
            if clss[i]==1:
                cx, cy, w, h = p.anchor[i,0], p.anchor[i,1], p.anchor[i,2], p.anchor[i,3]
                bb_center = patches.Circle((cx+127, cy+127), color='r', radius=0.5)
                ax.add_patch(bb_center)
        plt.pause(0.01)

    def show_attacking(self, track_res, score, pscore, template, x_crop=None):
        fig, axes = plt.subplots(2,3, num='Attacking')

        ax = axes[0,0]
        ax.set_title('Result')
        res_cx, res_cy, res_w, res_h = track_res
        ax.imshow(x_crop.data.squeeze().cpu().numpy().transpose(1,2,0).astype(int))
        rect = patches.Rectangle((res_cx-res_w/2, res_cy-res_h/2), res_w, res_h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        ax = axes[0,1]
        ax.set_title('Pscore_bef')
        ax.imshow(pscore.reshape(5,25,25).sum(axis=0))

        ax = axes[0,2]
        ax.set_title('score_bef')
        ax.imshow(0.2*score.reshape(5,25,25).sum(axis=0))

        if x_crop is None: x_crop = self.x_crop
        track_res, score, pscore = self.get_tracking_result(template, x_crop)
        ax = axes[0,0]
        res_cx, res_cy, res_w, res_h = track_res
        rect = patches.Rectangle((res_cx-res_w/2, res_cy-res_h/2), res_w, res_h, linewidth=1, edgecolor='y', facecolor='none')
        ax.add_patch(rect)

        ax = axes[1,0]
        ax.set_title('template')
        ax.imshow(template.data.squeeze().cpu().numpy().transpose(1,2,0).astype(int))
        
        ax = axes[1,1]
        ax.set_title('Pscore')
        ax.imshow(pscore.reshape(5,25,25).sum(axis=0))

        ax = axes[1,2]
        ax.set_title('score')
        ax.imshow(0.2*score.reshape(5,25,25).sum(axis=0))

        plt.pause(0.01)

    def show_pscore_delta(self, pscore, delta, track_bbox, fig_num='pscore_delta'):
        if torch.is_tensor(delta):
            delta = delta.detach().cpu().numpy()
        if not len(delta.shape) == 3:
            delta = delta.reshape((-1, 4, 3125))
        if type(track_bbox) != np.array:
            track_bbox = np.array(track_bbox).reshape(-1, 4)
        # anchor = self.model.all_anchors.detach().cpu().numpy()
        anchor = self.state['p'].anchor
        cx = delta[:, 0, :] * anchor[:, 2] + anchor[:, 0]
        cy = delta[:, 1, :] * anchor[:, 3] + anchor[:, 1]
        w = np.exp(delta[:, 2, :]) * anchor[:, 2]
        h = np.exp(delta[:, 3, :]) * anchor[:, 3]

        iou_list = list()
        for i in range(track_bbox.shape[0]):
            tx, ty, tw, th = track_bbox[i]
            tcx, tcy = tx+tw/2, ty+th/2
            overlap = IoU(center2corner(np.array((cx[i]+127,cy[i]+127,w[i],h[i]))), center2corner(np.array((tcx,tcy,tw,th))))
            iou_list.append(overlap)
        ious = np.array(iou_list) # (B, 3125)
        ious_img = ious.mean(axis=0).reshape(-1, 25)# (B*5, 3125)

        fig, axes = plt.subplots(1,2,num=fig_num)
        ax = axes[0]
        ax.set_title('score')
        ax.imshow(pscore.detach().reshape(-1, 3125).mean(dim=0).reshape(-1, 25).cpu().numpy(),
                vmin=0, vmax=1)
        ax = axes[1]
        ax.set_title('delta')
        ax.imshow(ious_img, vmin=0, vmax=1)
        plt.pause(0.001)

        return ious, ious_img

def norms(Z):
    """Compute norms over all but the first dimension"""
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]

def change(r):
    return np.maximum(r, 1. / r)

def sz(w, h):
    pad = (w + h) * 0.5
    sz2 = (w + pad) * (h + pad)
    return np.sqrt(sz2)

def sz_wh(wh):
    pad = (wh[0] + wh[1]) * 0.5
    sz2 = (wh[0] + pad) * (wh[1] + pad)
    return np.sqrt(sz2)

def get_bbox_in_searchWindow(orig_w, orig_h, target_wh, patch_wh):

    w = int(orig_w * patch_wh / target_wh)
    h = int(orig_h * patch_wh / target_wh)
    x = int(patch_wh/2 - w/2)
    y = int(patch_wh/2 - h/2)

    return x, y, w, h

if __name__ == "__main__":

    import matplotlib.patches as patches
    import numpy as np

    img = cv2.imread('../SiamMask/data/tennis/00000.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[0:2]
    x, y, w, h = 305, 112, 163, 253

    # crop the search region -> z_crop
    target_pos = np.array([x + w / 2, y + h / 2])
    target_sz = np.array([w, h])
    avg_chans = np.mean(img)

    wc_x = target_sz[0] + 0.5 * sum(target_sz)
    hc_x = target_sz[1] + 0.5 * sum(target_sz)
    s_x = round(np.sqrt(wc_x * hc_x))

    scale_x = 127 / s_x
    d_search = 127
    pad = d_search / scale_x
    s_x = s_x + 2 * pad

    z_crop = test.get_subwindow_tracking_(img, target_pos, 255, s_x, avg_chans)
    plt.figure()
    ax = plt.subplot(221)
    ax.imshow(z_crop.astype(int))
    x_, y_, w_, h_ = get_bbox_in_searchWindow(w, h, s_x, 255)
    rect = patches.Rectangle((x_,y_),w_,h_,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    
    ax = plt.subplot(222)
    ax.imshow(img)
    rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)

    mask = circle_mask(shape=img.shape[0:2], loc=target_pos.astype(int), diameter=int(target_sz.min()/5))
    ax = plt.subplot(223)
    ax.imshow(mask)

    mask 
    ax = plt.subplot(224)
    ax.imshow

    plt.show()
