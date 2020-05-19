from datasets.siam_rpn_dataset import AnchorTargetLayer
from utils.anchors import Anchors
from utils.tracker_config import TrackerConfig
from utils.bbox_helper import IoU, corner2center, center2corner
from models.siammask_sharp import select_cross_entropy_loss, get_cls_loss, weight_l1_loss

from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import torch

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize

from collections import namedtuple
Corner = namedtuple('Corner', 'x1 y1 x2 y2')
Center = namedtuple('Center', 'x y w h')


class AttackWrapper(object):

    def __init__(self, x_crop, state, scale_x):

        self.x_crop = x_crop
        self.template = state['template']

        self.state = state
        self.model = state['net']
        self.scale_x = scale_x

    def attack(self):

        alpha = 1e4
        num_iters = 80

        track_res, score_res, pscore_res = self.get_tracking_result(self.template)
        labels = self.get_label(track_res, need_iou=True)
        mask = torch.from_numpy(circle_mask()).cuda().permute(2,0,1).unsqueeze(dim=0)

        pert = torch.tensor(self.template, requires_grad=True)
        # optimizer = torch.optim.Adam([torch.masked_select(pert, mask==1)], lr=0.5)
        for i in range(num_iters):
            score, delta = self.model.track(self.x_crop, pert)
            loss = self.loss2(score, delta, pscore_res, labels)
            # optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # optimizer.step()
            # pert.data = pert.detach().clamp(0, 255)

            grad = mask * pert.grad.detach()
            grad = grad / norms(grad)
            pert.data = (pert + alpha * grad).clamp(0, 255)
            pert.grad.zero_()
        
        self.show_label(labels, track_res)
        self.show_attacking(track_res, score_res, pscore_res, pert)
        plt.show()

        return pert.detach()

    def get_tracking_result(self, template):
        p = self.state['p']
        window = self.state['window']
        target_pos = self.state['target_pos']
        target_sz = self.state['target_sz']

        score_, delta = self.model.track(self.x_crop, template)
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
        res_w = int(target_sz[0] * (1 - lr) + pred_in_crop[2] * lr)
        res_h = int(target_sz[1] * (1 - lr) + pred_in_crop[3] * lr)

        return (res_cx, res_cy, res_w, res_h), score, pscore
    
    def get_label(self, track_res, thr_iou=0.5, need_iou=False):

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

        clss_label = torch.from_numpy(clss_label).cuda()
        pos = clss_label.data.eq(1).nonzero().squeeze().cuda()
        neg = clss_label.data.eq(0).nonzero().squeeze().cuda()
        deltas_label = torch.tensor(deltas_label, device='cuda')

        ######################################
        b, a2, h, w = score.size()
        assert b==1
        score = score.view(b, 2, a2//2, h, w).permute(0, 2, 3, 4, 1).contiguous()
        clss_pred = F.softmax(score, dim=4).view(-1,2)[...,1]
        
        pred_pos = torch.index_select(clss_pred, 0, pos)
        pred_neg = torch.index_select(clss_pred, 0, neg)
        loss_clss = torch.max(pred_neg) - torch.mean(pred_pos)

        ######################################   
        deltas_pred = delta.view(4,-1)
        diff = (deltas_pred - deltas_label).abs().sum(dim=0)
        loss_delta = torch.index_select(diff, 0, pos).sum()

        print('Loss -> pred_pos: {:.2f}, pred_neg: {:.2f}, delta: {:.2f}'\
                .format(torch.mean(pred_pos).cpu().data.numpy(),\
                        torch.max(pred_neg).cpu().data.numpy(),\
                        loss_delta.cpu().data.numpy()))
        return loss_clss * 100 
        

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

    def show_attacking(self, track_res, score, pscore, pert):
        fig = plt.figure('Attacking')

        ax = fig.add_subplot(231)
        ax.set_title('Before')
        res_cx, res_cy, res_w, res_h = track_res
        ax.imshow(self.x_crop.data.squeeze().cpu().numpy().transpose(1,2,0).astype(int))
        rect = patches.Rectangle((res_cx-res_w/2, res_cy-res_h/2), res_w, res_h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        ax = fig.add_subplot(232)
        ax.set_title('Pscore')
        ax.imshow(pscore.reshape(5,25,25).sum(axis=0))

        ax = fig.add_subplot(233)
        ax.set_title('score')
        ax.imshow(0.2*score.reshape(5,25,25).sum(axis=0))

        track_res, score, pscore = self.get_tracking_result(pert)

        ax = fig.add_subplot(234)
        ax.set_title('After')
        res_cx, res_cy, res_w, res_h = track_res
        ax.imshow(pert.data.squeeze().cpu().numpy().transpose(1,2,0).astype(int))
        rect = patches.Rectangle((res_cx-res_w/2, res_cy-res_h/2), res_w, res_h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        ax = fig.add_subplot(235)
        ax.set_title('Pscore')
        ax.imshow(pscore.reshape(5,25,25).sum(axis=0))

        ax = fig.add_subplot(236)
        ax.set_title('score')
        ax.imshow(0.2*score.reshape(5,25,25).sum(axis=0))

        plt.pause(0.01)

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

def circle_mask(shape=(127,127), loc=(64,64), diameter=10, sharpness=40):
  """Return a circular mask of a given shape"""

  x1 = loc[0]-diameter
  y1 = loc[1]-diameter
  x2 = loc[0]+diameter
  y2 = loc[1]+diameter
  assert x1>=0 and y1>=0 and x2<=shape[0] and y2<=shape[1]

  x = np.linspace(-1, 1, 2*diameter)
  y = np.linspace(-1, 1, 2*diameter)
  xx, yy = np.meshgrid(x, y, sparse=True)
  z = (xx**2 + yy**2) ** sharpness
  circle = 1 - np.clip(z, -1, 1)
  
  mask = np.zeros(shape)
  mask[x1:x2, y1:y2] = circle
  mask = np.expand_dims(mask, axis=2)
  mask = np.broadcast_to(mask, (shape[0],shape[1],3)).astype(np.float32)
  
  return mask

def get_bbox_in_searchWindow(orig_w, orig_h, target_wh, patch_wh):

    w = int(orig_w * patch_wh / target_wh)
    h = int(orig_h * patch_wh / target_wh)
    x = int(patch_wh/2 - w/2)
    y = int(patch_wh/2 - h/2)

    return x, y, w, h

if __name__ == "__main__":

    import matplotlib.patches as patches
    import numpy as np

    from tools.test import get_subwindow_tracking

    img = cv2.imread('data/tennis/00000.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[0:2]
    x, y, w, h = 305, 112, 163, 253

    # crop the search region -> z_crop
    target_pos = np.array([x + w / 2, y + h / 2])
    target_sz = np.array([w, h])
    avg_chans = np.mean(img, axis=(0, 1))

    wc_x = target_sz[0] + 0.5 * sum(target_sz)
    hc_x = target_sz[1] + 0.5 * sum(target_sz)
    s_x = round(np.sqrt(wc_x * hc_x))

    scale_x = 127 / s_x
    d_search = 127
    pad = d_search / scale_x
    s_x = s_x + 2 * pad

    z_crop = get_subwindow_tracking(img, target_pos, 255, s_x, avg_chans, out_mode='numpy')
    plt.figure()
    ax = plt.subplot(121)
    ax.imshow(z_crop)

    # add bbox to search region
    x_, y_, w_, h_ = get_bbox_in_searchWindow(w, h, s_x, 255)
    rect = patches.Rectangle((x_,y_),w_,h_,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)

    ##
    aw = AttackWrapper()
    scores, _, _ =aw.get_target_label(bbox=Corner(x_, y_, x_+w_, y_+h_))
    print(scores.shape)
    print(aw.anchors.all_anchors[0].shape)

    idx = np.argmax(scores.reshape(-1))
    bbox = aw.anchors.all_anchors[0].reshape(4,-1)[:,idx]
    x, y, x2, y2 = bbox
    ax = plt.subplot(122)
    ax.imshow(z_crop)
    rect = patches.Rectangle((x,y),x2-x,y2-y,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)

    plt.show()
