import glob
import argparse
import torch
import cv2
import kornia
import numpy as np
from os.path import join, isdir, isfile
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.config_helper import load_config
from utils.load_helper import load_pretrain
from utils.tracker_config import TrackerConfig
from utils.bbox_helper import IoU, corner2center, center2corner

from tracker import Tracker, bbox2center_sz
from masks import get_bbox_mask, get_circle_mask, scale_bbox, warp, warp_patch
from attack_dataset import AttackDataset
from transform import TotalVariation

from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')
parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
parser.add_argument('--gt_file', default=None, type=str, help='ground truth txt file')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()


class PatchTrainer(object):

    def __init__(self, args):
        super(PatchTrainer, self).__init__()

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True

        # Setup tracker cfg
        cfg = load_config(args)
        p = TrackerConfig()
        p.renew()
        self.p = p

        # Setup tracker
        siammask = Tracker(p=p, anchors=cfg['anchors'])
        if args.resume:
            assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
            siammask = load_pretrain(siammask, args.resume)
        siammask.eval().to(self.device)
        self.model = siammask

        # Setup Dataset
        self.dataset = AttackDataset('data/Human2')

        # Setup Transformation
        self.total_variation = TotalVariation().to(self.device)
        

    def get_tracking_result(self, template_img, template_bbox, search_img, search_bbox, out_layer='score'):
        device = self.device
        model = self.model
        p = self.p

        pos_z, size_z = bbox2center_sz(template_bbox)
        pos_x, size_x = bbox2center_sz(search_bbox)

        model.template(template_img.to(device),
                       pos_z.to(device),
                       size_z.to(device))       
        pscore, delta, pscore_size = model.track(search_img.to(device),
                                                 pos_x.to(device),
                                                 size_x.to(device))
        if out_layer == 'score':
            return pscore, delta, pscore_size
        elif out_layer == 'bbox':
            scale_x = self.model.penalty.get_scale_x(size_x)
            res_bbox = list()
            for i in range(pscore.shape[0]):
                best_pscore_id = pscore[i,...].argmax()
                pred_in_crop = delta[i, :, best_pscore_id] # / scale_x
                lr = pscore_size[i, best_pscore_id] * p.lr  # lr for OTB
                target_sz_in_crop = size_x[i] * scale_x[i]

                res_cx = int(pred_in_crop[0] + 127)
                res_cy = int(pred_in_crop[1] + 127)
                res_w = int(target_sz_in_crop[0] * (1 - lr) + pred_in_crop[2] * lr)
                res_h = int(target_sz_in_crop[1] * (1 - lr) + pred_in_crop[3] * lr)
                res_x = int(res_cx - res_w / 2)
                res_y = int(res_cy - res_h / 2)
                res_bbox.append(((res_x, res_y, res_w, res_h)))
            return pscore, delta, pscore_size, np.array(res_bbox)

    def get_label(self, track_bbox, thr_iou=0.2, need_iou=False):
        '''Input track_bbox: np.array of size (B, 4)
           Return np.array type '''

        anchors = self.model.anchor.all_anchors[1].reshape(4, -1).transpose(1, 0)
        anchor_num = anchors.shape[0]
      
        cls_list, delta_list, iou_list = list(), list(), list()
        for i in range(track_bbox.shape[0]):
            tx, ty, tw, th = track_bbox[i]
            tcx, tcy = tx+tw/2, ty+th/2
            cx, cy, w, h = anchors[:,0]+127, anchors[:,1]+127, anchors[:,2], anchors[:,3] 

            clss = np.zeros((anchor_num,), dtype=np.int64)
            delta = np.zeros((4, anchor_num), dtype=np.float32)

            # delta
            delta[0] = (tcx - cx) / w
            delta[1] = (tcy - cy) / h
            delta[2] = np.log(tw / w)
            delta[3] = np.log(th / h)

            # IoU
            overlap = IoU(center2corner(np.array((cx,cy,w,h))), center2corner(np.array((tcx,tcy,tw,th))))
            pos = np.where(overlap > thr_iou)
            clss[pos] = 1

            cls_list.append(clss)
            delta_list.append(delta)
            iou_list.append(overlap)   
        # 
        # fig = plt.figure('Label')
        # for i in range(track_bbox.shape[0]):
        #     ax = fig.add_subplot(1, track_bbox.shape[0], i+1)
        #     tx, ty, tw, th = track_bbox[i]

        #     ax.imshow(kornia.tensor_to_image(self.model.search_cropped[i].byte()))
        #     rect = patches.Rectangle((tx, ty), tw, th, linewidth=2, edgecolor='r', facecolor='none')
        #     ax.add_patch(rect)
        #     for i in range(anchors.shape[0]):
        #         cx, cy, w, h = anchors[i,0], anchors[i,1], anchors[i,2], anchors[i,3]
        #         bb_center = patches.Circle((cx+127, cy+127), color='b', radius=0.2)
        #         ax.add_patch(bb_center)
        #     for i in range(anchors.shape[0]):
        #         if clss[i]==1:
        #             cx, cy, w, h = anchors[i,0], anchors[i,1], anchors[i,2], anchors[i,3]
        #             bb_center = patches.Circle((cx+127, cy+127), color='r', radius=0.2)
        #             ax.add_patch(bb_center)
        # plt.show()

        if not need_iou:
            return np.array(cls_list), np.array(delta_list)
        else:
            return np.array(cls_list), np.array(delta_list), np.array(iou_list)

    def loss(self, pscore, delta, labels):
        clss_label, deltas_label, ious_label = tuple(map(lambda x: torch.from_numpy(x).to(self.device), labels))
        delta = self.model.rpn_pred_loc.view(deltas_label.shape) # (B, 4, 3125)

        # target = np.array([-1, -, 2000, 2000])
        # diff = delta.permute(0,2,1) - torch.from_numpy(target).to(self.device)
        target = np.array([1, 1])
        diff = delta.permute(0,2,1)[...,2:] - torch.from_numpy(target).to(self.device)
        diff = diff.abs().mean(dim=2) # (B, 3125)
        idx = torch.topk(pscore, k=10, dim=1)[1]

        diffs = list()
        for i in range(diff.shape[0]):
            diffs.append(diff[i].take(idx[i]) )
        loss_delta = torch.stack(diffs).mean()

        return loss_delta

    def loss_clss(self, pscore, delta, labels):
        clss_label, deltas_label, ious_label = tuple(map(lambda x: torch.from_numpy(x).to(self.device), labels))

        # mask_neg = (ious_label>0.4) & (ious_label<=0.6)
        mask_neg = (ious_label<=0.6)
        mask_pos = (ious_label>0.6)
        loss_clss = (torch.max(pscore*mask_pos, dim=1)[0] - torch.max(pscore*mask_neg, dim=1)[0]).mean()

        target = np.array([0, 0])
        diff = delta.permute(0,2,1)[...,2:] - torch.from_numpy(target).to(self.device)
        idx = torch.topk(pscore, k=100, dim=1)[1]
        diff = diff.abs().mean(dim=2) # (B, 3125)
        diffs = list()
        for i in range(diff.shape[0]):
            diffs.append(diff[i].take(idx[i]) )
        loss_delta = torch.stack(diffs).mean()

        print('Loss -> loss_clss: {:.5f}, loss_delta: {:.5f}'.format(
              loss_clss.cpu().data.numpy(),
              loss_delta.cpu().data.numpy() ))
        return loss_clss

    def loss_delta(self, pscore, delta, labels):
        clss_label, deltas_label, ious_label = tuple(map(lambda x: torch.from_numpy(x).to(self.device), labels))
        delta = self.model.rpn_pred_loc.view(deltas_label.shape) # (B, 4, 3125)

        pos = clss_label.squeeze().eq(1).nonzero().squeeze().cuda()
        ######################################   
        deltas_pred = delta.view(4,-1)
        deltas_label = deltas_label.view(4, -1)
        diff = (deltas_pred - deltas_label).abs().sum(dim=0)
        loss_delta = torch.index_select(diff, 0, pos).mean()

        print('Loss -> loss_delta: {:.5f}'.format(loss_delta.cpu().data.numpy() ))
        return -loss_delta

    def loss_delta2(self, pscore, delta, labels):
        clss_label, deltas_label, ious_label = tuple(map(lambda x: torch.from_numpy(x).to(self.device), labels))
        delta = self.model.rpn_pred_loc.view(deltas_label.shape) # (B, 4, 3125)

        diff = (delta - deltas_label).abs().sum(dim=1)
        weight = pscore.detach().view(diff.shape)
        loss_delta = (diff * weight).mean()

        print('Loss -> loss_delta: {:.5f}'.format(loss_delta.cpu().data.numpy() ))
        return -loss_delta

    def loss_delta3(self, pscore, delta, labels):
        clss_label, deltas_label, ious_label = tuple(map(lambda x: torch.from_numpy(x).to(self.device), labels))
        delta = self.model.rpn_pred_loc.view(deltas_label.shape) # (B, 4, 3125)
        pos = clss_label.eq(1).nonzero()

        diff = (delta - deltas_label).abs().sum(dim=1)
        loss_delta = torch.index_select(diff, 0, pos).mean()
        print('Loss -> loss_delta: {:.5f}'.format(loss_delta.cpu().data.numpy() ))
        return -loss_delta

    def attack(self):
        device = self.device

        # Setup attacker cfg
        num_iters = 300
        adam_lr = 10
        mu, sigma = 127, 5
        patch_sz = (100, 75)
        label_thr_iou = 0.2
        pert_sz_ratio = (0.6, 0.3)

        BATCHSIZE = 2
        n_epochs = 10
        dataloader = DataLoader(self.dataset, 
                                batch_size=BATCHSIZE,
                                shuffle=True)

        # Generate patch and setup optimizer
        patch = (mu + sigma * torch.randn(3, patch_sz[0], patch_sz[1])).clamp(0,255)
        # patch = cv2.resize(cv2.imread('data/patchnew0.jpg'), (patch_sz[1], patch_sz[0])) # W, H
        # patch = kornia.image_to_tensor(patch).to(torch.float)
        patch = patch.clone().detach().to(self.device).requires_grad_(True) # (3, H, W)
        optimizer = torch.optim.Adam([patch], lr=adam_lr)

        for epoch in range(n_epochs):
            for data in dataloader:
                # Move tensor to device
                template_img, template_bbox, search_img, search_bbox = tuple(map(lambda x: x.to(device), data))

                # Tracking and Label
                data_track = (template_img, template_bbox, search_img, template_bbox)
                # with torch.no_grad():
                pscore, delta, pscore_size, bbox_src = self.get_tracking_result(*data_track, out_layer='bbox')
                
                # self.show_pscore_delta(pscore, self.model.rpn_pred_loc, bbox_src)
                labels = self.get_label(bbox_src, thr_iou=label_thr_iou, need_iou=True)
                self.bbox_src = bbox_src
        
                # Generate masks
                im_shape = template_img.shape[2:]
                patch_pos_temp = scale_bbox(template_bbox, pert_sz_ratio)
                patch_pos_search = scale_bbox(search_bbox, pert_sz_ratio)
                mask_template = get_bbox_mask(shape=im_shape, bbox=patch_pos_temp, mode='tensor').to(device)
                mask_search = get_bbox_mask(shape=im_shape, bbox=patch_pos_search, mode='tensor').to(device)

                # Iteratively opti pert
                for i in range(num_iters):
                    patch_warped_template = warp_patch(patch, template_img, patch_pos_temp)
                    patch_warped_search = warp_patch(patch, search_img, patch_pos_search)
                    patch_template = torch.where(mask_template==1, patch_warped_template, template_img)
                    patch_search = torch.where(mask_search==1, patch_warped_search, search_img)

                    pert_data = (patch_template, template_bbox, patch_search, template_bbox)
                    pscore, delta, pscore_size, bbox = self.get_tracking_result(*pert_data, out_layer='bbox')
                            
                    loss_delta = self.loss(pscore, delta, labels)
                    tv = 0.05 * self.total_variation(patch)
                    loss_tv = torch.max(tv, torch.tensor(2.5).to(device))
                    loss = loss_delta + loss_tv
                    
                    if i%20==0:
                        self.show_pscore_delta(pscore, self.model.rpn_pred_loc, bbox_src)
                        self.show_attack_plt(pscore, bbox, bbox_src, patch)
                        print('epoch {:} -> loss_delta: {:.5f}, tv: {:.5f}, loss: {:.5f} '.format(\
                        epoch+1, loss_delta.cpu().data.numpy(), tv.cpu().data.numpy(), loss.cpu().data.numpy() ))
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    patch.data = (patch.data).clamp(0, 255)
                plt.show()

                # print('epoch {:} -> loss_delta: {:.5f}, tv: {:.5f}, loss: {:.5f} '.format(\
                #         epoch+1, loss_delta.cpu().data.numpy(), tv.cpu().data.numpy(), loss.cpu().data.numpy() ))

        return kornia.tensor_to_image(patch.detach().byte())

    def show_pscore_delta(self, pscore, delta, track_bbox, fig_num='pscore_delta'):
        if torch.is_tensor(delta):
            delta = delta.detach().cpu().numpy()
        if not len(delta.shape) == 3:
            delta = delta.reshape((-1, 4, 3125))
        anchor = self.model.all_anchors.detach().cpu().numpy()
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
        ax.set_title('pscore')
        ax.imshow(pscore.detach().reshape(-1, 3125).mean(dim=0).reshape(-1, 25).cpu().numpy(),
                vmin=0, vmax=1)
        ax = axes[1]
        ax.set_title('delta')
        ax.imshow(ious_img, vmin=0, vmax=1)
        plt.pause(0.001)

        return ious, ious_img

    def show_attack_plt(self, pscore, bbox, bbox_src, patch):
        fig, axes = plt.subplots(1,3,num='attacking')
        ax = axes[0]
        ax.set_title('patch')
        ax.imshow(kornia.tensor_to_image(patch.byte()))
        ax = axes[1]
        ax.set_title('template')
        ax.imshow(kornia.tensor_to_image(self.model.template_cropped.byte()).reshape(-1, 127, 3))
        ax = axes[2]
        ax.set_title('result')
        ax.imshow(kornia.tensor_to_image(self.model.search_cropped.byte()).reshape(-1, 255, 3))
        for i, xywh in enumerate(bbox):
            x, y, w, h = xywh
            rect = patches.Rectangle((x, y+i*255), w, h, linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
        for i, xywh in enumerate(bbox_src):
            x, y, w, h = xywh
            rect = patches.Rectangle((x, y+i*255), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.pause(0.01)

    def generate_patch(self, size=(300, 400)):
        """
        Generate a random patch as a starting point for optimization.
        """
        adv_patch_cpu = (mu + sigma * torch.randn(3, size[0], size[1])).clamp(0,255)

        return adv_patch_cpu

if __name__ == '__main__':
    import matplotlib.patches as patches

    trainer = PatchTrainer(args)
    patch = trainer.attack()
    cv2.imwrite('patch.png', patch)