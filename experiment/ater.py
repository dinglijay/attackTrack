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
from masks import get_bbox_mask, get_circle_mask, scale_bbox, warp
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
        self.dataset = AttackDataset()

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

        return np.array(cls_list), np.array(delta_list), np.array(iou_list)

        if not need_iou:
            return clss, delta
        else:
            return clss, delta, overlap

    def loss(self, pscore, delta, labels):
        clss_label, deltas_label, ious_label = tuple(map(lambda x: torch.from_numpy(x).to(self.device), labels))

        idx_neg = (ious_label>0.4) & (ious_label<=0.6)
        idx_pos = (ious_label>0.6)
        loss_clss = (torch.max(pscore*idx_pos, dim=1)[0] - torch.max(pscore*idx_neg, dim=1)[0]).mean()

        target = np.array([-127, -127, 0, 0])
        idx = (ious_label>0.6)
        diff = delta.permute(0,2,1) - torch.from_numpy(target).to(self.device)
        diff = diff.sum(dim=2)*idx
        loss_delta = diff.sum() / idx.sum()

        print('Loss -> loss_clss: {:.5f}, loss_delta: {:.5f}'.format(
              loss_clss.cpu().data.numpy(),
              loss_delta.cpu().data.numpy() ))

        return loss_clss + 0.01*loss_delta
    
    def attack(self):
        device = self.device

        # Setup attacker cfg
        num_iters = 1000
        adam_lr = 10
        mu, sigma = 127, 5
        label_thr_iou = 0.4
        pert_sz_ratio = (0.6, 0.3)

        # Load data
        dataloader = DataLoader(self.dataset, batch_size=5)
        data = next(iter(dataloader))
        
        # Move tensor to device
        template_img, template_bbox, search_img, search_bbox = tuple(map(lambda x: x.to(device), data))

        # Tracking and Label
        pscore, delta, pscore_size, bbox = self.get_tracking_result(*data, out_layer='bbox')
        labels = self.get_label(bbox, thr_iou=label_thr_iou, need_iou=True)

        # Generate masks
        im_shape = template_img.shape[2:]
        bbox_pert_temp = scale_bbox(template_bbox, pert_sz_ratio)
        bbox_pert_xcrop = scale_bbox(search_bbox, pert_sz_ratio)
        mask_template = get_bbox_mask(shape=im_shape, bbox=bbox_pert_temp, mode='tensor').to(device)
        mask_search = get_bbox_mask(shape=im_shape, bbox=bbox_pert_xcrop, mode='tensor').to(device)

        # Iteratively opti pert
        pert = (mu + sigma * torch.randn(template_img.shape[-3:])).clamp(0,255)
        pert = pert.clone().detach().to(self.device).requires_grad_(True)
        optimizer = torch.optim.Adam([pert], lr=adam_lr)
        for i in range(num_iters):
            pert_template = template_img * (1-mask_template) + mask_template * pert
            pert_warped = warp(pert, bbox_pert_temp, bbox_pert_xcrop)
            pert_search = search_img * (1-mask_search) + pert_warped * mask_search

            pert_data = (pert_template, template_bbox, pert_search, search_bbox)
            pscore, delta, pscore_size, bbox = self.get_tracking_result(*pert_data, out_layer='bbox')

            if i%10==0:
                plt.figure('loss')
                plt.imshow(pscore.detach().reshape(-1, 5, 25, 25).mean(dim=1).reshape(-1, 25).cpu().numpy())
                plt.pause(0.01)

                plt.figure('pert')
                plt.imshow(kornia.tensor_to_image(pert.byte()))
                plt.pause(0.01)

                # fig, ax = plt.subplots(1,1,num='attacking')
                # # ax[0].set_title('pert_template')
                # # ax[0].imshow(kornia.tensor_to_image(pert_template.byte()))
                # # ax[1].set_title('pert_search')
                # # ax[1].imshow(kornia.tensor_to_image(pert_search.byte()))
                # ax.set_title('result')
                # ax.imshow(kornia.tensor_to_image(self.model.search_cropped.byte()))
                # x, y, w, h = bbox.squeeze()
                # rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
                # ax.add_patch(rect)
                # plt.pause(0.01)

            loss = self.loss(pscore, delta, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pert.data = (pert.data).clamp(0, 255)

    def show_attack_pscore(self):
        pass

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import matplotlib.patches as patches

    trainer = PatchTrainer(args)
    trainer.attack()
