import glob
import configparser
import json
import torch
import cv2
import kornia
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from os.path import isfile, join 

from utils.load_helper import load_pretrain
from utils.tracker_config import TrackerConfig
from utils.bbox_helper import IoU, center2corner

from pysot.core.config import cfg
from siamrpn_tracker import SiamRPNModel, SiamRPNTracker

from tracker import Tracker, bbox2center_sz
from masks import get_bbox_mask_tv, scale_bbox, scale_bbox_keep_ar, warp_patch
from dataset.attack_dataset import AttackDataset
from tmp import rand_shift
from style import get_style_model_and_losses
from style_trans import gram_matrix
from nps import NPSCalculator

from matplotlib import patches
from matplotlib import pyplot as plt


class PatchTrainer(object):

    def __init__(self, config_f='config/config.ini'):
        super(PatchTrainer, self).__init__()

        # Setup device
        self.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True

        # Setup tracker
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        self.config.read(config_f)
        self.setup_tracker(self.config)

        self.attack()
        
        # # Setup style feature layers
        # self.cnn, self.style_losses, self.content_losses = get_style_model_and_losses(device)

    def setup_tracker(self, config):
        if config['train']['victim'] == 'siammask':
            self.setup_siammask()
        elif config['train']['victim'] == 'siamrpn':
            self.setup_siamrpn()

    def setup_siammask(self):
        # load config
        resume = "../SiamMask/experiments/siammask_sharp/SiamMask_DAVIS.pth"
        config_f = "../SiamMask/experiments/siammask_sharp/config_davis.json"
        config = json.load(open(config_f))
        self.p = TrackerConfig()
        self.p.renew()
        self.smooth_lr = self.p.lr

        # create model
        siammask = Tracker(p=self.p, anchors=config['anchors'])

        # load model
        assert isfile(resume), 'Please download {} first.'.format(resume)
        siammask = load_pretrain(siammask, resume)
        siammask.eval().to(self.device)
        self.model = siammask

    def setup_siamrpn(self):
        # load config
        base_path = "../pysot/experiments"
        snapshot = join(base_path, self.config.get('train', 'victim_nn'), 'model.pth')
        nnConfig = join(base_path, self.config.get('train', 'victim_nn'), 'config.yaml')
        cfg.merge_from_file(nnConfig)
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        self.smooth_lr = cfg.TRACK.LR

        # create and load model
        model = SiamRPNModel()
        model.load_state_dict(torch.load(snapshot, map_location=lambda storage, loc: storage.cpu()))
        model.eval().to(self.device)

        # build tracker
        model = SiamRPNTracker(model)
        model.get_subwindow.to(self.device)
        model.penalty.to(self.device)
        self.model = model

    def get_tracking_result(self, template_img, template_bbox, search_img, track_bbox, out_layer='score'):
        device = self.device
        model = self.model

        pos_z, size_z = bbox2center_sz(template_bbox)
        pos_x, size_x = bbox2center_sz(track_bbox)

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
                lr = pscore_size[i, best_pscore_id] * self.smooth_lr  # lr for OTB
                target_sz_in_crop = size_x[i] * scale_x[i]

                res_cx = int(pred_in_crop[0] + 127)
                res_cy = int(pred_in_crop[1] + 127)
                res_w = int(target_sz_in_crop[0] * (1 - lr) + pred_in_crop[2] * lr)
                res_h = int(target_sz_in_crop[1] * (1 - lr) + pred_in_crop[3] * lr)
                res_x = int(res_cx - res_w / 2)
                res_y = int(res_cy - res_h / 2)
                res_bbox.append(((res_x, res_y, res_w, res_h)))
            return pscore, delta, pscore_size, np.array(res_bbox)

    def forward_cnn(self, patch):
        self.cnn(patch.unsqueeze(0)/255.0)
        style_score = 0
        content_score = 0

        for sl in self.style_losses:
            style_score += sl.loss
        for cl in self.content_losses:
            content_score += cl.loss

        style_weight=100
        content_weight=0.01
        style_score *= style_weight
        content_score *= content_weight

        return style_score, content_score

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

    def loss_delta(self, pscore, margin=0.7, topK=15):
        ''' Note that delta is from model.rpn_pred_loc 
        Loss = max(L1(delta, target), margin), among topK bboxs.
        Input: pscore (B, 3125)
        '''
        loss_target =  1 if self.config.get('train', 'target') == 'large' else -1
        delta = self.model.rpn_pred_loc.view((-1, 4, 3125)) # (B, 4, 3125)

        target = torch.tensor([loss_target, loss_target], device=self.device)
        diff = delta.permute(0,2,1)[..., 2:] - target # (B, 3125, 2)
        diff = torch.max(diff.abs(), torch.tensor(margin, device=self.device))
        diff = diff.mean(dim=2) # (B, 3125)
        idx = torch.topk(pscore, k=topK, dim=1)[1]

        diffs = list()
        for i in range(diff.shape[0]):
            diffs.append(diff[i].take(idx[i]) )
        loss_delta = torch.stack(diffs).mean()

        return loss_delta

    def loss_clss(self, pscore):
        clss = self.model.rpn_pred_cls.view(-1, 2, 3125)
        clss = F.softmax(clss, dim=1)[:,1].view(-1, 5, 625)

        # G = torch.bmm(clss.transpose(1, 2), clss)

        # fig, axes = plt.subplots(1,2,num='loss_clss')
        # ax = axes[0]
        # ax.imshow(clss.mean(dim=(0,1)).view(25,25).detach().cpu().numpy(), vmin=0, vmax=1)
        # ax = axes[1]
        # ax.imshow(G.mean(dim=0).detach().cpu().numpy())
        # plt.pause(0.001)

        return 10*clss.mean()
       
    def loss_feat(self, model):
        losses = list()
        for zf, xf in zip(model.zf_all, model.xf_all):
            zgramf = gram_matrix(zf)
            xgramf = gram_matrix(xf)
            loss = F.mse_loss(zgramf, xgramf)
            losses.append(loss)
        if self.config['train']['victim'] == 'siammask':
            loss0, loss1, loss2, loss3 = losses
            loss_feat = loss0 + 1e1*loss1 + 1e3*loss2 + 1e4*loss3
            loss_feat = loss_feat * 5e5
        elif self.config['train']['victim'] == 'siamrpn':
            loss1, loss2, loss3 = losses            
            loss_feat = loss1 + loss2 + loss3
            # print('loss1: {:.3f}, loss2: {:.3f}, loss3: {:.3f}, loss_feat: {:.3f}'.format \
            #         (loss1.item()*1e6, loss2.item()*1e6, loss3.item()*1e6, loss_feat.item()*1e6) )

            loss_feat = loss_feat * 1e3
        return -loss_feat

    def loss_overall(self, losses):
        loss = 0
        for name in self.config.get('loss', 'loss_component').split('-'):
            loss += losses[name]
        return loss

    def attack(self):
        device = self.device
        config = self.config

        # Setup attacker cfg
        mu = config.getfloat('patch','mu')
        sigma = config.getfloat('patch','sigma')
        patch_sz = eval(config['patch']['patch_sz'])
        shift_pos = eval(config['patch']['shift_pos'])
        shift_wh = eval(config['patch']['shift_wh'])
        pert_sz_ratio = eval(config['patch']['pert_sz_ratio'])
        pert_pos_delta = eval(config['patch']['pert_pos_delta'])
        get_bbox = scale_bbox_keep_ar if config.getboolean('patch', 'scale_bbox_keep_ar') else scale_bbox

        loss_tv_margin = config.getfloat('loss','loss_tv_margin')
        loss_delta_topk = config.getint('loss','loss_delta_topk')
        loss_delta_margin = config.getfloat('loss','loss_delta_margin')

        para_trans_color = eval(config['transformParam']['color'])
        para_trans_affine = eval(config['transformParam']['affine'])
        para_trans_affine_t = eval(config['transformParam']['affine_t'])

        video = config['train']['video']
        train_nFrames = config.getint('train', 'train_nFrames')
        adam_lr = config.getfloat('train', 'adam_lr')
        BATCHSIZE = config.getint('train', 'BATCHSIZE')
        n_epochs = config.getint('train', 'n_epochs')
        target = config.get('train', 'target')
        frame_sample = config.get('train', 'frame_sample')

        # Transformation Aug
        trans_color = kornia.augmentation.ColorJitter(**para_trans_color)
        trans_affine = kornia.augmentation.RandomAffine(**para_trans_affine)
        trans_affine_t = kornia.augmentation.RandomAffine(**para_trans_affine_t)
        total_variation = kornia.losses.TotalVariation()

        # Non-printability Score
        nps = NPSCalculator('experiment/30values.txt', patch_sz).to(device)

        # Setup Dataset
        dataset = AttackDataset(video, n_frames=train_nFrames, frame_sample=frame_sample)
        dataloader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=8)

        # Generate patch and setup optimizer
        if config.getboolean('train', 'patch_snapshot'):
            patch = cv2.resize(cv2.imread(config.get('train', 'patch_snapshot_f')), (patch_sz[1], patch_sz[0]))
            patch = kornia.image_to_tensor(patch).to(torch.float).clamp(0.1, 255)
        else:
            patch = (mu + sigma * torch.randn(3, patch_sz[0], patch_sz[1])).clamp(0.1,255)
        patch = patch.clone().detach().to(self.device).requires_grad_(True) # (3, H, W)
        optimizer = torch.optim.Adam([patch], lr=adam_lr)

        for epoch in range(n_epochs):
            for data in dataloader:
                # Move tensor to device
                template_img, template_bbox, search_img, search_bbox = tuple(map(lambda x: x.to(device), data))

                # Gen tracking bbox  
                track_bbox = rand_shift(template_img.shape[-2:], search_bbox, shift_pos, shift_wh, target)
                
                # # Tracking and get label
                # data_track = (template_img, template_bbox, search_img, track_bbox)
                # pscore, delta, pscore_size, bbox_src = self.get_tracking_result(*data_track, out_layer='bbox')

                # Calc patch position 
                patch_pos_temp = get_bbox(template_bbox, pert_sz_ratio, patch_sz[0]/patch_sz[1], pert_pos_delta)
                patch_pos_search = get_bbox(search_bbox, pert_sz_ratio, patch_sz[0]/patch_sz[1], pert_pos_delta)

                # Transformation on patch, (N, W, H) --> (B, N, W, H)
                patch_c = patch.expand(template_img.shape[0], -1, -1, -1)
                patch_c = trans_color(patch_c / 255.0) * 255.0
                patch_c = patch_c.clamp(0.1, 255) # Just a trick for masking operand
                patch_warpped_t = warp_patch(patch_c, template_img, patch_pos_temp)
                patch_warpped_s = warp_patch(patch_c, search_img, patch_pos_search)
                patch_warpped_t = trans_affine_t(patch_warpped_t)
                patch_warpped_s = trans_affine(patch_warpped_s)
                patch_template = torch.where(patch_warpped_t==0, template_img, patch_warpped_t)
                patch_search = torch.where(patch_warpped_s==0, search_img, patch_warpped_s)
                # mask = (patch_warpped_t.sum(dim=1)==0).unsqueeze(dim=1).expand(-1,3,-1,-1)
                # patch_template = torch.where(mask, template_img, patch_warpped_t)
                # mask = (patch_warpped_s.sum(dim=1)==0).unsqueeze(dim=1).expand(-1,3,-1,-1)
                # patch_search = torch.where(mask, search_img, patch_warpped_s)
                # self.show_patch_warpped_t(patch_template, template_bbox, patch_search, track_bbox)

                # Forward tracking net
                pert_data = (patch_template, template_bbox, patch_search, track_bbox)
                pscore, delta, pscore_size, bbox = self.get_tracking_result(*pert_data, out_layer='bbox')

                loss_delta = self.loss_delta(pscore, loss_delta_margin, loss_delta_topk)
                loss_feat = self.loss_feat(self.model)
                loss_nps = nps(patch/255.0)
                tv = 0.05 * total_variation(patch)/torch.numel(patch)
                loss_tv = torch.max(tv, torch.tensor(loss_tv_margin).to(device))
                # loss_clss = self.loss_clss(pscore)
                # loss_style, loss_content = self.forward_cnn(patch)feat

                losses = {'delta':loss_delta, 'feat':loss_feat, 'tv':loss_tv, 'nps': loss_nps}
                loss = self.loss_overall(losses)
                print('epoch {:} -> loss_feat: {:.3f}, loss_delta: {:.3f}, loss_tv: {:.3f}, loss_nps: {:.3f}'.format \
                    (epoch, loss_feat.item(), loss_delta.item(), loss_tv.item(), loss_nps.item()) )
                
                # self.show_pscore_delta(pscore, self.model.rpn_pred_loc, bbox_src)
                # self.show_attack_plt(pscore, bbox, bbox_src, patch)
                # plt.pause(0.001)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                patch.data = (patch.data).clamp(0.1, 255)

            patch_save_p = config.get('train', 'patch_save_f')
            cv2.imwrite(patch_save_p, kornia.tensor_to_image(patch.detach().byte()))

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
        ious_img = ious.mean(axis=0).reshape(-1, 25)# (B*5, 25)

        fig, axes = plt.subplots(1,2,num=fig_num)
        ax = axes[0]
        ax.set_title('pscore')
        ax.imshow(pscore.detach().reshape(-1, 3125).mean(dim=0).reshape(-1, 25).cpu().numpy(),
                vmin=0, vmax=1)
        ax = axes[1]
        ax.set_title('iou: {:.2f}'.format(ious.max()))
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

    def show_patch_warpped_t(self, patch_template, template_bbox, patch_search, track_bbox):
        cv2.namedWindow("patch_template", cv2.WND_PROP_FULLSCREEN)
        cv2.namedWindow("patch_search", cv2.WND_PROP_FULLSCREEN)

        img_h, img_w = patch_template.shape[-2:]
        patch_temp_img = kornia.tensor_to_image(patch_template.byte()).reshape(-1, img_w, 3)
        patch_sear_img = kornia.tensor_to_image(patch_search.byte()).reshape(-1, img_w, 3)
        patch_temp_img = np.ascontiguousarray(patch_temp_img)
        patch_sear_img = np.ascontiguousarray(patch_sear_img)

        for i, xywh in enumerate(track_bbox.cpu().numpy()):
            x, y, w, h = list(map(int, xywh))
            y += i*img_h
            cv2.rectangle(patch_sear_img, (x, y), (x+w, y+h), (0, 255, 0), 3)

        cv2.imshow('patch_template', patch_temp_img)
        cv2.imshow('patch_search', patch_sear_img)
        cv2.waitKey(0)

    def load_patch(self, img_name, size=(300, 400)):
        """
        Generate a random patch as a starting point for optimization.
        """
        # adv_patch_cpu = (mu + sigma * torch.randn(3, size[0], size[1])).clamp(0,255)
        adv_patch_cpu = cv2.resize(cv2.imread(img_name), (size[1], size[0])) # W, H
        adv_patch_cpu = kornia.image_to_tensor(patch).to(torch.float)

        return adv_patch_cpu

def main(config_f='config/config.ini'):
    trainer = PatchTrainer(config_f=config_f)
    patch = trainer.attack()

if __name__ == '__main__':
    import fire
    fire.Fire(main)
    
    