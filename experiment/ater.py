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
from utils.bbox_helper import IoU, corner2center, center2corner


from tracker import Tracker, tracker_init, tracker_track
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

        model.template(kornia.image_to_tensor(template_img).to(device).float(), 
                       torch.from_numpy(pos_z).to(device),
                       torch.from_numpy(size_z).to(device))
        pscore, delta, pscore_size = model.track(kornia.image_to_tensor(search_img).to(device).float(),
                                                   torch.from_numpy(pos_x).to(device),
                                                   torch.from_numpy(size_x).to(device))    
        if out_layer == 'score':
            return pscore, delta, pscore_size
        elif out_layer == 'bbox':
            wc_x = size_x[1] + p.context_amount * sum(size_x)
            hc_x = size_x[0] + p.context_amount * sum(size_x)
            scale_x = p.exemplar_size / np.sqrt(wc_x * hc_x)

            best_pscore_id = np.argmax(pscore.squeeze().detach().cpu().numpy())
            pred_in_crop = delta.squeeze().detach().cpu().numpy()[:, best_pscore_id] # / scale_x
            lr = pscore_size.squeeze().detach().cpu().numpy()[best_pscore_id] * p.lr  # lr for OTB

            target_sz_in_crop = size_x * scale_x
            res_cx = int(pred_in_crop[0] + 127)
            res_cy = int(pred_in_crop[1] + 127)
            res_w = int(target_sz_in_crop[0] * (1 - lr) + pred_in_crop[2] * lr)
            res_h = int(target_sz_in_crop[1] * (1 - lr) + pred_in_crop[3] * lr)

            res_x = int(res_cx - res_w / 2)
            res_y = int(res_cy - res_h / 2)
            return pscore, delta, pscore_size, (res_x, res_y, res_w, res_h)
            
    def attack(self):

        # Setup attacker cfg
        num_iters = 100
        adam_lr = 300
        mu, sigma = 127, 5
        label_thr_iou = 0.3
        pert_sz_ratio = (0.6, 0.3)

        # tracking
        data = next(iter(self.dataset))
        template_img, template_bbox, search_img, search_bbox = data
        pscore, delta, pscore_size, bbox = self.get_tracking_result(*data, out_layer='bbox')

        # Label
        labels = self.get_label(bbox, thr_iou=label_thr_iou, need_iou=True)


        fig, axes = plt.subplots(2,2, num='attack')
        
        ax = axes[0,0]
        ax.set_title('before')
        search_cropped =  kornia.tensor_to_image(self.model.search_cropped.byte())
        ax.imshow(search_cropped)
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        ax = axes[0,1]
        ax.imshow(pscore.detach().reshape(5,25,25).cpu().numpy().sum(axis=0))
        
        plt.show()

    def get_label(self, track_bbox, thr_iou=0.2, need_iou=False):

        anchors = self.model.anchor.all_anchors[1].reshape(4, -1).transpose(1, 0)
        anchor_num = anchors.shape[0]
        clss = np.zeros((anchor_num,), dtype=np.int64)
        delta = np.zeros((4, anchor_num), dtype=np.float32)

        tx, ty, tw, th = track_bbox
        tcx, tcy = tx+tw/2, ty+th/2
        cx, cy, w, h = anchors[:,0]+127, anchors[:,1]+127, anchors[:,2], anchors[:,3] 

        # delta
        delta[0] = (tcx - cx) / w
        delta[1] = (tcy - cy) / h
        delta[2] = np.log(tw / w)
        delta[3] = np.log(th / h)

        # IoU
        overlap = IoU(center2corner(np.array((cx,cy,w,h))), center2corner(np.array((tcx,tcy,tw,th))))
        pos = np.where(overlap > thr_iou)
        clss[pos] = 1

        # show       
        fig = plt.figure('Label')
        ax = fig.add_subplot(121)
        plt.imshow(np.sum(overlap.reshape(5,25,25), axis=0))

        ax = fig.add_subplot(122)
        ax.imshow(kornia.tensor_to_image(self.model.search_cropped.byte()))
        rect = patches.Rectangle((tx, ty), tw, th, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        for i in range(anchors.shape[0]):
            cx, cy, w, h = anchors[i,0], anchors[i,1], anchors[i,2], anchors[i,3]
            bb_center = patches.Circle((cx+127, cy+127), color='b', radius=0.5)
            ax.add_patch(bb_center)
        for i in range(anchors.shape[0]):
            if clss[i]==1:
                cx, cy, w, h = anchors[i,0], anchors[i,1], anchors[i,2], anchors[i,3]
                bb_center = patches.Circle((cx+127, cy+127), color='r', radius=0.5)
                ax.add_patch(bb_center)
        plt.pause(0.01)

        if not need_iou:
            return clss, delta
        else:
            return clss, delta, overlap

        
def bbox2center_sz(bbox):
    x, y, w, h = bbox
    pos = np.array([x + w / 2, y + h / 2])
    sz = np.array([w, h])
    return pos, sz

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import matplotlib.patches as patches

    trainer = PatchTrainer(args)
    trainer.attack()
