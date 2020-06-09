import torch
import kornia
import torch.nn.functional as F
import numpy as np 

from custom import Custom


class SubWindow(torch.nn.Module):
    """
    Crop image at pos with size_wh: Croppping + Padding + Resizing.
    Forward Input:
        im: input images, torch.tensor of shape ([B,] C, H, W)
        pos: crop postions, torch.tensor of shape ([B,] 2)
        size_wh: crop sizes, torch.tensor of shape ([B,] )
        out_size: 127 or 255
    Forward Output:
        cropped images (B, C, model_sz, model_sz)
    To do:
        --> Looping through each image seems like stupid
        --> How resize mode matters? 
    """
    def __init__(self):
        super(SubWindow, self).__init__()
        
    def forward(self, im, pos, size_wh, out_size=127):

        if len(im.shape) == 3:
            im.unsqueeze_(dim=0)
            pos.unsqueeze_(dim=0)
            size_wh.unsqueeze_(dim=0)

        B, C, H, W = im.shape
        ims = im.split(1) 
        poss = pos.split(1)
        size_whs = size_wh.split(1)
        out_size = (out_size, out_size)

        out_ims = []
        for im, pos, sz in zip(ims, poss, size_whs):
            avg = im.mean()

            c = (sz + 1) / 2
            context_xmin = torch.round(pos[0,0] - c)
            context_xmax = context_xmin + sz - 1
            context_ymin = torch.round(pos[0,1] - c)
            context_ymax = context_ymin + sz - 1
            left_pad = int(max(0., -context_xmin))
            top_pad = int(max(0., -context_ymin))
            right_pad = int(max(0., context_xmax - W + 1))
            bottom_pad = int(max(0., context_ymax - H + 1))

            context_xmin = max(int(context_xmin), 0)
            context_xmax = min(int(context_xmax), W-1)
            context_ymin = max(int(context_ymin), 0)
            context_ymax = min(int(context_ymax), H-1)

            im = torch.narrow(im, 2, context_ymin, context_ymax-context_ymin+1)
            im = torch.narrow(im, 3, context_xmin, context_xmax-context_xmin+1)
            im_sub = F.pad(im, pad=(left_pad, right_pad, top_pad, bottom_pad), mode='constant', value=avg) 
            im_sub = F.interpolate(im_sub, size=out_size, mode='bilinear')
            out_ims.append(im_sub)
        
        return torch.cat(out_ims)


class PenaltyLayer(torch.nn.Module):
    '''
    Penal size change and moving
    Forward Input:
        score: (B, 10, 25, 25)
        delta: (B, 20, 25, 25)
        target_sz: (B, 2)
    Forward Output:
        
    '''
    def __init__(self, anchor, p):
        super(PenaltyLayer, self).__init__()

        self.anchor = torch.nn.Parameter(anchor, requires_grad=False)
        self.penalty_k = torch.nn.Parameter(torch.tensor(p.penalty_k), requires_grad=False)
        self.window_influence = torch.nn.Parameter(torch.tensor(p.window_influence), requires_grad=False)
 
        if p.windowing == 'cosine':
            window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
        elif p.windowing == 'uniform':
            window = np.ones((p.score_size, p.score_size))
        window = np.tile(window.flatten(), p.anchor_num)
        self.window = torch.nn.Parameter(torch.from_numpy(window), requires_grad=False)

        self.context_amount = p.context_amount
        self.exemplar_size = p.exemplar_size

    def forward(self, score, delta, target_sz):
        B = score.shape[0]
        delta = delta.view(B, 4, -1)
        score = F.softmax(score.view(B, 2, -1), dim=1)[:,1]

        anchor = self.anchor.clone().detach().requires_grad_(False)
        delta[:, 0, :] = delta[:, 0, :] * anchor[:, 2] + anchor[:, 0]
        delta[:, 1, :] = delta[:, 1, :] * anchor[:, 3] + anchor[:, 1]
        delta[:, 2, :] = torch.exp(delta[:, 2, :]) * anchor[:, 2]
        delta[:, 3, :] = torch.exp(delta[:, 3, :]) * anchor[:, 3]

        def change(r):
            return torch.max(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return torch.sqrt(sz2)

        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return torch.sqrt(sz2)

        # size penalty
        scale_x = self.get_scale_x(target_sz)
        target_sz_in_crop = (target_sz*scale_x).clone().detach().split(1)
        s_c = change(sz(delta[:, 2, :], delta[:, 3, :]) / (sz_wh(target_sz_in_crop)))  # scale penalty
        r_c = change((target_sz_in_crop[0] / target_sz_in_crop[1]) / (delta[:, 2, :] / delta[:, 3, :]))  # ratio penalty

        penalty = torch.exp(-(r_c * s_c - 1) * self.penalty_k)
        pscore_size = penalty * score

        # cos window (motion model)
        pscore = pscore_size * (1 - self.window_influence) + self.window * self.window_influence

        return pscore, delta, pscore_size

    def get_scale_x(self, target_sz):
        device = target_sz.device
        target_sz = target_sz.cpu().numpy()
        
        if len(target_sz.shape) == 1:
            target_sz = np.expand_dims(target_sz, axis=0)

        wc = target_sz[:,0] + self.context_amount * target_sz.sum(axis=1)
        hc = target_sz[:,1] + self.context_amount * target_sz.sum(axis=1)
        crop_sz = np.sqrt(wc * hc).round() 
        scale_x = self.exemplar_size / crop_sz

        return torch.from_numpy(scale_x).requires_grad_(False).to(device)


class Tracker(Custom):
    def __init__(self, p, pretrain=False, **kwargs):
        super(Tracker, self).__init__(pretrain=False, **kwargs)
        self.p = p
        self.get_subwindow = SubWindow()
        self.set_all_anchors(0, size=p.score_size)
        self.penalty = PenaltyLayer(anchor=self.all_anchors, p=p)

    def set_all_anchors(self, image_center, size):
        self.anchor.generate_all_anchors(image_center, size)
        all_anchors = self.anchor.all_anchors[1].reshape(4, -1).transpose(1, 0)
        self.all_anchors = torch.from_numpy(all_anchors).float()

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

    def template(self, template_img, template_pos, template_sz):
        crop_sz = self.get_crop_sz(template_sz)
        self.template_cropped = self.get_subwindow(template_img, template_pos, crop_sz, out_size=self.p.exemplar_size)
        self.zf = self.features(self.template_cropped)

    def track(self, search_img, target_pos, target_sz):
        crop_sz = self.get_crop_sz(target_sz, is_search=True)
        self.search_cropped = self.get_subwindow(search_img, target_pos, crop_sz, out_size=self.p.instance_size)
        search = self.features(self.search_cropped)
        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, search)
        pscore, delta, pscore_size = self.penalty(rpn_pred_cls, rpn_pred_loc, target_sz)
        return pscore, delta, pscore_size


def tracker_init(im, target_pos, target_sz, model, device='cpu'):
    state = dict()
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]
 
    # initialize the exemplar
    model.template(kornia.image_to_tensor(im).to(device).float(), 
                   torch.from_numpy(target_pos).to(device),
                   torch.from_numpy(target_sz).to(device) )
    
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz

    return state


def tracker_track(state, im, model, device='cpu', debug=False):
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    p = model.p
    wc_x = target_sz[1] + p.context_amount * sum(target_sz)
    hc_x = target_sz[0] + p.context_amount * sum(target_sz)
    s_x = np.sqrt(wc_x * hc_x)
    scale_x = p.exemplar_size / s_x

    pscore, delta, pscore_size = model.track(kornia.image_to_tensor(im).to(device).float(),
                                                torch.from_numpy(target_pos).to(device),
                                                torch.from_numpy(target_sz).to(device))

    best_pscore_id = np.argmax(pscore.squeeze().detach().cpu().numpy())

    pred_in_crop = delta.squeeze().detach().cpu().numpy()[:, best_pscore_id] / scale_x
    lr = pscore_size.squeeze().detach().cpu().numpy()[best_pscore_id] * p.lr  # lr for OTB

    res_x = pred_in_crop[0] + target_pos[0]
    res_y = pred_in_crop[1] + target_pos[1]
    res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
    res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])

    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))

    state['target_pos'] = target_pos
    state['target_sz'] = target_sz

    return state


if __name__ == "__main__":
    import glob
    from os.path import join
    import cv2

    # setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # load images and gt
    fpath = '/DataServer/car/car-1/img/'
    N_IMS = 2
    img_files = sorted(glob.glob(join(fpath, '*.jp*')))
    ims = [cv2.imread(imf) for imf in img_files[:N_IMS]]
    ims = torch.tensor(ims, device=device, dtype=float).permute(0,3,1,2)

    with open(fpath + '/../groundtruth.txt', "r") as f:
        gts = f.readlines()
        split_flag = ',' if ',' in gts[0] else '\t'
        gts = list(map(lambda x: list(map(int, x.rstrip().split(split_flag))), gts))
    gts = np.array(gts)[:N_IMS]
    pos = torch.tensor(gts[:,:2] + gts[:,2:]/2, device=device)
    sz = torch.tensor(gts[:,2:], device=device).max(dim=1)[0]

    # test
    model = SubWindow()
    out = model(ims, pos, sz)
    for i in range(out.shape[0]):
        cv2.imshow('im_ori', kornia.tensor_to_image(ims[i].byte()))
        cv2.imshow('im', kornia.tensor_to_image(out[i].byte()))
        cv2.waitKey(0)