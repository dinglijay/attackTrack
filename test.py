from tools.test import * 
from custom import Custom

from attacker import AttackWrapper
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np 
import torch
import torch.nn.functional as F



class Custom_(Custom):
    def __init__(self, pretrain=False, **kwargs):
        super(Custom_, self).__init__(**kwargs)

    def track(self, search, template):
        # Dylan --> Override Custom.track
        self.zf = self.features(template)
        # <--
        search = self.features(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, search)
        return rpn_pred_cls, rpn_pred_loc


def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans):
    """get differentiable subwindow wrt content subimage around pos. 
    im: input numpy image at k-th frame.
    pos: coordinate estimate of target center at (k-1)-th frame, ie, (y,x)
    model_sz: input dim of model. 
    original_sz: original dim. 
    avg_chans: averaged value of channels for padding. 
    Return: (possibly) padded and resized subimage. 
    """
    if isinstance(pos, float):
        pos = [pos, pos]
    hei, wid, chan = im.shape
    length_half = int(round ((original_sz + 1) / 2)) #half length of square box
    avg_chans = np.mean(avg_chans)
    
    x_min, pad_left = get_coordinate_min(pos[1], length_half)
    y_min, pad_top = get_coordinate_min(pos[0], length_half)
    
    x_max, pad_right = get_coordinate_max(pos[1], length_half, wid)
    y_max, pad_bottom = get_coordinate_max(pos[0], length_half, hei)
    
    
    im_sub = torch.Tensor(im[y_min:y_max+1, x_min:x_max+1,:])
    im_sub = im_sub.permute(2, 0, 1).unsqueeze(0)  #[wid, hei, 3] -> [1, 3, hei, wid]
    
    im_sub = F.pad(im_sub, pad=(pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=avg_chans) #padding 
    im_sub = F.interpolate(im_sub, size=(model_sz, model_sz), mode='nearest') #resizing 
    im_sub = im_sub.squeeze(0) #[1, 3, hei, wid] -> [3, hei, wid]
    
    return im_sub
    
    

def get_coordinate_min(coord_center, length_half):
    """get min-side border coordinate and padding number.
    coord_center: a center coordinate. 
    length_half: half length of user-given (processed) square box. 
    """
    coord_center = int(round(coord_center))
    if coord_center - length_half >= 0:
        coord_min = coord_center - length_half
        pad_num = 0
    else:
        coord_min = 0
        pad_num = length_half - coord_center
    return coord_min, pad_num



def get_coordinate_max(coord_center, length_half, org_coord_max):
    """get max-side border coordinate and padding number.
    coord_center: a center coordinate. 
    length_half: half length of user-given (processed) square box. 
    """
    coord_center = int(round(coord_center))
    if coord_center + length_half <= org_coord_max:
        coord_max = coord_center + length_half
        pad_num = 0
    else:
        coord_max = org_coord_max
        pad_num = coord_center + length_half - coord_max 
    return coord_max, pad_num


def siamese_init(im, target_pos, target_sz, model, hp=None, device='cpu'):
    state = dict()
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]
    p = TrackerConfig()
    p.update(hp, model.anchors)

    p.renew()

    net = model
    p.scales = model.anchors['scales']
    p.ratios = model.anchors['ratios']
    p.anchor_num = model.anchor_num
    p.anchor = generate_anchor(model.anchors, p.score_size)
    avg_chans = np.mean(im, axis=(0, 1))

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    # initialize the exemplar
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

    z = Variable(z_crop.unsqueeze(0))
    net.template(z.to(device))

    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)

    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    # Dylan -> Add template and n_frame to state dict
    state['template'] = z.to(device)
    state['n_frame'] = 0
    # <--
    return state


def siamese_track(state, im, mask_enable=False, refine_enable=False, device='cpu', debug=False):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_x = target_sz[1] + p.context_amount * sum(target_sz)
    hc_x = target_sz[0] + p.context_amount * sum(target_sz)
    s_x = np.sqrt(wc_x * hc_x)
    scale_x = p.exemplar_size / s_x
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_x
    s_x = s_x + 2 * pad
    crop_box = [target_pos[0] - round(s_x) / 2, target_pos[1] - round(s_x) / 2, round(s_x), round(s_x)]

    if debug:
        im_debug = im.copy()
        crop_box_int = np.int0(crop_box)
        cv2.rectangle(im_debug, (crop_box_int[0], crop_box_int[1]),
                      (crop_box_int[0] + crop_box_int[2], crop_box_int[1] + crop_box_int[3]), (255, 0, 0), 2)
        cv2.imshow('search area', im_debug)
        cv2.waitKey(0)

    # extract scaled crops for search region x at previous target position
    x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))

    # Dylan --> attack on first frame
    state['n_frame'] += 1
    if state['n_frame'] == 1:
        attacker = AttackWrapper(x_crop.to(device), state, scale_x)
        pert = attacker.attack()
        state['tem_pert'] = torch.clamp(pert, 0, 255).data
        
        plt.figure('img_pert')
        perted_img = pert.data.squeeze().cpu().numpy().transpose(1,2,0)[...,::-1].astype(int)
        plt.imshow(perted_img)
        plt.pause(0.01)
    # <--
    
    if mask_enable:
        score, delta, mask = net.track_mask(x_crop.to(device))
    else:
        # Dylan --> add 'template' to net.track input
        score, delta = net.track(x_crop.to(device), state['tem_pert']) 
        # <--

    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0), dim=1).data[:,
            1].cpu().numpy()

    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

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

    # size penalty
    target_sz_in_crop = target_sz*scale_x
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz_in_crop)))  # scale penalty
    r_c = change((target_sz_in_crop[0] / target_sz_in_crop[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
    pscore = penalty * score

    # cos window (motion model)
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    best_pscore_id = np.argmax(pscore)

    pred_in_crop = delta[:, best_pscore_id] / scale_x
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr  # lr for OTB

    res_x = pred_in_crop[0] + target_pos[0]
    res_y = pred_in_crop[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
    res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])

    # for Mask Branch
    if mask_enable:
        best_pscore_id_mask = np.unravel_index(best_pscore_id, (5, p.score_size, p.score_size))
        delta_x, delta_y = best_pscore_id_mask[2], best_pscore_id_mask[1]

        if refine_enable:
            mask = net.track_refine((delta_y, delta_x)).to(device).sigmoid().squeeze().view(
                p.out_size, p.out_size).cpu().data.numpy()
        else:
            mask = mask[0, :, delta_y, delta_x].sigmoid(). \
                squeeze().view(p.out_size, p.out_size).cpu().data.numpy()

        def crop_back(image, bbox, out_sz, padding=-1):
            a = (out_sz[0] - 1) / bbox[2]
            b = (out_sz[1] - 1) / bbox[3]
            c = -a * bbox[0]
            d = -b * bbox[1]
            mapping = np.array([[a, 0, c],
                                [0, b, d]]).astype(np.float)
            crop = cv2.warpAffine(image, mapping, (out_sz[0], out_sz[1]),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=padding)
            return crop

        s = crop_box[2] / p.instance_size
        sub_box = [crop_box[0] + (delta_x - p.base_size / 2) * p.total_stride * s,
                   crop_box[1] + (delta_y - p.base_size / 2) * p.total_stride * s,
                   s * p.exemplar_size, s * p.exemplar_size]
        s = p.out_size / sub_box[2]
        back_box = [-sub_box[0] * s, -sub_box[1] * s, state['im_w'] * s, state['im_h'] * s]
        mask_in_img = crop_back(mask, back_box, (state['im_w'], state['im_h']))

        target_mask = (mask_in_img > p.seg_thr).astype(np.uint8)
        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 100:
            contour = contours[np.argmax(cnt_area)]  # use max area polygon
            polygon = contour.reshape(-1, 2)
            # pbox = cv2.boundingRect(polygon)  # Min Max Rectangle
            prbox = cv2.boxPoints(cv2.minAreaRect(polygon))  # Rotated Rectangle

            # box_in_img = pbox
            rbox_in_img = prbox
        else:  # empty mask
            location = cxy_wh_2_rect(target_pos, target_sz)
            rbox_in_img = np.array([[location[0], location[1]],
                                    [location[0] + location[2], location[1]],
                                    [location[0] + location[2], location[1] + location[3]],
                                    [location[0], location[1] + location[3]]])

    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))

    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score[best_pscore_id]
    state['mask'] = mask_in_img if mask_enable else []
    state['ploygon'] = rbox_in_img if mask_enable else []
    return state
