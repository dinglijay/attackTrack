import torch
import numpy as np

from scipy.stats import truncnorm
from matplotlib import pyplot as plt
from matplotlib import patches

def rand_shift_bk(img_hw, bbox, shift_pos=(-0.2, 0.2), shift_wh=(-0.05, 0.9), cor=0.3):
    ''' Random shift and scale the bbox
    shift_pos and shift_wh is offset range
    cor is the correlation coefficient of delta_w and delta_h
    '''
    device = bbox.device
    B = bbox.shape[0]
    x, y, w, h = bbox.detach().split(1, dim=1)
    cx, cy = x+w//2, y+h//2

    delta_x = shift_pos[0] + (shift_pos[1] - shift_pos[0]) * torch.rand((B,1)).to(device)
    delta_y = shift_pos[0] + (shift_pos[1] - shift_pos[0]) * torch.rand((B,1)).to(device)

    # delta_w = shift_wh[0] + (shift_wh[1] - shift_wh[0]) * torch.rand((B,1), device=device)
    # ratio = 1.6*w//h
    # delta_h = ratio * delta_w + cor * (torch.rand((B, 1), device=device)*2 -1)
    # delta_h = delta_h.clamp(shift_wh[0], shift_wh[1])

    delta_w = shift_wh[0] + (shift_wh[1] - shift_wh[0]) * torch.rand((B,1)).to(device)
    delta_h = shift_wh[0] + (shift_wh[1] - shift_wh[0]) * torch.rand((B,1)).to(device)

    cx = cx + w * delta_x
    cy = cy + h * delta_y
    w = w * (1 + delta_w)
    h = h * (1 + delta_h)
    x = cx - w//2
    y = cy - h//2

    # Limi
    img_h, img_w = img_hw
    x = x.clamp(0, img_w-1)
    y = y.clamp(0, img_h-1)
    w = torch.min(w, img_w-x)
    h = torch.min(h, img_h-y)

    return torch.cat([x, y, w, h], dim=1).to(bbox.dtype)

def rand_shift(img_hw, bbox, shift_pos=(-0.1, 0.1), shift_wh=(-0.3, 0.1), target='small', pos=(0.0, 0.0)):
    ''' Random shift and scale the bbox
    shift_pos and shift_wh is offset range
    target: small or large
    '''
    device = bbox.device
    B = bbox.shape[0]
    x, y, w, h = bbox.detach().split(1, dim=1)
    cx, cy = x+w//2, y+h//2

    delta_x = shift_pos[0] + (shift_pos[1] - shift_pos[0]) * torch.rand((B,1)).to(device)
    delta_y = shift_pos[0] + (shift_pos[1] - shift_pos[0]) * torch.rand((B,1)).to(device)

    if target == 'large':
        delta_w = torch.tensor(truncnorm.rvs(shift_wh[0], shift_wh[1], size=(B,1)), device=device)
        delta_h = torch.randn_like(delta_w) / 30 + delta_w
    elif target == 'small':
        delta_w = torch.tensor(truncnorm.rvs(shift_wh[0], shift_wh[1], size=(B,1)), device=device)
        delta_h = torch.randn_like(delta_w) / 30 + delta_w*2
        delta_y += delta_h/5
        # delta_y += delta_h/2.0
        # delta_x += delta_w/2.0
    elif target == 'random':
        delta_w = torch.tensor(truncnorm.rvs(shift_wh[0], shift_wh[1], size=(B,1)), device=device)
        delta_h = torch.tensor(truncnorm.rvs(shift_wh[0], shift_wh[1], size=(B,1)), device=device)
    else:
        raise

    cx = cx + w * delta_x
    cy = cy + h * delta_y
    w = w * (1 + delta_w)
    h = h * (1 + delta_h)
    x = cx - w//2
    y = cy - h//2

    # Limitation
    img_h, img_w = img_hw
    x = x.clamp(0, img_w-1)
    y = y.clamp(0, img_h-1)
    w = torch.min(w, img_w-x)
    h = torch.min(h, img_h-y)

    return torch.cat([x, y, w, h], dim=1).to(bbox.dtype)

if __name__ == "__main__":
    search_bbox = torch.tensor([[568,125,230,360]], device='cuda:0')//4
    img_hw = np.array([720,1080]) // 4

    fig, ax = plt.subplots(1,1,num='bbox')
    for i in range(1000):
        track_box = rand_shift(img_hw, search_bbox,  (-0.02, 0.02), (-0.02, 0.02), target='random')
        ax.imshow(np.zeros((img_hw[0], img_hw[1], 3)))
        for i in range(search_bbox.shape[0]):
            x, y, w, h = search_bbox[i].cpu().numpy()
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            x, y, w, h = track_box[i].cpu().numpy()
            rect = patches.Rectangle((x, y), w, h, linewidth=5, edgecolor='y', facecolor='none')
            ax.add_patch(rect) 
            plt.pause(0.001) 
            rect.remove()
    plt.show()
