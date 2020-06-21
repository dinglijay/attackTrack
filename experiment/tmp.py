import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches

from tracker import bbox2center_sz

def rand_shift(bbox, shift_pos=0.5, shift_wh=0.5):
    ''' Random shift and scale the bbox
    shift_pos is maximum offset of x and y [-shift_pos, shift_pos]
    shift_wh is maximum offset of w and h [0, shift_wh]
    '''
    B = bbox.shape[0]
    x, y, w, h = bbox.detach().split(1, dim=1)
    cx, cy = x+w//2, y+h//2

    delta_x = shift_pos * (2 * torch.rand((B,1), device=bbox.device) - 1)
    delta_y = shift_pos * (2 * torch.rand((B,1), device=bbox.device) - 1)
    delta_w = shift_wh * (torch.rand((B,1), device=bbox.device))
    delta_h = shift_wh * (torch.rand((B,1), device=bbox.device))

    cx = cx + w * delta_x 
    cy = cy + h * delta_y
    w = w * torch.exp(delta_w)
    h = h * torch.exp(delta_h)
    x = cx - w//2
    y = cy - h//2

    return torch.cat([x, y, w, h], dim=1).to(bbox.dtype)

if __name__ == "__main__":
    pass

    search_bbox = torch.tensor([[201, 291, 102, 312]], device='cuda:0')
    # search_bbox = torch.tensor([[201, 291, 102, 312], [257, 274, 154, 456]], device='cuda:0')

    fig, ax = plt.subplots(1,1,num='bbox')
    for i in range(100):
        track_box = rand_shift(search_bbox)
        ax.imshow(np.zeros((960, 540, 3)))
        for i in range(search_bbox.shape[0]):
            x, y, w, h = search_bbox[i].cpu().numpy()
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            x, y, w, h = track_box[i].cpu().numpy()
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect) 
            plt.pause(0.001) 
    plt.show()