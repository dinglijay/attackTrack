import torch
import torch.nn.functional as F

class SubWindow(torch.nn.Module):
    """
    Crop image at pos with size_wh: Croppping + Padding + Resizing.
    Forward Input:
        im: input images, torch.tensor of shape (B, C, H, W)
        pos: crop postions, torch.tensor of shape (B, 2)
        size_wh: crop sizes, torch.tensor of shape (B, )
    Forward Output:
        cropped images (B, C, model_sz, model_sz)
    To do:
        --> Looping through each image seems like stupid
        --> How resize mode matters? 
    """
    def __init__(self, out_size=127):
        super(SubWindow, self).__init__()
        self.out_size = (out_size, out_size)

    def forward(self, im, pos, size_wh):

        B, C, H, W = im.shape
        ims = im.split(1) 
        poss = pos.split(1)
        size_whs = size_wh.split(1)

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
            im_sub = F.interpolate(im_sub, size=self.out_size, mode='bilinear')
            out_ims.append(im_sub)
        
        return torch.stack(out_ims)


if __name__ == "__main__":
    import glob
    from os.path import join
    import cv2
    import kornia
    import numpy as np

    # setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # load images and gt
    fpath = '/DataServer/car/car-1/img/'
    N_IMS = 10
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
    model = SubWindow(out_size=255)
    out = model(ims, pos, sz)
    for i in range(out.shape[0]):
        cv2.imshow('im_ori', kornia.tensor_to_image(ims[i].byte()))
        cv2.imshow('im', kornia.tensor_to_image(out[i].byte()))
        cv2.waitKey(0)