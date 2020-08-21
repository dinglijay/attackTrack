import math
import numpy as np
import torch
import kornia
import torch.nn.functional as F

def get_circle_mask(shape=(127,127), loc=(64,64), diameter=12, sharpness=40):
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
    mask[y1:y2, x1:x2] = circle
    mask = np.expand_dims(mask, axis=2)
    mask = np.broadcast_to(mask, (shape[0],shape[1],3)).astype(np.float32)
  
    return mask

def get_bbox_mask(shape=(127,127), bbox=(50,50,20,20), mode='numpy'):
    """Return a rectangle mask of a given shape"""
    if type(bbox) == torch.Tensor:
        bbox = bbox.cpu()
    bbox = np.array(bbox).reshape(-1,4)
    masks = list()
    for i in range(bbox.shape[0]):
        x,y,w,h = bbox[i]
        assert (x+w)<shape[1] and (y+h)<shape[0]
        mask = np.zeros(shape)
        mask[y:y+h, x:x+w] = 1
        mask = np.expand_dims(mask, axis=2)
        masks.append(np.broadcast_to(mask, (shape[0],shape[1],3)).astype(np.float32))

    return np.array(masks) if mode=='numpy' else torch.tensor(masks).permute(0,3,1,2)

def get_bbox_mask_tv(shape=(127,127), bbox=(50,50,20,20)):
    """Return a rectangle mask of a given shape"""
    masks = list()
    for i in range(bbox.shape[0]):
        x,y,w,h = bbox[i]
        assert (x+w)<shape[1] and (y+h)<shape[0]
        
        mask = torch.zeros(shape, device='cuda')
        mask[y:y+h, x:x+w] = 1
        mask = mask.expand(3,-1,-1)
        masks.append(mask)
    return torch.stack(masks, dim=0)

def scale_bbox_keep_ar(bbox, scale_wh, aspect, delta_xy=(0.0, 0.0)):
    if type(bbox) == np.ndarray or type(bbox) == torch.Tensor:
        bbox = bbox.clone().detach()
        w = bbox[:, 2]
        h = bbox[:, 3]
        c_x = bbox[:, 0] + w // 2
        c_y = bbox[:, 1] + h // 2

        c_x = c_x + w * delta_xy[0]
        c_y = c_y + h * delta_xy[1]
    
        scale_w, scale_h = scale_wh
        H = (scale_w * scale_h * w * h * aspect).sqrt()
        H = torch.min(h*(1-abs(2*delta_xy[1])), H)
        W = (scale_w * scale_h * w * h / aspect).sqrt()
        W = torch.min(w*(1-abs(2*delta_xy[0])), W)

        bbox[:, 0] = c_x - W//2
        bbox[:, 1] = c_y - H//2
        bbox[:, 2] = W
        bbox[:, 3] = H
        return bbox
    else:
        x, y, w, h = bbox
        c_x = x + w//2
        c_y = y + h//2

        c_x = c_x + w * delta_xy[0]
        c_y = c_y + h * delta_xy[1]

        scale_w, scale_h = scale_wh
        H = math.sqrt(scale_w * scale_h * w * h * aspect)
        H = min(H, h*(1-abs(2*delta_xy[1])))
        W = math.sqrt(scale_w * scale_h * w * h / aspect)
        W = min(W, w*(1-abs(2*delta_xy[0])))
        X = c_x - W//2
        Y = c_y - H//2
        return tuple(map(int, (X, Y, W, H)))

def scale_bbox(bbox, scale_wh, aspect):
    # pseudo argument --> aspect

    if type(bbox) == np.ndarray or type(bbox) == torch.Tensor:
        bbox = bbox.clone().detach()
        c_x = bbox[:, 0] + bbox[:, 2]//2
        c_y = bbox[:, 1] + bbox[:, 3]//2
        scale_w, scale_h = scale_wh
        bbox[:, 2] = bbox[:, 2] * scale_w
        bbox[:, 3] = bbox[:, 3] * scale_h
        bbox[:, 0] = c_x - bbox[:, 2]//2
        bbox[:, 1] = c_y - bbox[:, 3]//2
        return bbox
    else:
        x, y, w, h = bbox
        c_x = x + w//2
        c_y = y + h//2

        scale_w, scale_h = scale_wh
        w *= scale_w
        h *= scale_h
        x = c_x - w//2
        y = c_y - h//2
        return tuple(map(int, (x, y, w, h)))

def warp(pert_tensor, bbox_src, bbox_dest):
    '''
    Input: pert_tensor : Tensor (3, W, H)
           bbox_src and bbox_dest: (B, 4)
    Output: Tensor (B, 3, W, H)
    '''

    if type(bbox_src) == torch.Tensor:
        bbox_src = bbox_src.cpu()
        bbox_dest = bbox_dest.cpu()
    bbox_src = np.array(bbox_src).reshape(-1,4)
    bbox_dest = np.array(bbox_dest).reshape(-1,4)
    
    masks = list()
    for i in range(bbox_src.shape[0]):
        x, y, w, h = bbox_src[i]
        points_src = torch.FloatTensor([[[x, y], [x+w, y], [x, y+h], [x+w, y+h],]])
        x, y, w, h = bbox_dest[i]
        points_dst = torch.FloatTensor([[[x, y], [x+w, y], [x, y+h], [x+w, y+h],]])

        M = kornia.get_perspective_transform(points_src, points_dst).to(pert_tensor.device)
        size = pert_tensor.shape[-2:]
        masks.append(kornia.warp_perspective(pert_tensor, M, size))
    return torch.cat(masks)

def warp_patch(patch_tensor, img_tensor, bbox_dest):
    '''
    Apply the patch to images.
    Input: patch_tensor : Tensor ([B, ]3, h0, w0)
           img_tensor: Tensor(B, 3, H, W) 
           bbox_dest: Tensor(B, 4)
    Output: Tensor (B, 3, H, W)
    '''
    B = bbox_dest.shape[0]

    x, y, w, h = 0, 0, patch_tensor.shape[-1], patch_tensor.shape[-2]
    points_src = torch.FloatTensor([[[x, y], [x+w-1, y], [x, y+h-1], [x+w-1, y+h-1],]])
    points_src = points_src.expand(bbox_dest.shape[0],4,2).to(img_tensor.device)
    
    xy  = torch.stack((bbox_dest[:,0],bbox_dest[:,1]), dim=1)
    x2y = torch.stack((bbox_dest[:,0]+bbox_dest[:,2]-1, bbox_dest[:,1]),dim=1)
    xy2 = torch.stack((bbox_dest[:,0], bbox_dest[:,1]+bbox_dest[:,3]-1),dim=1)
    x2y2 = torch.stack((bbox_dest[:,0]+bbox_dest[:,2]-1, bbox_dest[:,1]+bbox_dest[:,3]-1),dim=1)
    points_dst = torch.stack([xy, x2y, xy2, x2y2], dim=1).to(torch.float32)

    M = kornia.get_perspective_transform(points_src, points_dst).to(img_tensor.device)

    if len(patch_tensor.shape) == 3:
        patch_tensor = patch_tensor.expand(B, -1, -1, -1)
    patch_warped = kornia.warp_perspective(patch_tensor, M, (img_tensor.shape[2], img_tensor.shape[3]))
    # res_img = torch.where((patch_warped==0), img_tensor, patch_warped)

    return patch_warped

def pad_patch(patch_tensor, img_tensor, bbox_dest):
    '''
    Pad the patch to the size of img_tensor.
    Input: patch_tensor : Tensor ([B, ]3, h, w)
           img_tensor: Tensor(B, 3, H, W) 
           bbox_dest: Tensor(B, 4)
    Output: Tensor (B, 3, H, W)
    '''
    B = bbox_dest.shape[0]
    if len(patch_tensor.shape) == 3:
        patch_tensor = patch_tensor.expand(B, -1, -1, -1)

    size = img_tensor.shape[-2:]
    pad_h, pad_w = (np.array(size) - np.array(patch_tensor.shape[-2:]) ) / 2
    patch_paded = F.pad(patch_tensor, pad=(int(pad_w+0.5), int(pad_w), int(pad_h+0.5), int(pad_h)), value=0) 

    return patch_paded

if __name__ == '__main__':

    import cv2

    img = cv2.imread('data/own/Human/Human1/img/0001.jpg')
    img2 = cv2.imread('data/own/Human/Human1/img/0100.jpg')
    H, W = 200, 400
    patch1 = cv2.resize(cv2.imread('patches/patch_la.png'), (W,H))
    patch2 = cv2.resize(cv2.imread('patches/patch_sm.png'), (W,H))
    bbox = [[200,200,200,400], [300,100,200,200]]

    cv2.namedWindow('img', cv2.WND_PROP_FULLSCREEN)
    x, y, w, h = bbox[0]
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 4)
    cv2.imshow('img', img)

    cv2.namedWindow('img2', cv2.WND_PROP_FULLSCREEN)
    x, y, w, h = bbox[1]
    cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 255, 0), 4)
    cv2.imshow('img2', img2)

    # cv2.imshow('patch1', patch1)
    # cv2.imshow('patch2', patch2)


    img_tensor = kornia.image_to_tensor(img).unsqueeze(0).to(torch.float32)
    img_tensor2 = kornia.image_to_tensor(img2).unsqueeze(0).to(torch.float32)
    patch_tensor1 = kornia.image_to_tensor(patch1).to(torch.float32)
    patch_tensor2 = kornia.image_to_tensor(patch2).to(torch.float32)
    bbox = torch.tensor(bbox).to(torch.float)
    bbox_dest = scale_bbox_keep_ar(bbox, (0.5, 0.5), 1.0, (0, -0.25))

    print(bbox)
    print(bbox_dest)

    # mask = get_bbox_mask_tv(img_tensor.shape[-2:], bbox)
    # cv2.namedWindow('mask1', cv2.WND_PROP_FULLSCREEN)
    # cv2.namedWindow('mask2', cv2.WND_PROP_FULLSCREEN)
    # cv2.imshow('mask1', kornia.tensor_to_image(mask[0]))
    # cv2.imshow('mask2', kornia.tensor_to_image(mask[1]))


    res_img = warp_patch(torch.stack([patch_tensor1, patch_tensor2], 0), torch.cat([img_tensor, img_tensor2], 0), bbox_dest)

    cv2.namedWindow('res_img1', cv2.WND_PROP_FULLSCREEN)
    cv2.namedWindow('res_img2', cv2.WND_PROP_FULLSCREEN)
    cv2.imshow('res_img1', kornia.tensor_to_image(res_img[0].byte()))
    cv2.imshow('res_img2', kornia.tensor_to_image(res_img[1].byte()))

    cv2.waitKey(0)

