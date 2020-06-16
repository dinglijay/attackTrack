import numpy as np
import torch
import kornia

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


def scale_bbox(bbox, scale_wh):
    # todo: unify operation on tensor and int list

    if type(bbox) == np.ndarray or type(bbox) == torch.Tensor:
        bbox = bbox.clone().detach()
        c_x = bbox[:, 0] + bbox[:, 2]/2
        c_y = bbox[:, 1] + bbox[:, 3]/2
        scale_w, scale_h = scale_wh
        bbox[:, 2] = bbox[:, 2] * scale_w
        bbox[:, 3] = bbox[:, 3] * scale_h
        bbox[:, 0] = c_x - bbox[:, 2]/2
        bbox[:, 1] = c_y - bbox[:, 3]/2
        return bbox
    else:
        x, y, w, h = bbox
        c_x = x + w/2
        c_y = y + h/2

        scale_w, scale_h = scale_wh
        w *= scale_w
        h *= scale_h
        x = c_x - w/2
        y = c_y - h/2
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
    Input: patch_tensor : Tensor (3, h0, w0)
           img_tensor: Tensor(B, 3, H, W) 
           bbox_dest: Tensor(B, 4)
    Output: Tensor (B, 3, H, W)
    '''
    B = bbox_dest.shape[0]

    x, y, w, h = 0, 0, patch_tensor.shape[2], patch_tensor.shape[1]
    points_src = torch.FloatTensor([[[x, y], [x+w-1, y], [x, y+h-1], [x+w-1, y+h-1],]])
    points_src = points_src.expand(bbox_dest.shape[0],4,2).to(img_tensor.device)
    
    xy  = torch.stack((bbox_dest[:,0],bbox_dest[:,1]), dim=1)
    x2y = torch.stack((bbox_dest[:,0]+bbox_dest[:,2]-1, bbox_dest[:,1]),dim=1)
    xy2 = torch.stack((bbox_dest[:,0], bbox_dest[:,1]+bbox_dest[:,3]-1),dim=1)
    x2y2 = torch.stack((bbox_dest[:,0]+bbox_dest[:,2]-1, bbox_dest[:,1]+bbox_dest[:,3]-1),dim=1)
    points_dst = torch.stack([xy, x2y, xy2, x2y2], dim=1).to(torch.float32)

    M = kornia.get_perspective_transform(points_src, points_dst).to(img_tensor.device)

    # patch_tensor = patch_tensor.expand(B, -1, -1, -1)
    patch_tensor = torch.stack([patch_tensor for i in range(B)], dim=0)
    patch_warped = kornia.warp_perspective(patch_tensor, M, (img_tensor.shape[2], img_tensor.shape[3]))
    # res_img = torch.where((patch_warped==0), img_tensor, patch_warped)

    return patch_warped


def add_patch(patch_tensor, img_tensor, bbox_dest):
    '''
    Apply the patch to images.
    Input: patch_tensor : Tensor (3, h0, w0)
           img_tensor: Tensor(B, 3, H, W) 
           bbox_dest: Tensor(B, 4)
    Output: Tensor (B, 3, H, W)
    '''
    B = bbox_dest.shape[0]
    masks = list()
    for i in range(B):
        size = img_tensor.shape[-2:]
        pad_h, pad_w = (np.array(size) - np.array(patch_tensor.shape[-2:]) ) / 2
        mypad = torch.nn.ConstantPad2d((int(pad_w + 0.5), int(pad_w), int(pad_h + 0.5), int(pad_h)), 0)

        masks.append(mypad(patch_tensor.unsqueeze(0)))
    return torch.cat(masks)

if __name__ == '__main__':

    import cv2

    img = cv2.imread('data/Human1/imgs/0001.jpg')
    img2 = cv2.imread('data/Human1/imgs/0100.jpg')
    patch = cv2.imread('data/patchnew0.jpg')
    H, W = 400, 300
    patch = cv2.resize(patch, (W,H)) # W, H
    bbox = [[419,605,207,595], [410,557,310,856]]

    cv2.namedWindow('img', cv2.WND_PROP_FULLSCREEN)
    x, y, w, h = bbox[0]
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 4)
    cv2.imshow('img', img)

    cv2.namedWindow('img2', cv2.WND_PROP_FULLSCREEN)
    x, y, w, h = bbox[1]
    cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 255, 0), 4)
    cv2.imshow('img2', img2)

    cv2.imshow('patch', patch)


    img_tensor = kornia.image_to_tensor(img).unsqueeze(0).to(torch.float32)
    img_tensor2 = kornia.image_to_tensor(img2).unsqueeze(0).to(torch.float32)
    patch_tensor = kornia.image_to_tensor(patch).to(torch.float32)
    bbox = torch.tensor(bbox)
    bbox_dest = scale_bbox(bbox, (0.6, 0.3))
    
    res_img = warp_patch2(patch_tensor, torch.cat([img_tensor, img_tensor2], 0), bbox_dest)

    cv2.namedWindow('res_img1', cv2.WND_PROP_FULLSCREEN)
    cv2.namedWindow('res_img2', cv2.WND_PROP_FULLSCREEN)
    cv2.imshow('res_img1', kornia.tensor_to_image(res_img[0].byte()))
    cv2.imshow('res_img2', kornia.tensor_to_image(res_img[1].byte()))

    cv2.waitKey(0)

