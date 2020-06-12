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
        masks.append(kornia.warp_perspective(pert_tensor.unsqueeze(0), M, size))
    return torch.cat(masks)


if __name__ == '__main__':

    import cv2
    img = cv2.imread('../SiamMask/data/tennis/00000.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (500,500))
    
    bbox = (100,100,200,100)
    mask = get_bbox_mask(shape=(500,500), bbox=bbox)
    img = img*mask.squeeze()

    mask_warped = warp(kornia.image_to_tensor(img), bbox, scale_bbox(bbox, (0.5, 0.5)))
    cv2.imshow('mask_img', img/255.0)
    cv2.imshow('scaled', get_bbox_mask(shape=(500,500), bbox=scale_bbox(bbox,(0.5, 0.5))).squeeze())
    cv2.imshow('mask_warped', kornia.tensor_to_image(mask_warped.byte()) )
    cv2.waitKey(0)

