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
    
    x,y,w,h = bbox
    assert (x+w)<shape[1] and (y+h)<shape[0]

    mask = np.zeros(shape)
    mask[y:y+h, x:x+w] = 1
    mask = np.expand_dims(mask, axis=2)
    mask = np.broadcast_to(mask, (shape[0],shape[1],3)).astype(np.float32)

    return mask if mode=='numpy' else torch.Tensor(mask).permute(2,0,1).unsqueeze(0)

def scale_bbox(bbox, scale_wh):

    x, y, w, h = bbox
    c_x = x + w/2
    c_y = y + h/2

    scale_w, scale_h = scale_wh
    w *= scale_w
    h *= scale_h
    x = c_x - w/2
    y = c_y - h/2

    return tuple(map(int, (x, y, w, h)))

def warp(im_tensor, bbox_src, bbox_dest):
    x, y, w, h = bbox_src
    points_src = torch.FloatTensor([[[x, y], [x+w, y], [x, y+h], [x+w, y+h],]])
    x, y, w, h = bbox_dest
    points_dst = torch.FloatTensor([[[x, y], [x+w, y], [x, y+h], [x+w, y+h],]])

    M = kornia.get_perspective_transform(points_src, points_dst).to(im_tensor.device)
    size = im_tensor.squeeze().shape[1:]

    return kornia.warp_perspective(im_tensor, M, size)


if __name__ == '__main__':

    import cv2
    img = cv2.imread('../SiamMask/data/tennis/00000.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (500,500))
    
    bbox = (100,100,200,100)
    mask = get_bbox_mask(shape=(500,500), bbox=bbox)
    img = img*mask

    mask_warped = warp(kornia.image_to_tensor(img).unsqueeze(0), bbox, scale_bbox(bbox, 0.5, 0.5))
    cv2.imshow('mask_img', img/255.0)
    cv2.imshow('scaled', get_bbox_mask(shape=(500,500), bbox=scale_bbox(bbox, 0.5, 0.5)))
    cv2.imshow('mask_warped', kornia.tensor_to_image(mask_warped.byte()) )
    cv2.waitKey(0)

