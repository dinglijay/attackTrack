# import json

# json_fpath = 'data/OTB100/OTB100.json'

# with open(json_fpath, 'r') as f:
#     annos = json.load(f)

# print('ok')

import math
import numpy as np
import torch
import cv2


def scale_bbox_keep_ar(bbox, scale_wh, aspect):
    if type(bbox) == np.ndarray or type(bbox) == torch.Tensor:
        bbox = bbox.clone().detach()
        w = bbox[:, 2]
        h = bbox[:, 3]
        c_x = bbox[:, 0] + w // 2
        c_y = bbox[:, 1] + h // 2
    
        scale_w, scale_h = scale_wh
        H = (scale_w * scale_h * w * h * aspect).sqrt()
        H = torch.min(h, H.to(int))
        W = (scale_w * scale_h * w * h / aspect).sqrt()
        W = torch.min(w, W.to(int))

        bbox[:, 0] = c_x - W//2
        bbox[:, 1] = c_y - H//2
        bbox[:, 2] = W
        bbox[:, 3] = H
        return bbox
    else:
        x, y, w, h = bbox
        c_x = x + w//2
        c_y = y + h//2

        scale_w, scale_h = scale_wh
        H = math.sqrt(scale_w * scale_h * w * h * aspect)
        H = min(H, h)
        W = math.sqrt(scale_w * scale_h * w * h / aspect)
        W = min(W, w)
        X = c_x - W//2
        Y = c_y - H//2
        return tuple(map(int, (X, Y, W, H)))

if __name__ == "__main__":
        
    for i in range(100):

        img = np.zeros((500,500,3 ))

        x, y = 100, 100
        w = np.random.randint(50, 300)
        h = np.random.randint(50, 300)
        bbox = x, y, w, h

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)

        bbox = torch.tensor(bbox).reshape(-1, 4)
        bbox_ = scale_bbox_keep_ar(bbox, (0.5, 0.5), 2)
        print(bbox, bbox_)

        x, y, w, h = bbox_.reshape(-1)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)


        cv2.imshow('img', img)
        cv2.waitKey(0)
