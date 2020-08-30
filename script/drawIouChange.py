import os
from utils.bbox_helper import IoU, center2corner
import numpy as np
from matplotlib import pyplot as plt

def xywh2corner(xywh):
    x, y, w, h = xywh[:,0], xywh[:,1], xywh[:,2], xywh[:,3]
    return np.array([x, y, x+w, y+h])

video_path = 'data/lasot/person/person-16'

with open(os.path.join(video_path, "attacked_bbox.txt"),'r') as f:
    attacked_bbox = [list(map(float, x.strip().split(','))) for x in f.readlines()]

with open(os.path.join(video_path, "attacked_bbox_sm.txt"),'r') as f:
    attacked_bbox_sm = [list(map(float, x.strip().split(','))) for x in f.readlines()]

with open(os.path.join(video_path, "original_bbox.txt"),'r') as f:
    original_bbox = [list(map(float, x.strip().split(','))) for x in f.readlines()]

with open(os.path.join(video_path, "groundtruth.txt"),'r') as f:
    gt_bbox = [list(map(float, x.strip().split(','))) for x in f.readlines()]

attacked_bbox = np.array(attacked_bbox)
attacked_bbox_sm = np.array(attacked_bbox_sm)
original_bbox = np.array(original_bbox)
gt_bbox = np.array(gt_bbox)

results = IoU(xywh2corner(attacked_bbox), xywh2corner(gt_bbox))
results_sm = IoU(xywh2corner(attacked_bbox_sm), xywh2corner(gt_bbox))
ori_results = IoU(xywh2corner(original_bbox), xywh2corner(gt_bbox))


x = np.arange(results.shape[0])


plt.figure('result')
plt.plot(results, label='Dilation Attack')
plt.plot(results_sm, label='Shrink Attack')
plt.plot(ori_results, label='Clean')
plt.plot(x[results==1.0], np.ones_like(x[results==1.0]), 'ro', label='Initialization')
plt.ylabel('IoU')
plt.xlabel('#Frame')

plt.legend()
plt.show()



