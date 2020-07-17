import json
import os
import cv2

from glob import glob
from fire import Fire

def process_one_category(category_name='data/lasot/cup'):

    videos = os.listdir(category_name)
    videos = [i for i in videos if os.path.isdir(os.path.join(category_name, i))]

    # if category_name == 'VOT2016':
    meta_data = {}
    for video in videos:
        with open(os.path.join(category_name, video, "groundtruth.txt"),'r') as f:
            gt_traj = [list(map(float, x.strip().split(','))) for x in f.readlines()]
        with open(os.path.join(category_name, video, "full_occlusion.txt"), 'r') as f:
            full_occ = [list(map(int, x.strip().split(','))) for x in f.readlines()][0]
        with open(os.path.join(category_name, video, "out_of_view.txt"), 'r') as f:
            out_of_view = [list(map(int, x.strip().split(','))) for x in f.readlines()][0]
        assert len(gt_traj)==len(full_occ)==len(out_of_view)

        img_names = sorted(glob(os.path.join(category_name, video, 'img', '*.jp*')))
        im = cv2.imread(img_names[0])
        # img_names = [x.split('/', 1)[1] for x in img_names]

        meta_data[video] = {'category': video.split('-')[0],
                            'init_rect': gt_traj[0],
                            'img_names': img_names,
                            'gt_rect': gt_traj,
                            'full_occ': full_occ,
                            'out_of_view': out_of_view,
                            'width': im.shape[1],
                            'height': im.shape[0]}
    json.dump(meta_data, open(os.path.join(category_name, 'anno.json'), 'w'))
    print('Create annotation json file for ', category_name, ': ', os.path.join(category_name, 'anno.json'))
    return meta_data

def process(dataset_name='data/lasot'):
    categories = os.listdir(dataset_name)
    categories = [i for i in categories if os.path.isdir(os.path.join(dataset_name, i))]
    print(categories)

    anna_data = {}
    for cate in categories:
        meta_data = process_one_category(os.path.join(dataset_name, cate))
        anna_data.update(meta_data)

    json.dump(anna_data, open(os.path.join(dataset_name, 'anno.json'), 'w'))
    print('Create annotation json file: ', os.path.join(dataset_name, 'anno.json'))

if __name__ == '__main__':
    Fire(process)

