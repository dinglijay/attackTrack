import cv2
import glob
import pickle
from os.path import join

from torch.utils.data import Dataset


class AttackDataset(Dataset):

    def __init__(self, root_dir='data/Human1', transform=None):
        self.root_dir = root_dir
        self.img_names = sorted(glob.glob(join(root_dir, 'imgs', '*.jp*')))

        with open(join(root_dir, 'groundtruth_rect.txt'), "r") as f:
            gts = f.readlines()
            split_flag = ',' if ',' in gts[0] else '\t'
            self.bbox = list(map(lambda x: list(map(int, x.rstrip().split(split_flag))), gts))
        
        with open(join(root_dir, 'corners.dat'), 'rb') as f:
            data = pickle.load(f)
            self.rets = data['ret']
            self.corners = data['corners']

        assert len(self.bbox) == len(self.img_names) == self.rets.shape[0] ==self.corners.shape[0]

        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):

        print(idx)

        template_img = cv2.imread(self.img_names[0]) 
        template_bbox = self.bbox[0]
        search_img = cv2.imread(self.img_names[idx+1])
        search_bbox = self.bbox[idx]
        
        return template_img, template_bbox, search_img, search_bbox

if __name__ =='__main__':

    data = AttackDataset()
    template_img, template_bbox, search_img, search_bbox = iter(data).__next__()

    x, y, w, h = template_bbox
    cv2.rectangle(template_img, (x, y), (x+w, y+h), (0, 255, 0), 4)
    cv2.imshow('template', template_img)
    cv2.waitKey(1)

    x, y, w, h = search_bbox
    cv2.rectangle(search_img, (x, y), (x+w, y+h), (0, 0, 255), 4)
    cv2.imshow('search', search_img)

    cv2.waitKey(0)