import cv2
import glob
import pickle
from os.path import join
from torch.utils.data import Dataset, DataLoader
import numpy as np


class AttackDataset(Dataset):

    def __init__(self, root_dir='data/Car1', step=1, transform=None):
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
        self.step = step

    def __len__(self):
        return len(self.img_names) - self.step 

    def __getitem__(self, idx):
        template_idx = 0
        search_idx = idx + self.step
        print(self.img_names[template_idx], self.img_names[search_idx])
        
        template_img = np.transpose(cv2.imread(self.img_names[template_idx]), (2, 0, 1)).astype(np.float32)
        search_img = np.transpose(cv2.imread(self.img_names[search_idx]), (2, 0, 1)).astype(np.float32)
        template_bbox = np.array(self.bbox[template_idx])
        search_bbox = np.array(self.bbox[search_idx])
       
        return template_img, template_bbox, search_img, search_bbox

if __name__ =='__main__':
    import kornia

    dataset = AttackDataset(step=50)
    dataloader = DataLoader(dataset, batch_size=2)

    cv2.namedWindow("template", cv2.WND_PROP_FULLSCREEN)
    cv2.namedWindow("search", cv2.WND_PROP_FULLSCREEN)

    for data in dataloader:
        data = list(map(lambda x: x.split(1), data))
        for template_img, template_bbox, search_img, search_bbox in zip(*data):
            x, y, w, h = template_bbox.squeeze()
            template_img = np.ascontiguousarray(kornia.tensor_to_image(template_img.byte()))
            cv2.rectangle(template_img, (x, y), (x+w, y+h), (0, 0, 255), 4)
            cv2.imshow('template', template_img)
            cv2.waitKey(1)

            x, y, w, h = search_bbox.squeeze()
            search_img = np.ascontiguousarray(kornia.tensor_to_image(search_img.byte()))
            cv2.rectangle(search_img, (x, y), (x+w, y+h), (0, 255, 0), 4)
            cv2.imshow('search', search_img)

            cv2.waitKey(0)