import cv2
import glob
import json
from os.path import join
from torch.utils.data import Dataset, DataLoader
import numpy as np


def permutations(iterable, max_dist=10):
    # generate indices pairs with distance limitation
    r = 2
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r
    if r > n:
        return
    indices = list(range(n))
    cycles = list(range(n, n-r, -1))
    yield tuple(pool[i] for i in indices[:r])
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i+1:] + indices[i:i+1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                if abs(indices[1]-indices[0]) > max_dist:
                    break
                yield tuple(pool[i] for i in indices[:r])
                break
        else:
            return

class AttackDataset(Dataset):

    def __init__(self, root_dir='data/lasot/cup/cup-7', n_frames=None, step=1, test=False):
        with open(join(root_dir, 'anno.json'), 'r') as f:
            annos = json.load(f)

        self.img_names = list()
        self.bboxs = list()
        for anno in annos.values():
            if not n_frames:
                self.img_names.extend(anno['img_names'])
                self.bboxs.extend(anno['gt_rect'])
            else:
                indices = np.random.choice(len(anno['img_names']), n_frames, replace=False)
                indices.sort()
                self.img_names.extend([anno['img_names'][idx] for idx in indices])
                self.bboxs.extend([anno['gt_rect'][idx] for idx in indices])
        assert len(self.bboxs) == len(self.img_names)

        self.imgs = None
        if len(self.img_names) < 1000:
            self.imgs = [np.transpose(cv2.imread(im_name).astype(np.float32), (2, 0, 1)) \
                         for im_name in self.img_names]

        self.step = step
        self.test = test
    
    def gen_ind_combinations(self):
        n_imgs = len(self.img_names)
        list(permutations(n_imgs, 5))

    def __len__(self):
        return len(self.img_names) - self.step 

    def __getitem__(self, idx):
        template_idx = 0 if self.test else np.random.randint(self.__len__())
        search_idx = idx + self.step
        # print(self.img_names[template_idx], self.img_names[search_idx])
        
        if self.imgs:
            template_img = self.imgs[template_idx]
            search_img = self.imgs[search_idx]
        else:
            template_img = np.transpose(cv2.imread(self.img_names[template_idx]).astype(np.float32), (2, 0, 1))
            search_img = np.transpose(cv2.imread(self.img_names[search_idx]).astype(np.float32), (2, 0, 1))
        template_bbox = np.array(self.bboxs[template_idx])
        search_bbox = np.array(self.bboxs[search_idx])
       
        return template_img, template_bbox, search_img, search_bbox

if __name__ =='__main__':
    import kornia

    dataset = AttackDataset(step=1)
    dataloader = DataLoader(dataset, batch_size=30, shuffle=True, num_workers=8)

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