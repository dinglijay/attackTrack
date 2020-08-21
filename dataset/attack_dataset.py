import cv2
import glob
import json
from os.path import join, exists
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


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
    ''' NOTE
    DataLoader cannot batching images with different shape.
    To ensure this ugly code works, keep (batchsize % n_frames)==0 and 
    shuffle=False when setup DataLoader
    ''' 
    def __init__(self, root_dir='data/lasot/person/person-14', frame_sample='random', n_frames=20, test=False):
        # load annotation file
        if 'OTB100' in root_dir: json_fname = 'OTB100.json'
        if 'VOT2019' in root_dir: json_fname = 'VOT2019.json'
        if 'lasot' in root_dir: json_fname = 'anno.json'
        if 'own' in root_dir: json_fname = 'anno.json'
        with open(join(root_dir, json_fname), 'r') as f:
            annos = json.load(f)

        # process video resolutions
        img_shapes = []
        for anno in annos.values():
            img_0 = anno['img_names'][0]
            img_0 = img_0 if 'data' in img_0 else join(root_dir, img_0)
            img_shapes.append( cv2.imread(img_0).shape[0:2] )
        unique, counts = np.unique(np.array(img_shapes), return_counts=True, axis=0)
        video_res = unique[counts.argsort()[::-1]]
            
        if test: n_frames = None

        # image name list and gt_bbox list
        self.img_names = list()
        self.bboxs = list()
        video_lens = list()
        for anno in annos.values():
            # # video resolution filter 
            # img_0 = anno['img_names'][0]
            # img_0 = img_size if 'data' in img_0 else join(root_dir, img_0)
            # img_size = cv2.imread(img_0).shape[0:2]
            # if 'OTB' in root_dir and np.all(img_size != video_res[1]): # [480, 640] for OTB
            #     continue

            if not n_frames:
                self.img_names.extend(anno['img_names'])
                self.bboxs.extend(anno['gt_rect'])
                video_lens.append(len(anno['img_names']))
            else:
                video_len = len(anno['img_names']) if frame_sample == 'random' else n_frames
                indices = np.random.choice(video_len, n_frames, replace=False)
                indices.sort()
                anno_selected = [anno['img_names'][idx] for idx in indices]
                self.img_names.extend(anno_selected)
                self.bboxs.extend([anno['gt_rect'][idx] for idx in indices])
                video_lens.append(len(anno_selected))
        self.video_seg = np.add.accumulate(video_lens)
        self.video_seg = np.insert(self.video_seg, 0, 0) 
        assert len(self.bboxs) == len(self.img_names)

        # load images to ram if they are not too much
        self.imgs = None
        if len(self.img_names) <= 1000:
            if 'data' in self.img_names[0]:
                self.imgs = [np.transpose(cv2.imread(im_name).astype(np.float32), (2, 0, 1)) \
                            for im_name in self.img_names]
            else:
                self.imgs = [np.transpose(cv2.imread(join(root_dir, im_name)).astype(np.float32), (2, 0, 1)) \
                            for im_name in self.img_names]

        self.n_frames = n_frames
        self.test = test
        self.root_dir = root_dir
    
    def gen_ind_combinations(self):
        n_imgs = len(self.img_names)
        list(permutations(n_imgs, 5))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        video_idx = np.searchsorted(self.video_seg, idx, 'right')
        if self.test:
            template_idx = self.video_seg[video_idx-1]
        else:
            # numpy’s RNG is not forkable.
            template_idx = torch.randint(self.video_seg[video_idx-1], self.video_seg[video_idx], (1,)).item()
            
        search_idx = idx
        # print(self.img_names[template_idx], self.img_names[search_idx])
        
        if self.imgs:
            template_img = self.imgs[template_idx]
            search_img = self.imgs[search_idx]
        else:
            template_img_name = self.img_names[template_idx]
            search_img_name = self.img_names[search_idx]
            if 'data' not in template_img_name:
                template_img_name = join(self.root_dir, template_img_name)
            if 'data' not in search_img_name:
                search_img_name = join(self.root_dir, search_img_name)                
            template_img = np.transpose(cv2.imread(template_img_name).astype(np.float32), (2, 0, 1))
            search_img = np.transpose(cv2.imread(search_img_name).astype(np.float32), (2, 0, 1))
        template_bbox = np.array(self.bboxs[template_idx])
        search_bbox = np.array(self.bboxs[search_idx])
       
        return template_img, template_bbox, search_img, search_bbox


if __name__ =='__main__':
    import kornia

    dataset = AttackDataset(n_frames=10)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=5)

    cv2.namedWindow("template", cv2.WND_PROP_FULLSCREEN)
    cv2.namedWindow("search", cv2.WND_PROP_FULLSCREEN)

    for i in range(20):
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