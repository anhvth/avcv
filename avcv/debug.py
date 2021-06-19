import os
import os.path as osp
from unicodedata import category

import matplotlib.pyplot as plt

from avcv.utils import json, mkdir, read_json, shutil, tqdm, identify
from avcv.vision import plot_images, show, mmcv
import numpy as np

def debug_make_mini_dataset(json_path, image_prefix, out_dir, n=1000, file_name=None):
    new_img_prefix = osp.join(out_dir, "images")
    from pycocotools.coco import COCO
    mkdir(new_img_prefix)
    out_json = os.path.join(out_dir, "annotations", "mini_json.json")
    if not osp.exists(out_json):
        print("Making mini dataset", out_dir, "num images:", n)
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "annotations"), exist_ok=True)
        coco = COCO(json_path)
        # imgs = coco.imgs
        img_ids = list(coco.imgs.keys())
        np.random.seed(0)
        selected_img_ids = np.random.choice(img_ids, n, replace=False)
        imgs = coco.loadImgs(selected_img_ids)
        selected_ann_ids = coco.getAnnIds(selected_img_ids)
        anns = coco.loadAnns(selected_ann_ids)
        out_dict = dict(
            images = imgs,
            annotations=anns,
            categories=coco.dataset['categories'],
            
        )
        for img in imgs:
            path = osp.join(image_prefix, img['file_name'])
            new_path = osp.join(new_img_prefix, img['file_name'])
            shutil.copy(path, new_path)

        with open(out_json, "w") as f:
            json.dump(out_dict, f)
        print(out_json)
    return out_json, new_img_prefix


# def make_pickle_dataset(dataset, n=100, out_dir='./cache/debug-data'):
#     from torch.utils.data import Dataset
#     mkdir(out_dir)

#     class PickleDataset(Dataset):
#         def __init__(self, dataset, n):
#             self.dataset = dataset
#             self.CLASSES = dataset.CLASSES
#             self.flag = np.zeros(n)
#             self.cache = dict()
            
#             for i in tqdm(range(n)):
#                 item = self.dataset[i]
#                 id = identify(item)
#                 out_path = osp.join(out_dir, f"{id}.pkl")
#                 mmcv.dump(item, out_path)
#                 self.cache[i] = out_path
#                 self.flag[i] = self.dataset.flag[i]

#         def __len__(self):
#             return len(self.cache)

#         def __getitem__(self, index):
#             return mmcv.load(self.cache[index])
#     return PickleDataset(dataset, n)


def vsl(image_or_tensor, order="bhwc", normalize=True, out_file='cache/vsl.jpg'):
    if 'Tensor' in str(type(image_or_tensor)):
        if len(image_or_tensor.shape) == 4 and (image_or_tensor.shape[1] == 1 or image_or_tensor.shape[1] == 3):
            image_or_tensor = image_or_tensor.permute([0, 2, 3, 1])
        if len(image_or_tensor.shape) == 3 and (image_or_tensor.shape[0] == 1 or image_or_tensor.shape[1] == 3):
            image_or_tensor = image_or_tensor.permute([1, 2, 0])
            image_or_tensor = image_or_tensor[None]

        images = image_or_tensor.detach().cpu().numpy()
    elif 'ndarray' in str(type(image_or_tensor)):
        if len(image_or_tensor.shape) == 3:
            if (image_or_tensor.shape[0] == 1 or image_or_tensor.shape[1] == 3):
                raise NotImplemented
            else:
                image_or_tensor = image_or_tensor[None]

        images = image_or_tensor
    elif isinstance(image_or_tensor, list):
        assert isinstance(image_or_tensor[0], np.ndarray)
        images = image_or_tensor
    if normalize:
        outs = []
        for i, image in enumerate(images):
            image = image-image.min()
            image = image / image.max()
            image *= 255
            image = image.astype('uint8')
            outs += [image]
            images = outs

    mkdir('cache')
    if len(images) == 1:
        show(images[0], dpi=150)
        plt.savefig(out_file)
    else:
        plot_images(images, out_file=out_file)
    print(out_file)
