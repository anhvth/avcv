# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/04_debug.ipynb (unless otherwise specified).

__all__ = ['make_mini_coco']

# Cell
import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import mmcv
from fastcore.script import *
from pycocotools.coco import COCO
import shutil


@call_parse
def make_mini_coco(json_path: Param(),
                   image_prefix: Param(),
                   out_dir: Param(),
                   num_samples: Param("Num of sample",type=int) = 1000):
    """
        Helper function for creating a mini-dataset ensembles it's father
    """
    new_img_prefix = osp.join(out_dir, "images")

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
        for i, ann in enumerate(anns):
            ann['iscrowd'] = False
            anns[i] = ann
        out_dict = dict(
            images=imgs,
            annotations=anns,
            categories=coco.dataset['categories'],

        )
        for img in imgs:
            path = osp.join(image_prefix, img['file_name'])
            new_path = osp.join(new_img_prefix, img['file_name'])
            shutil.copy(path, new_path)

        mmcv.dump(out_dict, out_json)
    print(out_json, new_img_prefix)
    return out_json, new_img_prefix
