# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/04_debug.ipynb (unless otherwise specified).

__all__ = ['make_mini_coco', 'dpython']

# Cell
import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import mmcv
from fastcore.script import *
import shutil
from loguru import logger
import mmcv


@call_parse
def make_mini_coco(json_path: Param(),
                   image_prefix: Param(),
                   out_dir: Param(),
                   num_samples: Param("Num of sample",type=int) = 1000):
    """
        Helper function for creating a mini-dataset ensembles it's father
    """
    from pycocotools.coco import COCO
    new_img_prefix = osp.join(out_dir, "images")

    out_json = os.path.join(out_dir, "annotations", "mini_json.json")
    if not osp.exists(out_json):
        logger.info(f"Making mini dataset out_dir-> {out_dir}, num images:{num_samples}")
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "annotations"), exist_ok=True)
        coco = COCO(json_path)
        # imgs = coco.imgs
        img_ids = list(coco.imgs.keys())
        np.random.seed(0)
        selected_img_ids = np.random.choice(img_ids, num_samples, replace=False)
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
    logger.info(f"{out_json}, {new_img_prefix}")
    return out_json, new_img_prefix

# Cell
@call_parse
def dpython(cmd: Param(type=str)):
    for _ in range(3):
        cmd = cmd.replace('  ', '')
    i_split = cmd.index(".py")+4
    file = cmd[:i_split].strip().split(' ')[1]

    args = cmd[i_split:].split(' ')
    cfg_name = os.environ.get("DNAME", "Latest-Generated")
    cfg = {
        "name": f"Python: {cfg_name}",
        "type": "python",
        "request": "launch",
        "program": f"{file}",
        "console": "integratedTerminal",
        "args": args,
    }
    # pp(cfg)
    mmcv.mkdir_or_exist(".vscode")
    try:
        lauch = mmcv.load(".vscode/launch.json")
    except Exception as e:
        lauch = {
            "version": "0.2.0",
            "configurations": [

            ]
        }
        logger.warning(e)

    replace = False
    for i, _cfg in enumerate(lauch['configurations']):
        if _cfg["name"] == cfg["name"]:
            lauch["configurations"][i] = cfg
            replace = True
    if not replace:
        lauch["configurations"] += [cfg]
        mmcv.dump(lauch, '.vscode/launch.json')