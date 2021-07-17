# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03_utils.ipynb (unless otherwise specified).

__all__ = ['images_to_video', 'get_paths']

# Cell
import os
from glob import glob
import os
import mmcv
import cv2
from tqdm import tqdm
from fastcore.script import call_parse, Param

@call_parse
def images_to_video(
    images:Param("Path to the images folder or list of images"),
    out_path:Param("Output output video path", str),
    fps:Param("Frame per second", int)=30,
    sort:Param("Sort images", bool)=True,
    max_num_frame:Param("Max num of frame", int)=10e12,
    with_text:Param("Add additional index to image when writing vidoe", bool)=False):

    sort = bool(sort)
    if isinstance(images, str) and os.path.isdir(images):
        from glob import glob
        images = glob(os.path.join(images, "*.jpg")) + \
            glob(os.path.join(images, "*.png"))

    imgs = []

    def get_num(s):
        try:
            s = os.path.basename(s)
            num = int(''.join([c for c in s if c.isdigit()]))
        except:
            num = s
        return num
    global f

    def f(img_or_path):
        if isinstance(img_or_path, str):
            name = os.path.basename(img_or_path)
            img = cv2.imread(img_or_path)
            assert img is not None, img_or_path
            if with_text:
                img = put_text(img, (20, 20), name)
        else:
            img = img_or_path
        return img
    if sort and isinstance(images[0], str):
        images = list(sorted(images, key=get_num))

    max_num_frame = int(max_num_frame)
    max_num_frame = min(len(images), max_num_frame)

    h, w = mmcv.imread(images[0]).shape[:2]
    size = (w, h)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    images = images[:max_num_frame]
    pbar = tqdm(range(len(images)))
    for i in pbar:
        img = f(images[i])
        im = cv2.resize(img, size)
        out.write(im)

    out.release()
    print(out_path)

def get_paths(directory, input_type='png', sort=True):
    """
        Get a list of input_type paths
        params args:
        return: a list of paths
    """
    paths = glob(os.path.join(directory, '*.{}'.format(input_type)))
    if sort:
        paths = list(sorted(paths))
    print('Found and sorted {} files {}'.format(len(paths), input_type))
    return paths

