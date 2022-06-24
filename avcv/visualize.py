# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/00_visualize.ipynb (unless otherwise specified).

__all__ = ['plot_images', 'imshow', 'show', 'tensor2imgs', 'bbox_visualize']

# Cell
import matplotlib.pyplot as plt
import numpy as np
import cv2
from  loguru import logger
def plot_images(images,
                labels=None,
                cls_true=None,
                cls_pred=None,
                space=(0.3, 0.3),
                mxn=None,
                size=(5, 5),
                dpi=300,
                max_w=1500,
                out_file=None,
                cmap='binary'):

    if mxn is None:
        # n = max(max_w // max([img.shape[1] for img in images]), 1)
        n = int(np.sqrt(len(images)))
        n = min(n, len(images))
        m = len(images) // n
        m = max(1, m)
        mxn = (m, n)
        logger.info(f"Grid size: {mxn}")

    fig, axes = plt.subplots(*mxn)
    fig.subplots_adjust(hspace=space[0], wspace=space[1])
    fig.figsize = size
    fig.dpi = dpi
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], cmap=cmap)
            if labels is not None:
                xlabel = labels[i]
            elif cls_pred is None and cls_true is not None:
                xlabel = "True: {0}".format(cls_true[i])
            elif cls_pred is None and cls_true is not None:
                xlabel = "True: {0}, Pred: {1}".format(cls_true[i],
                                                       cls_pred[i])
            else:
                xlabel = None
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            ax.set_xticks([])
            ax.set_yticks([])
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)
        logger.info('Save fig:', out_file)
        plt.close()

def imshow(inp,  dpi=100, size=10, cmap='gray', out_file=None):
    """
        Input: either a path or image
    """
    # inp = mmcv.imread(inp)
    if len(inp.shape) == 4:
        inp = inp[0]
    inp = np.squeeze(inp)
    if type(inp) is str:
        assert os.path.exists(inp)
        inp = cv2.imread(inp)
    if size is None:
        size = max(5, inp.shape[1] // 65)
    plt.figure(figsize=(size, size), dpi=dpi)
    plt.imshow(inp, cmap=cmap)
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)
    plt.close()
show = imshow

# Cell
import torch
import mmcv
import numpy as np

def tensor2imgs(tensor, mode='bhwc',
                    mean=(123.675, 116.28, 103.53), std= (58.395, 57.120000000000005, 57.375), **kwargs):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    tensor = tensor.cpu()
    if mode == 'bhwc':
        tensor = tensor.permute([0,3,1,2])
        return tensor2imgs(tensor, mode='bchw',std=std, mean=mean)
    if mode == 'hwc':
        tensor = tensor[None].permute([0,3,1,2])
        return tensor2imgs(tensor, mode='bchw', std=std, mean=mean)[0]
    if mode == 'chw':
        tensor = tensor[None]
        return tensor2imgs(tensor, mode='bchw', std=std, mean=mean)[0]
    return mmcv.tensor2imgs(tensor, mean=mean, std=std, **kwargs)

# Cell

#export
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)

def bbox_visualize(img, boxes, scores, cls_ids, conf=0.5, class_names=None, texts=None, box_color=None):
    img = img.copy()
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        if box_color is None:

            color_id = cls_id % len(_COLORS)
            _color = _COLORS[color_id]
        else:
            _color = np.array(box_color)/255.

        color = (_color * 255).astype(np.uint8).tolist()

        txt_color = (0, 0, 0) if np.mean(_color) > 0.5 else (255, 255, 255)
        txt_bk_color = (_color * 255 * 0.7).astype(np.uint8).tolist()



        if texts is None:
            text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        else:
            text = texts[i]

        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)


        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img