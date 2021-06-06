import os

import cv2
# import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os.path as osp

from avcv import utils as au
# try:
#     import mmcv
# except:
#     mmcv = None


def get_min_rect(c, resize_ratio):
    """input a contour and return the min box of it"""
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box = (box / resize_ratio).astype(np.int32)
    return box


def get_skeleton(img, line_size):
    """ Get skeleton mask of a binary image 
        Arguments:
            img: input image 2d
        Returns:
            binnary mask skeleton
    """

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    size = np.size(img)
    done = False
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    kernel = np.ones(shape=[line_size, line_size])
    _ = cv2.dilate(skeleton, kernel, iterations=1)
    return _


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
    import matplotlib.pyplot as plt
    if mxn is None:
        # n = max(max_w // max([img.shape[1] for img in images]), 1)
        n = int(np.sqrt(len(images)))
        n = min(n, len(images))
        m = len(images) // n
        m = max(1, m)
        mxn = (m, n)
        print(mxn)

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
        print('Save fig:', out_file)
        plt.close()


def show(inp, size=10, dpi=300, cmap='gray', out_file=None):
    """
        Input: either a path or image
    """
    # inp = mmcv.imread(inp)
    import matplotlib.pyplot as plt
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


def find_contours(thresh):
    """
        Get contour of a binary image
            Arguments:
                thresh: binary image
            Returns:
                Contours: a list of contour
                Hierarchy:

    """
    try:
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
        return contours, hierarchy[0]
    except:
        return None, None


def sort_contours(contours, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    (contours, bounding_boxes) = zip(*sorted(
        zip(contours, bounding_boxes), key=lambda b: b[1][i], reverse=reverse))

    return contours, bounding_boxes


def put_text(image, pos, text, color=(255, 255, 255)):
    return cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                       color, 2)


def torch_tensor_to_image(tensor, cdim=1):
    permute = [0, 2, 3, 1] if len(tensor.shape) == 4 else [1, 2, 0]
    np_array = tensor.detach().permute(permute).cpu().numpy()
    np_array = (np_array - np_array.min()) / (np_array.max() - np_array.min())
    np_array = (np_array) * 255
    return np_array.astype('uint8')


# def visualize_torch_tensor(input_tensor, normalize=True, output_path=None, input_type='onehot'):
#     import torch

#     if isinstance(input_tensor, torch.Tensor):
#         input_tensor = input_tensor.detach().cpu().numpy()
#         if len(np.unique(input_tensor)) == 2:
#             normalize = False
#     # if none save in cache
#     if output_path is None:
#         os.makedirs('cache', exist_ok=True)
#         output_path = 'cache/temp.jpg'
#     if normalize:
#         input_tensor = (input_tensor-input_tensor.min())/(input_tensor.max()-input_tensor.min())
#         input_tensor *= 255

#     if input_type == 'onehot':
#         c,h,w = input_tensor.shape
#         color = np.random.choice(256, (c, 3))
#         color_tensor = input_tensor[:,None,:,:]*color[:,:,None,None]
#         out_tensor = np.zeros([h, w, 3], color_tensor.dtype)
#         for i in range(c):
#             out_tensor += np.transpose(color_tensor[i], [1,2,0])
#         out_tensor = out_tensor *255 /out_tensor.max()
#     elif input_type=='image':
#         out_tensor = np.transpose(input_tensor, [1,2,0])
#     else:
#         raise NotImplemented
#     cv2.imwrite(output_path, out_tensor)
#     print('out-> :', output_path)


def images_to_video(images, out_path, fps=30, sort=True, max_num_frame=10e12, with_text=False):
    fps = int(fps)
    max_num_frame = int(max_num_frame)
    max_num_frame = min(len(images), max_num_frame)
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

    # img_array = au.multi_thread(f, images, verbose=True)
    if sort and isinstance(images[0], str):
        images = list(sorted(images, key=get_num))
    # imgs = au.multi_thread(f, images[:max_num_frame], verbose=True)  #[mmcv.imread(path) for path in tqdm(images)]
    # [mmcv.imread(path) for path in tqdm(images)]
    # imgs = au.multi_process(f, images[:max_num_frame], 16)

    h, w = mmcv.imread(images[0]).shape[:2]
    size = (w, h)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(images)):
        img = f(images[i])
        im = cv2.resize(img, size)
        out.write(im)
    out.release()
    print(out_path)

def video_to_images(input_video, output_dir, skip=1):
    import cv2 
    import os 
    from imutils.video import count_frames
    skip = int(skip)
    # Read the video from specified path 
    cam = cv2.VideoCapture(input_video) 
    total_frames = count_frames(input_video)
    os.makedirs(output_dir, exist_ok=True) 
    # frame 
    currentframe = 0

    
    # while(True):
    for current_frame in tqdm(range(0, total_frames, skip)): 
        # reading from frame 
        ret,frame = cam.read() 
        
        if ret: 
            # if video is still left continue creating images 
            name =  os.path.join(output_dir,f'{current_frame:05d}' + '.jpg') 
            if currentframe % skip == 0:
                cv2.imwrite(name, frame) 
        else: 
            break
    # Release all space and windows once done 
    cam.release() 
    cv2.destroyAllWindows() 

def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    # import ipdb; ipdb.set_trace()
    return color[...,0] + 256 * color[...,1] + 256 * 256 * color[...,2]

def gt_to_color_mask(gt, mask=None,palette=None):
    # import ipdb; ipdb.set_trace()
    if len(gt.shape) == 3:
        # from panopticapi.utils import rgb2id
        gt = rgb2id(gt)
    class_ids = np.unique(gt)
    h, w = gt.shape[:2]
    if palette is None:
        palette = dict()
        for cls_id in class_ids:
            np.random.seed(cls_id+1)
            if cls_id == 0:
                color = np.array([0,0,0])
            else:
                color = np.random.choice(256, 3)
            palette[cls_id] = color
    if mask is None:
        mask = np.zeros([h,w,3], 'uint8')
    for cls_id in class_ids:
        ids = gt == cls_id
        if cls_id < len(palette):
            color = palette[cls_id]
            mask[ids] = color
    return mask

def vis_ids_to_segmask(input_dir, output_dir, palete=None):
    import mmcv
    paths = au.get_paths(input_dir, 'png')
    def fun(path_in_out):    
        path, output_dir = path_in_out
        gt = mmcv.imread(path, cv2.IMREAD_UNCHANGED)
        mask = gt_to_color_mask(gt, palete)
        name = osp.basename(path)
        out_path = osp.join(output_dir, name)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mmcv.imwrite(mask, out_path)

    au.multi_thread(fun, [[path, output_dir] for path in paths], max_workers=1)


def vis_segmask_to_ids(input_dir, output_dir, id_to_color):
    paths = au.get_paths(input_dir, 'png')
    def fun(path_in_out):    
        path, output_dir = path_in_out
        gt = mmcv.imread(path, cv2.IMREAD_UNCHANGED)
        h, w = gt.shape[:2]
        mask = np.zeros((h, w), 'uint8')
        for id, color in id_to_color.items():
            ids = (gt == color).mean(-1) == 1
            mask[ids] = id
        name = osp.basename(path)
        out_path = osp.join(output_dir, name)
        mmcv.imwrite(mask, out_path)

    au.multi_thread(fun, [[path, output_dir] for path in paths], verbose=True, max_workers=4)
    
def vis_combine(dir_a, dir_b, combine_dir, split_txt=None, alpha=None):
    import mmcv
    import itertools
    from glob import glob

    paths_a = glob(os.path.join(dir_a, '**', '*.jpg'), recursive=True)+glob(os.path.join(dir_a, '**', '*.png'), recursive=True)+glob(os.path.join(dir_a, '**', '*.jpeg'), recursive=True)
    

    paths_b = []
    for path_a in paths_a:
        print(path_a)
        if split_txt is not None:
            name = path_a.split(split_txt)[-1]
        else:
            name = osp.basename(path_a)
        path_b = osp.join(dir_b, name)
        if not osp.exists(path_b):
            path_b = path_b.replace('.png', '.jpg')

        assert osp.exists(path_b), path_b
        paths_b.append(path_b)
        
    for pa, pb in zip(paths_a, paths_b):
        if osp.basename(pa).split('.')[0]!=osp.basename(pb).split('.')[0]: continue
        ima = mmcv.imread(pa)
        h, w= ima.shape[:2]
        imb = mmcv.imread(pb)
        imb = mmcv.imresize(imb, (w,h))
        if alpha is None:
            imab = np.concatenate([ima, imb], 1)
        else:
            imab = (0.5*ima + 0.5*imb).astype('uint8')
        # name = osp.basename(pa)
        name = pa.split(split_txt)[-1]
        out_path = osp.join(combine_dir, name)
        mmcv.imwrite(imab, out_path)


def resize_mask(mask, ratio=None, size=None):
    """Resize
        size: (w, h)
    """
    import torch.nn.functional as F
    def resize(input,
            size=None,
            scale_factor=None,
            mode='nearest',
            align_corners=None,
            warning=True):
        return F.interpolate(input, size, scale_factor, mode, align_corners)

    import torch
    mask = mmcv.imread(mask, cv2.IMREAD_UNCHANGED)
    assert len(mask.shape) == 2
    ori_h, ori_w = mask.shape[:2]
    if ratio is not None:
        h = int(ori_h*ratio)
        w = int(ori_w*ratio) 
    elif size is not None:
        w, h = size
    else:
        raise NotImplemented
    new_size = (h,w)
    mask = torch.from_numpy(mask)
    one_hot = torch.nn.functional.one_hot(mask.long(), mask.max()+1)[None].permute([0,3,1,2])
    one_hot_resize = resize(one_hot.float(), new_size)[0]
    out = one_hot_resize.argmax(0).cpu().numpy()
    return out.astype('uint8')


