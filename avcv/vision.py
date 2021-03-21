import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os.path as osp

from avcv import utils as au

try:
    import mmcv
except:
    mmcv = None


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


def convert_mask_to_cell(line_mask):
    """ Convert a mask of lines to cells.
        Arguments:
            line_mask: mask of lines
        Returns:
            a list of cells        
    """
    def is_rect(contour):
        """ Check if a contour is rectangle.
            Arguments:
                contour: contours.
            Returns:
                Boolean value if contour is a rectangle.
        """
        _, _, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 0 and w * h > 0 and area / w * h > 0.6:
            return True
        else:
            return False

    contours, hierarchy = find_contours(line_mask)
    out_cnts = {}
    for ci, (cnt, h) in enumerate(zip(contours, hierarchy)):
        if is_rect(cnt):
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            cnt = np.int0(box)
            if h[-1] == -1:
                out_cnts['table_{}'.format(ci)] = cnt
            else:
                out_cnts['cell_{}_table_{}'.format(ci, h[-1])] = cnt
    return out_cnts


def imread(path, to_gray=False, scale=(0, 255)):
    """ Read image given a path
        Arguments:
            path: path to image
            to_gray: convert image to gray
            scale: if scale (0, 255)
        Return: 
            output image
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if scale[1] != 255:
        min_val, max_val = scale
        # after: 0-1
        img = img / 255
        # scale 0 -> (max_val-min_val)
        img = img * (max_val - min_val)
        # scale: min_val -> max_val
        img = img + min_val

    return img


def imwrite(path, img, is_rgb=True):
    if is_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(path, img)


def resize_by_factor(image, factor):
    """ Resize image by a factor
        Arguments:
            image: input image
            factor: the factor by which the image being resized
        Returns:
            output image
    """
    if type(factor) == tuple:
        fx, fy = factor
    elif type(factor) == float:
        fx = fy = factor
    else:
        raise Exception("type of f must be tuple or float")

    return cv2.resize(image, (0, 0), fx=fx, fy=fy)


def resize_by_size(image, size):
    """ Resize image by given size
        Arguments:
            image: input image
            size: the size at which the image being reized

        Returns:
            output image
    """
    return cv2.resize(image, size)


def resize_to_receptive_field(image, receptive_field=256):
    """Resize to the factor of the wanted receptive field.
    Example: Image of size 1100-800 -> 1024-768
    Arguments:
        image: input iamge
        receptive_field:
    Returns:
        resieed image
    """
    new_h, new_w = np.ceil(np.array(image.shape[:2]) / receptive_field).astype(
        np.int32) * receptive_field
    image = cv2.resize(image, (new_w, new_h))
    return image


def batch_ratio(preds, targets):
    from fuzzywuzzy import fuzz
    rt = []
    for p, t in zip(preds, targets):
        r = fuzz.ratio(p, t)
        rt.append(r)
    return np.mean(rt)


def plot_images(images,
                labels=None,
                cls_true=None,
                cls_pred=None,
                space=(0.3, 0.3),
                mxn=None,
                size=(5, 5),
                dpi=300,
                max_w=1500,
                output_path=None,
                cmap='binary'):

    if mxn is None:
        n = max(max_w // max([img.shape[1] for img in images]), 1)
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
    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)
        plt.close()


def run_data_init(initers, shuffle):
    for initer in initers:
        data = initer['index']
        if shuffle:
            data = shuffle_by_batch(data, batch_size)
        sess.run(initer['initer'], {initer['x']: data})


def show(inp, size=10, dpi=300, cmap='gray'):
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
    plt.show()


def find_contours(thresh):
    """
        Get contour of a binary image
            Arguments:
                thresh: binary image
            Returns:
                Contours: a list of contour
                Hierarchy:

    """
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy[0]


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


def put_text(image, pos, text):
    return cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                       (255, 255, 255), 2)


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


def images_to_video(images, out_path, fps=30, sort=True, max_num_frame=1000):
    fps = int(fps)
    max_num_frame = int(max_num_frame)
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
            img = put_text(img, (20, 20), name)
        else:
            img = img_or_path
        return img

    # img_array = au.multi_thread(f, images, verbose=True)
    if sort and isinstance(images[0], str):
        images = list(sorted(images, key=get_num))
    # imgs = au.multi_thread(f, images[:max_num_frame], verbose=True)  #[mmcv.imread(path) for path in tqdm(images)]
    # [mmcv.imread(path) for path in tqdm(images)]
    imgs = au.multi_process(f, images[:max_num_frame], 16)

    h, w = imgs[0].shape[:2]
    size = (w, h)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(imgs)):
        im = cv2.resize(imgs[i], size)
        out.write(im)
    out.release()
    print(out_path)

def video_to_images(input_video, output_dir, skip=1):
    import cv2 
    import os 
    skip = int(skip)
    # Read the video from specified path 
    cam = cv2.VideoCapture(input_video) 
    os.makedirs(output_dir, exist_ok=True) 
    # frame 
    currentframe = 0
    while(True): 
        # reading from frame 
        ret,frame = cam.read() 
        
        if ret: 
            # if video is still left continue creating images 
            name =  os.path.join(output_dir,str(currentframe) + '.jpg') 
            currentframe += 1
            if currentframe % skip == 0:
                cv2.imwrite(name, frame) 
        else: 
            break
    # Release all space and windows once done 
    cam.release() 
    cv2.destroyAllWindows() 

def gt_to_color_mask(gt, palette=None):
    class_ids = np.unique(gt)
    h, w = gt.shape[:2]
    if palette is None:
        palette = dict()
        for cls_id in class_ids:
            np.random.seed(cls_id)
            color = np.random.choice(255, 3)
            palette[cls_id] = color

    mask = np.zeros([h,w,3], 'uint8')
    for cls_id in class_ids:
        ids = gt == cls_id
        mask[ids] = palette[cls_id]

    return mask

def visualize_seg_gt(input_dir, output_dir):
    paths = au.get_paths(input_dir, 'png')
    def fun(path_in_out):    
        path, output_dir = path_in_out
        gt = mmcv.imread(path, cv2.IMREAD_UNCHANGED)
        mask = gt_to_color_mask(gt)
        name = osp.basename(path)
        out_path = osp.join(output_dir, name)
        
        mmcv.imwrite(mask, out_path)

    au.multi_thread(fun, [[path, output_dir] for path in paths])