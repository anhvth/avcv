# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03_utils.ipynb (unless otherwise specified).

__all__ = ['get_name', 'find_contours', 'download_file_from_google_drive', 'mkdir', 'put_text', 'images_to_video',
           'get_paths', 'video_to_images', 'TimeLoger', 'identify', 'memoize']

# Cell
import os
from glob import glob
import os
import cv2
import os.path as osp
from tqdm import tqdm
from fastcore.script import call_parse, Param


def get_name(path):
    path = osp.basename(path).split('.')[:-1]
    return '.'.join(path)


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


@call_parse
def download_file_from_google_drive(id_or_link: Param("Link or file id"), destination: Param("Path to the save file")):
    if "https" in id_or_link:
        x = id_or_link
        id = x.split("/")[x.split("/").index("d")+1]
    else:
        id = id_or_link
    print("Download from id:", id)
    import requests

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)
    print("Done ->", destination)
    return osp.abspath(destination)


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def put_text(image, pos, text, color=(255, 255, 255)):
    return cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                       color, 2)


@call_parse
def images_to_video(
        images: Param("Path to the images folder or list of images"),
        out_path: Param("Output output video path", str),
        fps: Param("Frame per second", int) = 30,
        sort: Param("Sort images", bool) = True,
        max_num_frame: Param("Max num of frame", int) = 10e12,
        with_text: Param("Add additional index to image when writing vidoe", bool) = False):
    fps = int(fps)

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
                # img = cv2.putText(img, name, )
                img = put_text(img, (20, 20), name)
        else:
            img = img_or_path
        return img

    if sort and isinstance(images[0], str):
        images = list(sorted(images, key=get_num))

    max_num_frame = int(max_num_frame)
    max_num_frame = min(len(images), max_num_frame)

    h, w = cv2.imread(images[0]).shape[:2]
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

# Cell
@call_parse
def video_to_images(input_video:Param("", str), output_dir:Param("", str), skip:Param("", int)=1):
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

# Cell
from mmcv import Timer
import pandas as pd
import numpy as np
class TimeLoger:
    def __init__(self):
        self.timer = Timer()
        self.time_dict = dict()

    def start(self):
        self.timer.start()

    def update(self, name):
        # assert not name in self.time_dict
        duration = self.timer.since_last_check()
        if name in self.time_dict:
            self.time_dict[name].append(duration)
        else:
            self.time_dict[name] = [duration]

    def __str__(self):
        total_time = np.sum([np.sum(v) for v in self.time_dict.values()])
        s = f"------------------Time Loger Summary : Total {total_time:0.2f} ---------------------:\n"
        for k, v in self.time_dict.items():
            average = np.mean(v)
            times = len(v)
            percent = np.sum(v)*100/total_time
            # print()
            s += f'\t\t{k}:  \t\t{percent:0.2f}% ({average:0.4f}s) | Times: {times} \n'
        # print('-----------------------------------------------------------')
        return s

# Cell
import xxhash
import pickle

def identify(x):
    '''Return an hex digest of the input'''
    return xxhash.xxh64(pickle.dumps(x), seed=0).hexdigest()


def memoize(func):
    import os
    import pickle
    from functools import wraps

    import xxhash

    '''Cache result of function call on disk
    Support multiple positional and keyword arguments'''
    @wraps(func)
    def memoized_func(*args, **kwargs):
        cache_dir = 'cache'
        try:
            import inspect
            func_id = identify((inspect.getsource(func), args, kwargs))
            cache_path = os.path.join(cache_dir, func.__name__+'_'+func_id)

            if (os.path.exists(cache_path) and
                    not func.__name__ in os.environ and
                    not 'BUST_CACHE' in os.environ):
                result = pickle.load(open(cache_path, 'rb'))
            else:
                result = func(*args, **kwargs)
                os.makedirs(cache_dir, exist_ok=True)
                pickle.dump(result, open(cache_path, 'wb'))
            return result
        except (KeyError, AttributeError, TypeError):
            return func(*args, **kwargs)
    return memoized_func