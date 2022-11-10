
# import __main__ as main
# def is_interactive():
#     return not hasattr(main, '__file__')
from .lazy_modules import LazyObject
import os
import os.path as osp
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial, wraps
from glob import glob
from multiprocessing import Pool

from fastcore.all import *
from fastcore.parallel import threaded
from fastcore.script import *
from fastcore.script import Param, call_parse
from loguru import logger

import inspect
import json
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

mmcv = LazyObject('mmcv')
np = LazyObject('numpy')
ipdb = LazyObject('ipdb')
cv2 = LazyObject('cv2')