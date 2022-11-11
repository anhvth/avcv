# import __main__ as main
# def is_interactive():
#     return not hasattr(main, '__file__')
import inspect
import json
import os
import os.path as osp
import pickle
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial, wraps
from glob import glob
from multiprocessing import Pool

import xxhash
from fastcore.all import *
from fastcore.parallel import threaded
from fastcore.script import *
from fastcore.script import Param, call_parse
from loguru import logger
from PIL import Image
# from pycocotools.coco import COCO
from tqdm import tqdm
# from .lazy_modules import LazyModule
from lazy_module.core import LazyModule

mmcv = LazyModule('mmcv')
np = LazyModule('numpy')
cv2 = LazyModule('cv2')
matplotlib = LazyModule('matplotlib')
plt = LazyModule('plt', 'import matplotlib.pyplot as plt')
coco = LazyModule('coco', 'from pycocotools import coco')
ipdb = LazyModule('ipdb')
pd = LazyModule('pandas')
#import ipdb, cv2, matplotlib.pyplot as plt, mmcv, numpy as np, matplotlib
