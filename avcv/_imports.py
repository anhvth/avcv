# import __main__ as main
# def is_interactive():
#     return not hasattr(main, '__file__')
import inspect
import json
import os
import os.path as osp
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
from pycocotools.coco import COCO
from tqdm import tqdm
import pickle
from .lazy_modules import LazyObject

mmcv = LazyObject('mmcv')
np = LazyObject('numpy')
ipdb = LazyObject('ipdb')
cv2 = LazyObject('cv2')
plt = LazyObject('plt', 'import matplotlib.pyplot as plt')

# import ipdb, cv2, matplotlib.pyplot as plt, mmcv, numpy as np, matplotlib
