import inspect
import json
import os
import os.path as osp
import pickle
import time
import os
import pickle
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
from tqdm import tqdm
from lazy_module.core import LazyModule
import copy
from speedy import *
from loguru import logger
from fastcore.all import *
from fastcore.parallel import threaded
from fastcore.script import Param, call_parse
from lazy_module.core import LazyModule
from PIL import Image

# Use lazy imports for performance optimization
mmcv = LazyModule('mmcv')
np = LazyModule('numpy')
cv2 = LazyModule('cv2')
matplotlib = LazyModule('matplotlib')
plt = LazyModule('plt', 'import matplotlib.pyplot as plt')
coco = LazyModule('coco', 'from pycocotools import coco')
ipdb = LazyModule('ipdb')
pd = LazyModule('pandas')
speedy = LazyModule('speedy')  # I assumed speedy is a module since it's not imported in the original code. Please replace it with the correct module if needed.

# Removed unused imports and reordered for better readability