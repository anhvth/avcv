# AVCV
> Optimized functions for vision problems


```python
%load_ext autoreload
%autoreload 2
```

```python
from nbdev.showdoc import *
```

This file will become your README and also the index of your documentation.

## Install

`pip install avcv`

## How to use


<h4 id="plot_images" class="doc_header"><code>plot_images</code><a href="https://github.com/anhvth/avcv/tree/main/avcv/visualize.py#L9" class="source_link" style="float:right">[source]</a></h4>

> <code>plot_images</code>(**`images`**, **`labels`**=*`None`*, **`cls_true`**=*`None`*, **`cls_pred`**=*`None`*, **`space`**=*`(0.3, 0.3)`*, **`mxn`**=*`None`*, **`size`**=*`(5, 5)`*, **`dpi`**=*`300`*, **`max_w`**=*`1500`*, **`out_file`**=*`None`*, **`cmap`**=*`'binary'`*)




### Plot images

```python
from avcv.visualize import plot_images
from glob import glob
import numpy as np
import mmcv
paths = glob('/data/synthetic/SHARE_SVA_DATASET/val/000/frames/*')
imgs = [mmcv.imread(path, channel_order='rgb') for path in np.random.choice(paths, 10)]
plot_images(imgs)
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-4-c232fb9b8e39> in <module>
    ----> 1 from avcv.plot_images import plot_images
          2 from glob import glob
          3 import numpy as np
          4 import mmcv
          5 paths = glob('/data/synthetic/SHARE_SVA_DATASET/val/000/frames/*')


    ModuleNotFoundError: No module named 'avcv.plot_images'


### Multi thread

Elementwise multithreading a given function, the results are store in an array coresponding to the inputs

```python
# example
from glob import glob
import mmcv
import numpy as np
from avcv.process import multi_thread
from tqdm import tqdm

paths = glob('/data/synthetic/SHARE_SVA_DATASET/val/000/frames/*')
def f(x):
    mmcv.imread(x, channel_order='rgb')
    return None

inputs = np.random.choice(paths, 100)
fast_imgs = multi_thread(f, inputs)
```

```python
slow_imgs = [f(_) for _ in tqdm(inputs)]
```
