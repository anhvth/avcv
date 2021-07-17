# AVCV
> Optimized functions for vision problems


```
%load_ext autoreload
%autoreload 2
```

```
from nbdev.showdoc import *
```

This file will become your README and also the index of your documentation.

## Install

`pip install avcv`

## How to use


<h4 id="plot_images" class="doc_header"><code>plot_images</code><a href="https://github.com/anhvth/avcv/tree/main/avcv/visualize.py#L9" class="source_link" style="float:right">[source]</a></h4>

> <code>plot_images</code>(**`images`**, **`labels`**=*`None`*, **`cls_true`**=*`None`*, **`cls_pred`**=*`None`*, **`space`**=*`(0.3, 0.3)`*, **`mxn`**=*`None`*, **`size`**=*`(5, 5)`*, **`dpi`**=*`300`*, **`max_w`**=*`1500`*, **`out_file`**=*`None`*, **`cmap`**=*`'binary'`*)




### Plot images

```
from avcv.plot_images import plot_images
from glob import glob
import numpy as np
import mmcv
paths = glob('/data/synthetic/SHARE_SVA_DATASET/val/000/frames/*')
imgs = [mmcv.imread(path, channel_order='rgb') for path in np.random.choice(paths, 10)]
plot_images(imgs)
```

    (3, 3)



![png](docs/images/output_8_1.png)


### Multi thread


<h4 id="multi_thread" class="doc_header"><code>multi_thread</code><a href="https://github.com/anhvth/avcv/tree/main/avcv/process.py#L6" class="source_link" style="float:right">[source]</a></h4>

> <code>multi_thread</code>(**`fn`**, **`array_inputs`**, **`max_workers`**=*`None`*, **`desc`**=*`'Multi-thread Pipeline'`*, **`unit`**=*`'Samples'`*, **`verbose`**=*`True`*)




```
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

    Multi-thread Pipeline: 100%|██████████| 100/100 [00:00<00:00, 261.36Samples/s]

    Finished


    


```
slow_imgs = [f(_) for _ in tqdm(inputs)]
```

    100%|██████████| 100/100 [00:02<00:00, 49.02it/s]

