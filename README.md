# AVCV
> Summary description here.


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

Fill me in please! Don't forget code examples:

### Plot images

```python
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

> <code>multi_thread</code>(**`fn`**, **`array_inputs`**, **`max_workers`**=*`None`*, **`desc`**=*`'Multi-thread Pipeline'`*, **`unit`**=*`' Samples'`*, **`verbose`**=*`False`*)




```python
from glob import glob
import mmcv
import numpy as np


paths = glob('/data/synthetic/SHARE_SVA_DATASET/val/000/frames/*')
def f(x):
    return mmcv.imread(x, channel_order='rgb')

imgs = multi_thread(f, np.random.choice(paths, 100))
print(len(imgs))
```

    100

