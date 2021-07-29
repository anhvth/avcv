# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/00_visualize.ipynb (unless otherwise specified).

__all__ = ['plot_images', 'show', 'plot_images', 'show']

# Cell
import matplotlib.pyplot as plt
import numpy as np

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
    plt.close()

# Cell
import matplotlib.pyplot as plt
import numpy as np

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
    if isinstance(inp, str):
        inp = plt.imread(inp)
    assert len(inp.shape) in [2,3]

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