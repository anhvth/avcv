# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_process.ipynb.

# %% auto 0
__all__ = ['multi_thread', 'multi_process']

# %% ../nbs/01_process.ipynb 3
import mmcv
from loguru import logger
def multi_thread(fn, array_inputs, max_workers=None, desc="Multi-thread Pipeline", unit="Samples", verbose=True, pbar_iterval=50):
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial
    from glob import glob
    from multiprocessing import Pool
    from tqdm import tqdm

    def _wraper(x):
        i, input = x
        return {i: fn(input)}

    array_inputs = [(i, _) for i, _ in enumerate(array_inputs)]
    if verbose:
        progress_bar = tqdm(total=len(array_inputs))
    outputs = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, result in enumerate(executor.map(_wraper, array_inputs)):
            outputs.update(result)
            if verbose:# and i%pbar_iterval==0:
                progress_bar.update()
    if verbose:
        logger.info('multi_thread')
    outputs = list(outputs.values())
    return outputs


# %% ../nbs/01_process.ipynb 4
def multi_process(f, inputs, num_workers=10):
    from multiprocessing import Pool
    from tqdm import tqdm
    with Pool(10) as p:
        it = p.imap(f, inputs)
        total = len(inputs)
        return list(tqdm(it, total=total))
show_doc(multi_process)
