# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_process.ipynb.

# %% auto 0
__all__ = ['multi_thread', 'multi_process']

# %% ../nbs/01_process.ipynb 2
from ._imports import *

# %% ../nbs/01_process.ipynb 3
def multi_thread(fn, array_inputs, max_workers=None, 
                 desc="", unit="Samples", 
                 verbose=True, pbar_iterval=10):

    def _wraper(x):
        i, input = x
        return {i: fn(input)}

    array_inputs = [(i, _) for i, _ in enumerate(array_inputs)]
    if verbose:
        logger.info(desc)
        progress_bar = mmcv.ProgressBar(len(array_inputs))#tqdm(total=len(array_inputs))
    outputs = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, result in enumerate(executor.map(_wraper, array_inputs)):
            outputs.update(result)
            if verbose and i%pbar_iterval==0:
                progress_bar.update(pbar_iterval)
    # if verbose:
        # logger.info('multi_thread {}, {}', fn.__name__, desc)
    outputs = list(outputs.values())
    return outputs




# %% ../nbs/01_process.ipynb 5
def multi_process(f, inputs, max_workers=8, desc='',
               unit='Samples', verbose=True, pbar_iterval=10):
    if verbose:
        
        logger.info('Multi processing {} | Num samples: {}', f.__name__, len(inputs))
        pbar = mmcv.ProgressBar(len(inputs))
        
    with Pool(max_workers) as p:
        it = p.imap(f, inputs)
        total = len(inputs)
        # return list(tqdm(it, total=total))
        return_list = []
        for i, ret in enumerate(it):
            return_list.append(ret)
            if i % pbar_iterval == 0 and verbose:
                pbar.update(pbar_iterval)
    return return_list


