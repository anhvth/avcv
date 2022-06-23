# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/06_cli.ipynb (unless otherwise specified).

__all__ = ['convert_image']

# Cell
from glob import glob
from .utils import multi_thread
import os.path as osp
from fastcore.script import *
import mmcv
import os
from loguru import logger


@call_parse
def convert_image(path: Param(help="Root dir", type=str, default='**/*.png'),
                  type_from_to: Param(help='Convert image from one type to another default=png->jpg', type=str, default='png->jpg'),
                  remove: Param(help="Remove after converted", type=bool),
                  recursive: Param(help="glob recursive", type=bool)
                 ):
    assert '->' in 'type_from_to must include "->" '
    source_ext, target_ext = type_from_to.split('->')
    paths = glob(path, recursive=recursive)
    if remove:
        logger.info('Files will be removed after converted')
    def f_convert(path):
        dir_name, file_name = osp.dirname(path), osp.basename(path)
        new_file_name = file_name.replace(f'.{source_ext}', f'.{target_ext}')
        new_path = osp.join(dir_name, new_file_name)
        mmcv.imwrite(mmcv.imread(path), new_path)
        if remove:
            os.remove(path)
    multi_thread(f_convert, paths, desc=f"Converting {type_from_to} in {path}")

