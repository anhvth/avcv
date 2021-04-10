import mmcv
# from urllib.request import urlopen
from .utils import get_name, mkdir, osp, os


# @memoize
# def get_file(filename):
    
#     return 




class Config(mmcv.Config):
    @staticmethod
    def fromfile(filename,
                 use_predefined_variables=True,
                 import_custom_modules=True):
        if 'https://raw.githubusercontent.com/' in filename and "configs" in filename:
            name = filename.split("configs/")[-1]
            out_file = f'./configs/{name}'
            if not osp.exists(out_file):
                mkdir(osp.dirname(out_file))
                os.system(f"wget {filename} -O {out_file}")
            filename = out_file
            print("Online config saved->", filename)
        
        
        return mmcv.Config.fromfile(filename, use_predefined_variables, import_custom_modules)
