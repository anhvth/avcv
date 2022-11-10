from fastcore.all import threaded
import time

class LazyObject:
    def __init__(self, module_name):
        self.module_name = module_name
        self.done = False
        self.get_attr()

    @threaded
    def get_attr(self):
        print(f'import {self.module_name}')
        exec(f'import {self.module_name}')

        ds = dir(eval(self.module_name))
        for obj_name in ds:
            obj = eval(f'{self.module_name}.{obj_name}')
            setattr(self, obj_name, obj)
        self.done = True

mmcv = LazyObject('mmcv')
pd = LazyObject('pandas')
