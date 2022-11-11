class LazyModule(object):
    def __init__(self, module_name, old_import=None):
        self.module_name = module_name
        self.old_import = old_import
        self.__tree = dict()

    def __getattr__(self, item):
        real_module = self.get_real_module()
        return getattr(real_module, item)

    def get_real_module(self):
        if not 'real_module' in self.__tree:
            if self.old_import is None:
                exec(f'import {self.module_name}')
            else:
                exec(self.old_import)
            module = eval(self.module_name)
            assert not str(module).startswith('<class '), "Lazy module does not support import this class,\
plese use normal import or use the parrent module"
            self.__tree['real_module'] =  module

        return self.__tree['real_module']

    def __dir__(self):
        return dir(self.get_real_module())

    def __repr__(self):
        real_module = self.get_real_module()
        return repr(real_module)

if __name__ == '__main__':
    plt = LazyModule('plt', 'import matplotlib.pyplot as plt')
    mmcv = LazyModule('mmcv')

    # COCO = LazyModule('COCO', 'from pycocotools.coco import COCO')
    # print(COCO)
    # class COCO2(coco.COCO):
    #     pass