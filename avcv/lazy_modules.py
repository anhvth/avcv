# from avcv.utils import memoize



class LazyObject(object):
    # ATTRS = dict()
    def __init__(self, module_name):
        self.module_name = module_name
        self.is_loaded = False
        self.tree = dict()
    def __getattr__(self, item):
        try:
            return self.tree[item]
        except:
            real_module = self.get_real_module()
            for obj_name in dir(real_module):
                self.tree[obj_name] = getattr(real_module, obj_name)
            return self.tree[obj_name]

    def get_real_module(self):
        if not 'real_module' in self.tree:
            exec(f'import {self.module_name}')
            self.tree['real_module'] =  eval(self.module_name)
        return self.tree['real_module']

    def __dir__(self):
        return self.tree.keys()

    def __repr__(self):
        real_module = self.get_real_module()
        return real_module.__repr__()
        
mmcv = LazyObject('mmcv')