class LazyObject(object):
    def __init__(self, module_name, old_import=None):
        self.module_name = module_name
        self.old_import = old_import
        self.__tree = dict()
    def __getattr__(self, item):
        try:
            return self.__tree[item]
        except:
            real_module = self.get_real_module()
            for obj_name in dir(real_module):
                self.__tree[obj_name] = getattr(real_module, obj_name)
            return self.__tree[obj_name]

    def get_real_module(self):
        if not 'real_module' in self.__tree:
            if self.old_import is None:
                exec(f'import {self.module_name}')
            else:
                exec(self.old_import)
            self.__tree['real_module'] =  eval(self.module_name)
        return self.__tree['real_module']

    def __dir__(self):
        return self.__tree.keys()

    def __repr__(self):
        real_module = self.get_real_module()
        return real_module.__repr__()
    def __help__(self):
        real_module = self.get_real_module()
        return help(real_module)

if __name__ == '__main__':
    plt = LazyObject('plt', 'import matplotlib.pyplot as plt')
