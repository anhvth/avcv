class LazyObject(object):
    ATTRS = dict()
    def __init__(self, module_name):
        self.module_name = module_name
        self.__loaded = False

    def __getattr__(self, *args, **kwargs):
        if not self.__loaded:
            self.load()
        return getattr(self, *args)
    def load(self):
        exec(f'import {self.module_name}')
        self.__real = eval(self.module_name)
        ds = dir(self.__real)
        for obj_name in ds:
            obj = eval(f'{self.module_name}.{obj_name}')
            setattr(self, obj_name, obj)
            self.ATTRS[obj_name] = obj
        self.__loaded = True
    def __repr__(self):
        return self.__real.__repr__()


