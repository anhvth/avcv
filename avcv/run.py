from avcv.vision import *
from avcv.utils import *

if __name__ == '__main__':
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('task')
        parser.add_argument('--arguments', '-a', nargs="+", default=None)
        args = parser.parse_args()

        fn = eval(args.task)
        
        if args.arguments is  None:
            print(f"------Function  {fn.__name__} deffinition------------------")
            print_source(fn)
            print("----------------------------------------------------")
            
        else:
            print("Argument", "value")
            for value, var_name in zip(args.arguments, fn.__code__.co_varnames):
                print(var_name, ":", value)
            fn(*args.arguments)
    except:
        print("Available function:")
        for func in dir():
            if not '__' in func:
                print(func)
