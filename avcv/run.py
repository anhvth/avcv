from avcv.vision import *
from avcv.utils import *
from avcv.debug_utils import *
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
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("\n", "-"*40,"\nAvailable function:")
        n_suggestion = 0
        for func in dir():
            if not '__' in func and args.task in func:
                n_suggestion += 1
                print(func)
        if n_suggestion == 0:

            print("Found no suggestion")
