from avcv.vision import *
from avcv.vision import *

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=["images_to_video"])
    parser.add_argument('argument', nargs="+")
    args = parser.parse_args()
    fn = eval(args.task)
    print("Running ", fn.__name__)
    print("Argument", "value")
    for value, var_name in zip(args.argument, fn.__code__.co_varnames):
        print(var_name, ":", value)
    fn(*args.argument)

