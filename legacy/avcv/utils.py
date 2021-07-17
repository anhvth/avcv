import inspect
import json
import os
import os.path as osp
import pickle
import shelve
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from glob import glob
from multiprocessing import Pool

import mmcv
import numpy as np
import xxhash
from six.moves import map, zip
from tqdm import tqdm

def pp(d):
    import pprint
    _ = pprint.PrettyPrinter(4)
    _.pprint(d)


def mkdir(d):
    return os.makedirs(d, exist_ok=True)


def path2filename(path, with_ext=False):
    """Returns name of the file"
    """
    name = osp.basename(path)
    return name if with_ext else name.split('.')[0]


def print_source(x):
    print(inspect.getsource(x))
    print("---- Define in:")
    print(f"{inspect.getsourcefile(x)}: {x.__code__.co_firstlineno}")


def lib_reload(some_module):
    import importlib
    return importlib.reload(some_module)


def do_by_chance(chance):
    assert chance > 1 and chance < 100
    return np.random.uniform() < chance/100


def get_paths(directory, input_type='png', sort=True):
    """
        Get a list of input_type paths
        params args:
        return: a list of paths
    """
    paths = glob(os.path.join(directory, '*.{}'.format(input_type)))
    if sort:
        paths = list(sorted(paths))
    print('Found and sorted {} files {}'.format(len(paths), input_type))
    return paths


def read_json(path):
    '''Read a json path.
        Arguments: 
            path: string path to json 
        Returns:
             A dictionary of the json file
    '''
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def multi_process(fn, array_inputs, max_workers=4, verbose=True):
    print(f"Multi-Process: {max_workers} workers")
    with Pool(max_workers) as p:
        if verbose:
            r = list(tqdm(p.imap(fn, array_inputs), total=len(array_inputs)))
        else:
            r = list(p.imap(fn, array_inputs))

    return r


def multi_thread(fn, array_inputs, max_workers=None, desc="Multi-thread Pipeline", unit=" Samples", verbose=False):
    def _wraper(x):
        i, input = x
        return {i: fn(input)}

    array_inputs = [(i, _) for i, _ in enumerate(array_inputs)]
    if verbose:
        with tqdm(total=len(array_inputs), desc=desc, unit=unit) as progress_bar:
            outputs = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for result in executor.map(_wraper, array_inputs):
                    outputs.update(result)
                    progress_bar.update(1)
    else:
        outputs = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(_wraper, array_inputs):
                outputs.update(result)
    if verbose:
        print('Finished')
    outputs = list(outputs.values())
    return outputs


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' %
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed


def noralize_filenames(directory, ext='*'):
    paths = glob('{}.{}'.format(directory, ext))
    for i, path in enumerate(paths):
        base_dir, base_name = os.path.split(path)
        name, base_ext = base_name.split('.')
        new_name = '{:0>4d}.{}'.format(i, base_ext)
        new_path = os.path.join(base_dir, new_name)
        print('Rename: {} --> {}'.format(path, new_path))
        os.rename(path, new_path)


def identify(x):
    '''Return an hex digest of the input'''
    return xxhash.xxh64(pickle.dumps(x), seed=0).hexdigest()


def memoize(func):
    import os
    import pickle
    from functools import wraps

    import xxhash

    '''Cache result of function call on disk
    Support multiple positional and keyword arguments'''
    @wraps(func)
    def memoized_func(*args, **kwargs):
        cache_dir = 'cache'
        try:
            import inspect
            func_id = identify((inspect.getsource(func), args, kwargs))
            cache_path = os.path.join(cache_dir, func.__name__+'_'+func_id)

            if (os.path.exists(cache_path) and
                    not func.__name__ in os.environ and
                    not 'BUST_CACHE' in os.environ):
                result = pickle.load(open(cache_path, 'rb'))
            else:
                result = func(*args, **kwargs)
                os.makedirs(cache_dir, exist_ok=True)
                pickle.dump(result, open(cache_path, 'wb'))
            return result
        except (KeyError, AttributeError, TypeError):
            return func(*args, **kwargs)
    return memoized_func




def show_df(df, path_column=None, max_col_width=-1, height=512):
    """
        Turn a DataFrame which has the image_path column into a html table
            with watchable images
        Argument:
            df: the origin dataframe
            path_column: the column name which contains the paths to images
        Return:
            HTML object, a table with images
    """
    assert path_column is not None, 'if you want to show the image then tell me which column contain the path? if not what the point to use this?'
    import base64
    from io import BytesIO

    import cv2
    import pandas
    from IPython.display import HTML
    from PIL import Image

    pandas.set_option('display.max_colwidth', max_col_width)

    def get_thumbnail(path):
        img = cv2.imread(path)
        h, w = img.shape[:2]
        f = height/h
        img = cv2.resize(img, (0, 0), fx=f, fy=f)
        return Image.fromarray(img)

    def image_base64(im):
        if isinstance(im, str):
            im = get_thumbnail(im)
        with BytesIO() as buffer:
            im.save(buffer, 'jpeg')
            return base64.b64encode(buffer.getvalue()).decode()

    def image_formatter(im):
        return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'

    return HTML(df.to_html(formatters={path_column: image_formatter}, escape=False))


def get_name(path):
    path = osp.basename(path).split('.')[:-1]
    return '.'.join(path)


def download_file_from_google_drive(id, destination):
    import requests

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def parse_args(parser):
    cache_path = osp.join(os.environ['HOME'], '.cache', identify(
        osp.abspath(__file__))+'.json')
    # import ipdb; ipdb.set_trace()
    # try:
    args = parser.parse_args()
    args = mmcv.Config(args.__dict__)
    os.makedirs(osp.dirname(cache_path), exist_ok=True)
    mmcv.dump(args.__dict__, cache_path)
    # except Exception as e:
    # pp = pprint.PrettyPrinter(depth=4)
    # args = mmcv.Config(mmcv.load(cache_path))
    # pp.pprint(args.__dict__)
    # print('Exception: ', e)
    return args


# def change_vscode_run_cmd()


def save_workspace(filename, names_of_spaces_to_save, dict_of_values_to_save):
    '''
        filename = location to save workspace.
        names_of_spaces_to_save = use dir() from parent to save all variables in previous scope.
            -dir() = return the list of names in the current local scope
        dict_of_values_to_save = use globals() or locals() to save all variables.
            -globals() = Return a dictionary representing the current global symbol table.
            This is always the dictionary of the current module (inside a function or method,
            this is the module where it is defined, not the module from which it is called).
            -locals() = Update and return a dictionary representing the current local symbol table.
            Free variables are returned by locals() when it is called in function blocks, but not in class blocks.

        Example of globals and dir():
            >>> x = 3 #note variable value and name bellow
            >>> globals()
            {'__builtins__': <module '__builtin__' (built-in)>, '__name__': '__main__', 'x': 3, '__doc__': None, '__package__': None}
            >>> dir()
            ['__builtins__', '__doc__', '__name__', '__package__', 'x']
    '''
    print('save_workspace')
    print('C_hat_bests' in names_of_spaces_to_save)
    print(dict_of_values_to_save)
    my_shelf = shelve.open(filename, 'n')  # 'n' for new
    for key in names_of_spaces_to_save:
        try:
            my_shelf[key] = dict_of_values_to_save[key]
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            #print('ERROR shelving: {0}'.format(key))
            pass
    my_shelf.close()


def load_workspace(filename, parent_globals):
    '''
        filename = location to load workspace.
        parent_globals use globals() to load the workspace saved in filename to current scope.
    '''
    my_shelf = shelve.open(filename)
    for key in my_shelf:
        parent_globals[key] = my_shelf[key]
    my_shelf.close()


def change_vscode_run_cmd(cmd):
    # print("Cmd:",cmd)
    file_sh = osp.join(os.environ['HOME'], '.cache/', 'vs_cmd.sh')
    mkdir(os.path.dirname(file_sh))
    with open(file_sh, 'w') as f:
        f.write(f"{cmd}")
        f.write("\necho")
        f.write("\necho")
        f.write("\necho ---------------------------------")
        f.write(f"\necho '{cmd}'")



def parse_debug_command(cmd):
    # "python tools/test.py configs/road_seg/v1.py work_dirs/v1-finetune/iter_1000.pth --show --show-dir ./cache/"
    for _ in range(3):
        cmd = cmd.replace('  ', '')
    i_split = cmd.index(".py")+4
    file = cmd[:i_split].strip().split(' ')[1]

    args = cmd[i_split:].split(' ')

    cfg = {
        "name": "Python: Latest-Generated",
        "type": "python",
        "request": "launch",
        "program": f"{file}",
        "console": "integratedTerminal",
        "args": args,
    }
    # pp(cfg)
    mkdir(".vscode")
#     if osp.exists(".vscode/launch.json"):
    try:
        lauch = read_json(".vscode/launch.json")
#     else:
    except:
        lauch = {
            "version": "0.2.0",
            "configurations": [

            ]
        }
    replace = False
    for i, _cfg in enumerate(lauch['configurations']):
        if _cfg["name"] == cfg["name"]:
            lauch["configurations"][i] = cfg
            replace = True
    if not replace:
        lauch["configurations"] += [cfg]
    with open('.vscode/launch.json', 'w') as f:
        json.dump(lauch, f, indent=4)
    pp(lauch)


def copy_unzip(input_dir, output_dir):
    def process_path(path:str):
        relative_path = osp.relpath(path, input_dir)
        out_rel_dir = osp.dirname(relative_path)+get_name(relative_path)
        if path.endswith('.zip'):
            output_path = osp.abspath(osp.join(output_dir, out_rel_dir))
            return f"7z x {path} -o{output_path}"
        elif path[:-1].endswith('.zip.00'):
            if int(path.split('.')[-1]) == 1:
                output_path = osp.abspath(osp.join(output_dir, out_rel_dir))
                return f"7z x {path} -o{output_path}"
        else:
            output_path = osp.abspath(osp.join(output_dir, out_rel_dir))
            mkdir(output_path)
            return f"cp -r {path} {output_path}"


    def get_path(input_dir):
        if osp.isdir(input_dir):
            rt = []
            lst_files = os.listdir(input_dir)
            outs = [get_path(osp.join(input_dir, file)) for file in lst_files]
            for out in outs:
                if isinstance(out, str):
                    out_path = osp.abspath(out)
                    
                    rt.append(out_path)
                else:
                    rt.extend(out)
            return rt
            
        else:
            return input_dir

    cmds = []
    for path in get_path(input_dir):
        cmd = process_path(path)
        if cmd is not None:
            cmds.append(cmd)

    cmds = list(sorted(cmds, key=lambda x:x))
    workers=5
    with open("cmd.sh", "w") as f:
        for i, cmd in enumerate(cmds):
            if i>1 and i%5 == 0:
                f.write(cmd+"\n")
            else:
                f.write(cmd+" | ")

    print("cmd.sh")

def convert_image(path, out_dir, mode="png->jpg"):
    from_type, to_type = mode.split('->')
    paths = get_paths(path, from_type)
    # for path in paths:
    def f(path):
        img = mmcv.imread(path)
        out_path = osp.join(out_dir, osp.basename(path).replace(f'.{from_type}', f'.{to_type}'))
        mmcv.imwrite(img, out_path)
    multi_thread(f, paths, verbose=1)


class NameIdHash:
    def __init__(self):
        self.mapping = dict()

    def __call__(self, cat):
        if not cat in self.mapping:
            id = len(self.mapping)
            self.mapping[cat] = id
        return self.mapping[cat]

if __name__ == '__main__':
    def f(i):
        return i*2

    r = multi_process(f, range(10))
