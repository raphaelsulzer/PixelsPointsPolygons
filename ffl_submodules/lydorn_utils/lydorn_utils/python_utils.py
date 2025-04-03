#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import errno
import json
from jsmin import jsmin
from lydorn_utils import print_utils


def module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True


def choose_first_existing_path(path_list, return_tried_paths=False):
    paths_tried = []
    for path in path_list:
        path = os.path.expanduser(path)
        paths_tried.append(path)
        if os.path.exists(path):
            if return_tried_paths:
                return path, paths_tried
            else:
                return path

    if return_tried_paths:
        return None, paths_tried
    else:
        return None


def get_display_availability():
    return "DISPLAY" in os.environ


def get_filepaths(dir_path, endswith_str="", startswith_str="", not_endswith_str=None, not_startswith_str=None):
    if os.path.isdir(dir_path):
        image_filepaths = []
        for path, dnames, fnames in os.walk(dir_path):
            fnames = sorted(fnames)
            image_filepaths.extend([os.path.join(path, x) for x in fnames
                                    if x.endswith(endswith_str)
                                    and x.startswith(startswith_str)
                                    and (not_endswith_str is None or not x.endswith(not_endswith_str))
                                    and (not_startswith_str is None or not x.startswith(not_startswith_str))])
        return image_filepaths
    else:
        raise NotADirectoryError(errno.ENOENT, os.strerror(errno.ENOENT), dir_path)


def get_dir_list_filepaths(dir_path_list, endswith_str="", startswith_str=""):
    image_filepaths = []
    for dir_path in dir_path_list:
        image_filepaths.extend(get_filepaths(dir_path, endswith_str=endswith_str, startswith_str=startswith_str))
    return image_filepaths


def save_json(filepath, data):
    dirpath = os.path.dirname(filepath)
    os.makedirs(dirpath, exist_ok=True)
    with open(filepath, 'w') as outfile:
        json.dump(data, outfile)
    return True


def load_json(filepath):
    if not os.path.exists(filepath):
        return False
    try:
        with open(filepath, 'r') as f:
            minified = jsmin(f.read())
            data = json.loads(minified)
    except json.decoder.JSONDecodeError as e:
        print_utils.print_error(f"ERROR in load_json(filepath): {e} from JSON at {filepath}")
        exit()
    return data


def wipe_dir(dirpath):
    filepaths = get_filepaths(dirpath)
    for filepath in filepaths:
        os.remove(filepath)


def split_list_into_chunks(l, n, pad=False):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        if pad:
            chunk = l[i:i + n]
            if len(chunk) < n:
                chunk.extend([chunk[-1]]*(n - len(chunk)))
            yield chunk
        else:
            yield l[i:i + n]


def params_to_str(params):
    def to_str(value):
        if type(value) == float and value == int(value):
            return str(int(value))
        return str(value)

    return "_".join(["{}_{}".format(key, to_str(params[key])) for key in sorted(params.keys())])


def main():
    l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    batches = split_list_into_chunks(l, 4, pad=True)
    for batch in batches:
        print(batch)


if __name__ == '__main__':
    main()
