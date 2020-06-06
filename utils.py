import torch

def is_gpu_available():
    return torch.cuda.is_available()


# from https://github.com/rafaljozefowicz/lm
class HParams(object):
    def __init__(self, **kwargs):
        self._items = {}
        for k, v in kwargs.items():
            self._set(k, v)

    def _set(self, k, v):
        self._items[k] = v
        setattr(self, k, v)

    def parse(self, str_value):
        hps = HParams(**self._items)
        for entry in str_value.strip().split(","):
            entry = entry.strip()
            if not entry:
                continue
            key, sep, value = entry.partition("=")
            if not sep:
                raise ValueError("Unable to parse: %s" % entry)
            default_value = hps._items[key]
            if isinstance(default_value, bool):
                hps._set(key, value.lower() == "true")
            elif isinstance(default_value, int):
                hps._set(key, int(value))
            elif isinstance(default_value, float):
                hps._set(key, float(value))
            else:
                hps._set(key, value)
        return hps

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self._set(k, v)

    def show(self):
        print("\n----------- Current Hyperparameter Settings -----------")
        for k, v in self._items.items():
            print( u'{} : {}'.format(k,v) )
        print("-------------------------------------------------------\n")

    def get_all_param(self):
        all_param = {}
        for k, v in self._items.items():
            all_param[k] = v
        return all_param

import os
def prepare_dir(dir_name):
    if not os.path.exists(dir_name): os.makedirs(dir_name)


def load_model(model_fn, map_location=None):
    if map_location:
        return torch.load(model_fn, map_location=map_location)
    else:
        if torch.cuda.is_available():
            return torch.load(model_fn)
        else:
            return torch.load(model_fn, map_location='cpu')