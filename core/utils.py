import json
import torch
import gdown
from addict import Dict


def load_cfg(path):
    with open(path) as stream:
        cfg = Dict(json.load(stream))
    return cfg


def save_cfg(path, cfg):
    with open(path, 'w') as stream:
        json.dump(cfg, stream, indent=4)
    return cfg


def select_weights(ckpt, prefix="student."):
    _ckpt = {}
    for k, v in ckpt.items():
        if k.startswith(prefix):
            _ckpt[k.replace(prefix, "")] = v
    return _ckpt


def load_weights(target, source_state):
    from collections import OrderedDict
    new_dict = OrderedDict()
    for k, v in target.state_dict().items():
        if k in source_state and v.size() == source_state[k].size():
            new_dict[k] = source_state[k]
        elif k in source_state and v.size() != source_state[k].size():
            print(f"src: {source_state[k].size()}, tgt: {v.size()}")
            new_dict[k] = v
        else:
            print(f"key {k} not loaded...")
            new_dict[k] = v
    target.load_state_dict(new_dict)
