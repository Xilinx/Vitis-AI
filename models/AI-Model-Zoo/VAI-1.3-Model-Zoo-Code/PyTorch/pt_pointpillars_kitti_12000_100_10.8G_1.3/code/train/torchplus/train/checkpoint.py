
# This code is based on: https://github.com/nutonomy/second.pytorch.git
# 
# MIT License
# Copyright (c) 2018 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import logging
import os
import signal
from pathlib import Path

import torch


class DelayedKeyboardInterrupt(object):
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


def latest_checkpoint(model_dir, model_name):
    """return path of latest checkpoint in a model_dir
    Args:
        model_dir: string, indicate your model dir(save ckpts, summarys,
            logs, etc).
        model_name: name of your model. we find ckpts by name
    Returns:
        path: None if isn't exist or latest checkpoint path.
    """
    ckpt_info_path = Path(model_dir) / "checkpoints.json"
    if not ckpt_info_path.is_file():
        return None
    with open(ckpt_info_path, 'r') as f:
        ckpt_dict = json.loads(f.read())
    if model_name not in ckpt_dict['latest_ckpt']:
        return None
    latest_ckpt = ckpt_dict['latest_ckpt'][model_name]
    ckpt_file_name = Path(model_dir) / latest_ckpt
    if not ckpt_file_name.is_file():
        return None
    
    return str(ckpt_file_name)

def _ordered_unique(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

def save(model_dir,
         model,
         model_name,
         global_step,
         max_to_keep=8,
         keep_latest=True):
    """save a model into model_dir.
    Args:
        model_dir: string, indicate your model dir(save ckpts, summarys,
            logs, etc).
        model: torch.nn.Module instance.
        model_name: name of your model. we find ckpts by name
        global_step: int, indicate current global step.
        max_to_keep: int, maximum checkpoints to keep.
        keep_latest: bool, if True and there are too much ckpts, 
            will delete oldest ckpt. else will delete ckpt which has
            smallest global step.
    Returns:
        path: None if isn't exist or latest checkpoint path.
    """

    # prevent save incomplete checkpoint due to key interrupt
    with DelayedKeyboardInterrupt():
        ckpt_info_path = Path(model_dir) / "checkpoints.json"
        ckpt_filename = "{}-{}.tckpt".format(model_name, global_step)
        ckpt_path = Path(model_dir) / ckpt_filename
        if not ckpt_info_path.is_file():
            ckpt_info_dict = {'latest_ckpt': {}, 'all_ckpts': {}}
        else:
            with open(ckpt_info_path, 'r') as f:
                ckpt_info_dict = json.loads(f.read())
        ckpt_info_dict['latest_ckpt'][model_name] = ckpt_filename
        if model_name in ckpt_info_dict['all_ckpts']:
            ckpt_info_dict['all_ckpts'][model_name].append(ckpt_filename)
        else:
            ckpt_info_dict['all_ckpts'][model_name] = [ckpt_filename]
        all_ckpts = ckpt_info_dict['all_ckpts'][model_name]

        torch.save(model.state_dict(), ckpt_path)
        # check ckpt in all_ckpts is exist, if not, delete it from all_ckpts
        all_ckpts_checked = []
        for ckpt in all_ckpts:
            ckpt_path_uncheck = Path(model_dir) / ckpt
            if ckpt_path_uncheck.is_file():
                all_ckpts_checked.append(str(ckpt_path_uncheck))
        all_ckpts = all_ckpts_checked
        if len(all_ckpts) > max_to_keep:
            if keep_latest:
                ckpt_to_delete = all_ckpts.pop(0)
            else:
                # delete smallest step
                get_step = lambda name: int(name.split('.')[0].split('-')[1])
                min_step = min([get_step(name) for name in all_ckpts])
                ckpt_to_delete = "{}-{}.tckpt".format(model_name, min_step)
                all_ckpts.remove(ckpt_to_delete)
            os.remove(str(Path(model_dir) / ckpt_to_delete))
        all_ckpts_filename = _ordered_unique([Path(f).name for f in all_ckpts])
        ckpt_info_dict['all_ckpts'][model_name] = all_ckpts_filename
        with open(ckpt_info_path, 'w') as f:
            f.write(json.dumps(ckpt_info_dict, indent=2))


def restore(ckpt_path, model):
    if not Path(ckpt_path).is_file():
        raise ValueError("checkpoint {} not exist.".format(ckpt_path))
    model.load_state_dict(torch.load(ckpt_path))
    print("Restoring parameters from {}".format(ckpt_path))


def _check_model_names(models):
    model_names = []
    for model in models:
        if not hasattr(model, "name"):
            raise ValueError("models must have name attr")
        model_names.append(model.name)
    if len(model_names) != len(set(model_names)):
        raise ValueError("models must have unique name: {}".format(
            ", ".join(model_names)))


def _get_name_to_model_map(models):
    if isinstance(models, dict):
        name_to_model = {name: m for name, m in models.items()}
    else:
        _check_model_names(models)
        name_to_model = {m.name: m for m in models}
    return name_to_model


def try_restore_latest_checkpoints(model_dir, models):
    name_to_model = _get_name_to_model_map(models)
    for name, model in name_to_model.items():
        latest_ckpt = latest_checkpoint(model_dir, name)
        if latest_ckpt is not None:
            restore(latest_ckpt, model)

def restore_latest_checkpoints(model_dir, models):
    name_to_model = _get_name_to_model_map(models)
    for name, model in name_to_model.items():
        latest_ckpt = latest_checkpoint(model_dir, name)
        if latest_ckpt is not None:
            restore(latest_ckpt, model)
        else:
            raise ValueError("model {}\'s ckpt isn't exist".format(name))

def restore_models(model_dir, models, global_step):
    name_to_model = _get_name_to_model_map(models)
    for name, model in name_to_model.items():
        ckpt_filename = "{}-{}.tckpt".format(name, global_step)
        ckpt_path = model_dir + "/" + ckpt_filename
        restore(ckpt_path, model)


def save_models(model_dir,
                models,
                global_step,
                max_to_keep=15,
                keep_latest=True):
    with DelayedKeyboardInterrupt():
        name_to_model = _get_name_to_model_map(models)
        for name, model in name_to_model.items():
            save(model_dir, model, name, global_step, max_to_keep, keep_latest)
