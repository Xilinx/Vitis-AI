# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
import logging
import os
import pdb
import platform
import shutil
import sys
import time
import torch
from tensorboardX import SummaryWriter

def to_tuple_str(str_first, gpu_num, str_ind):
    if gpu_num > 1:
        tmp = '(' 
        for cpu_ind in range(gpu_num):
            tmp += '(' + str_first + '[' + str(cpu_ind) + ']' + str_ind +',)'  
            if cpu_ind != gpu_num-1: tmp +=  ', '
        tmp += ')'
    else:
        tmp = str_first + str_ind  
    return tmp

def to_cat_str(str_first, gpu_num, str_ind, dim_):
    if gpu_num > 1:
        tmp = 'torch.cat((' 
        for cpu_ind in range(gpu_num):
            tmp += str_first + '[' + str(cpu_ind) + ']' + str_ind  
            if cpu_ind != gpu_num-1: tmp +=  ', '
        tmp += '), dim=' + str(dim_) + ')'
    else:
        tmp = str_first + str_ind  
    return tmp

def to_tuple(list_data, gpu_num, sec_ind):
    out = (list_data[0][sec_ind],)
    for ind in range(1,gpu_num):
        out += (list_data[ind][sec_ind],)
    return out

def log_init(log_dir, name='log'):
    time_cur = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    if os.path.exists(log_dir) == False:
        os.makedirs(log_dir)
    logging.basicConfig(filename=log_dir + '/' + name + '_' + str(time_cur) + '.log',
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def write_tensorboder_logger(logger_path, epoch, **info):
    if os.path.exists(logger_path) == False:
        os.makedirs(logger_path)
    writer = SummaryWriter(logger_path)
    writer.add_scalars('accuracy',{'train_accuracy': info['train_accuracy'], 'test_accuracy': info['test_accuracy']}, epoch)
    for tag, value in info.items():
        if tag not in ['train_accuracy', 'test_accuracy']:
            writer.add_scalar(tag, value, epoch)
    writer.close()

def save_arg(args):
    #l = len(args.ckpt_path.split('/')[-1])
    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
    f = open(os.path.join(args.ckpt_path, 'arguments.txt'), 'w+')
    for key, val in args._get_kwargs():
        f.write(key + ' : ' + str(val)+'\n')
    f.close()


def load_S_model(args, model, with_module=True):
    logging.info('using load_ckpt to load pretrained weights....')
    load_ckpt(args, model, with_module)

def load_ckpt_dsbn(args, model,with_module = True):
    logging.info("------------")
    if args.is_load_imgnet:
        if os.path.isfile(args.pretrain_model_imgnet):
            saved_state_dict=torch.load(args.pretrain_model_imgnet)
            new_params=model.state_dict()
            updated_state_dict = _update_initial_weights_dsbn(state_dict=saved_state_dict, num_domains=args.domain_num)
            model.load_state_dict(updated_state_dict, strict=False)
            logging.info("=> load" + str(args.pretrain_model_imgnet))
        else:
            logging.info("=> the pretrain model on imgnet '{}' does not exit".format(args.pretrain_model_imgnet))


def load_ckpt(args, model, with_module = True):
    logging.info("------------")
    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
    if args.is_load_imgnet:
        if os.path.isfile(args.pretrain_model_imgnet):
            saved_state_dict=torch.load(args.pretrain_model_imgnet)
            new_params=model.state_dict()
            saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in new_params}
            new_params.update(saved_state_dict)
            model.load_state_dict(new_params)
            logging.info("=> load" + str(args.pretrain_model_imgnet))
        else:
            logging.info("=> the pretrain model on imgnet '{}' does not exit".format(args.pretrain_model_imgnet))
    else:
        if args.resume:
            file = args.ckpt_path + '/model_best.pth.tar'
            if os.path.isfile(file):
                checkpoint = torch.load(file)
                args.last_step = checkpoint['step'] if 'step' in checkpoint else None
                args.start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else None
                args.best_mean_IU = checkpoint['best_mean_IU'] if 'best_mean_IU' in checkpoint else None
                best_IU_array = checkpoint['IU_array'] if 'IU_array' in checkpoint else None
                state_dict = checkpoint['state_dict']
                new_state_dict = OrderedDict()
                if with_module == False:
                    new_state_dict = {k[7:]: v for k,v in state_dict.items()}
                else:
                    new_state_dict = state_dict
                model.load_state_dict(new_state_dict)
                logging.info("=> loaded checkpoint '{}' \n (epoch:{} step:{} best_mean_IU:{} \n )".format(
                    file, args.start_epoch, args.last_step, args.best_mean_IU, best_IU_array))
                logging.info("the best student accuracy is: %.3f", args.best_mean_IU)
            else:
                logging.info("=> checkpoint '{}' does not exit".format(file))
    logging.info("------------")


def save_checkpoint(state, is_best, fdir):
    filepath = os.path.join(fdir, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth.tar'))

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    return lr

def print_model_parm_nums(model, string):
    b = []
    for param in model.parameters():
        b.append(param.numel())
    logging.info(string + ': Number of params: %.2fM', sum(b) / 1e6)

