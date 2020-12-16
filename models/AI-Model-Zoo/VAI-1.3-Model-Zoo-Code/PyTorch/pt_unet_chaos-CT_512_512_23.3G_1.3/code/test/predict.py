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

import os
import sys
from networks.evaluate import evaluate_main_test
import logging
import warnings
warnings.filterwarnings("ignore")
import torch
from torch.utils import data
import numpy as np

sys.path.insert(0,'..')
from networks.UNet import unet
from networks.evaluate import evaluate_main_test
from dataset.datasets import ChaosCTDataValSet as DataSet
import argparse

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

class Configs():
    def __init__(self):

        parser = argparse.ArgumentParser("2D-UNet-LW for Choas-CT segmentation")

        #dataset options
        parser.add_argument('--data_root', type=str, default='../data/', help='path to dataset')

        parser.add_argument('--input_size', type=str, default='512,512', help='input size')
        parser.add_argument('--list_file', type=str, default='dataset/lists/val_img_seg.txt', help='validation list file')
        parser.add_argument('--weight', type=str, default=None, help='resume from weight')
        parser.add_argument('--classes_num', type=int, default=2, help='dataset class number')
        parser.add_argument('--ignore_index', type=int, default=255, help='dataset ignore index')
        parser.add_argument('--save_path', type=str, default='submision_test/', help='test result save path')
        #training options
        parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        print(args)
        return args


def get_dataset(input_size, data_dir):

    valloader = DataSet(root = data_dir,crop_size=(512, 512), mean=IMG_MEAN)
    valloader = valloader.__getitem__(root = data_dir)
    
    return valloader


def main():
    args = Configs.parse()
    
    model = unet(n_classes=args.classes_num)
    model.load_state_dict(torch.load(args.weight))
    filelist = os.listdir(args.data_root)
    for file in filelist:
        count = 0
        data_list = os.listdir(os.path.join(args.data_root, file+'/image/'))
        for item in data_list:
            print('test {} subset with {} / {}'.format(file, count, len(data_list)))
            count += 1
            data_dir = os.path.join(args.data_dir, file+'/image/'+item)
            test_data = get_dataset(input_size = '512,512',data_dir = data_dir)
            save_path = os.path.join(args.save_path, file+'/Results/')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_name = save_path +item[:-4]
            evaluate_main_test(model,test_data, 0, 512, 512, args.classes_num, ignore_label=args.ignore_index, recurrence = 1, save_path=save_name)

if __name__ == '__main__':
    main()
