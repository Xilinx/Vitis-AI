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


#coding=utf-8
import os 
import cv2
import sys
import json
import numpy as np
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="Imagenet resize script")
    parser.add_argument('--data-dir', '-d', type=str,
            default='./val_dataset',
            help='path to imagenet val images')
    parser.add_argument('--output-dir', '-o', type=str,
            default='./val_resize_256',
            help='reisize output path')
    parser.add_argument('--anno-file', '-a', type=str,
            default='val.txt',
            help='gen anno fileanme')
    parser.add_argument('--short-size', '-s', type=int,
            default=256,
            help='resize short size')
    parser.add_argument('--name2class_file', '-nc', type=str,
            default="imagenet_class_index.json",
            help='imagenet name to class file')
    return parser.parse_args()

args = parse_args()

def resize_shortest_edge(image, size):
    H, W = image.shape[:2]
    if H >= W:
        nW = size
        nH = int(float(H)/W * size)
    else:
        nH = size
        nW = int(float(W)/H * size)
    return cv2.resize(image,(nW,nH))

def gen_dict(name2class_file):
   fr = open(name2class_file, 'r')
   class2name_dict = json.load(fr)  
   name2class_dict = {}
   for key in class2name_dict.keys():
       name2class_dict[class2name_dict[key][0]] = key
   return name2class_dict

def gen_dataset(args):
   if not os.path.exists(args.output_dir):
       os.system('mkdir ' + args.output_dir)
   name2class_dict = gen_dict(args.name2class_file)
   classname_list = os.listdir(args.data_dir)
   fwa = open(args.anno_file, "w")
   for classname in classname_list:
       class_dir = os.path.join(args.data_dir, classname) 
       class_id = name2class_dict[classname]
       imagename_list = os.listdir(class_dir)
       for imagename in imagename_list:
           imagename_path = os.path.join(class_dir, imagename)
           os.system('cp ' + imagename_path + ' ' + args.output_dir + '/')
           fwa.writelines(imagename + ' ' + str(class_id) + '\n')        
   fwa.close()
 
if __name__ == "__main__":
   gen_dataset(args)
