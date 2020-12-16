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


#!/usr/bin/python
#-*-coding:utf-8-*-
import os
import json
from argparse import ArgumentParser
try:
    import commands
except ImportError:
    import subprocess as commands


def parse_args():
    parser = ArgumentParser(description="A script to convert voc-like annotations and image to ssd model require's lmdb.")
    parser.add_argument('--caffe-dir', '-c', type=str,
            default='../../../caffe-xilinx/',
            help='path to caffe')
    parser.add_argument('--image-dir', '-i', type=str,
            default='../../data/coco2014/Images/',
            help='path to images')
    parser.add_argument('--anno-dir', '-a', type=str,
            default='../../data/coco2014/Annotations/',
            help='path to Annotations')
    parser.add_argument('--train-image-list', '-t', type=str,
            default='../../data/coco2014/train2014.txt',
            help='path to train image list file')
    parser.add_argument('--train-image2anno', '-ti2a', type=str,
            default='../../data/coco2014/train_image2anno.txt',
            help='path to train image2anno file')
    parser.add_argument('--val-image-list', '-v', type=str,
            default='../../data/coco2014/val2014.txt',
            help='path to val image list file')
    parser.add_argument('--val-image2anno', '-vi2a', type=str,
            default='../../data/coco2014/val_image2anno.txt',
            help='path to val image2anno file')
    parser.add_argument('--labelmap', '-l', type=str,
            default='../../data/labelmap.prototxt',
            help='labelmap of model')
    parser.add_argument('--width', '-w', type=int,
            default=480,
            help='resize width')
    parser.add_argument('--height', '-he', type=int,
            default=360,
            help='resize height')
    parser.add_argument('--output', '-o', type=str,
            default='../../data/coco2014_lmdb',
            help='path to output lmdb')
    return parser.parse_args()


args = parse_args()

caffe_root = args.caffe_dir
image_dir = args.image_dir
map_file = args.labelmap
label_type = 'txt'
anno_type = 'detection'
min_dim = 0
max_dim = 0
extra_cmd = '--encode-type=jpg --encoded --redo'


def gen_image2anno(image_root, anno_root, image_list_file, image2anno_file):
    assert os.path.exists(image_root), 'image_root:[{}] not exist! '.format(image_root)
    assert os.path.exists(image_list_file), 'image_list_file:[{}] not exist!'.format(image_list_file) 
    with open(image_list_file, 'r') as fr_image_list:
        image_name_list = fr_image_list.readlines()
        with open(image2anno_file, 'w') as fw_image2anno:
            for image_name in image_name_list:
                image_name = image_name.strip() 
                image_path = os.path.join(image_root, image_name + '.jpg')
                anno_path = os.path.join(anno_root, image_name + '.txt')   
                fw_image2anno.write(image_path + " " + anno_path + '\n') 

def convert_lmdb(image_root, convert_file, db_file, height, width):
    caffe_py_path = caffe_root + 'python/create_annoset.py'
    if not os.path.isfile(caffe_py_path):
        print("caffe_py_path {} does not exist, try find it in pre-build docker env".format(caffe_py_path))
        caffe_py_path = '/opt/vitis_ai/conda/envs/vitis-ai-caffe/bin/create_annoset.py'
    cmd = 'python ' + caffe_py_path + ' --label-type=%s --anno-type=%s --label-map-file=%s --min-dim=%d --max-dim=%d --resize-width=%d --resize-height=%d --check-label %s %s %s %s ../../data/link_480_360/' %\
          (label_type, anno_type, map_file, min_dim, max_dim, width, height, extra_cmd, image_root, convert_file, db_file)
    print(cmd)
    print(commands.getoutput(cmd))


if __name__ == "__main__":
     
    if not os.path.exists(args.output):
        commands.getoutput('mkdir ' + args.output)
    db_file = os.path.join(args.output, args.train_image_list.split('/')[-1][:-4] + '_lmdb')
    gen_image2anno(args.image_dir, args.anno_dir, args.train_image_list, args.train_image2anno) 
    convert_lmdb('./', args.train_image2anno, db_file, args.height, args.width)
    db_file = os.path.join(args.output, args.val_image_list.split('/')[-1][:-4] + '_lmdb') 
    gen_image2anno(args.image_dir, args.anno_dir, args.val_image_list, args.val_image2anno) 
    convert_lmdb('./', args.val_image2anno, db_file, 0, 0)

    caffe_exec_path = caffe_root + 'build/tools/get_image_size'
    if not os.path.isfile(caffe_exec_path):
        print("caffe_exec_path {} does not exist, try find it in pre-build docker env".format(caffe_exec_path))
        caffe_exec_path = '/opt/vitis_ai/conda/envs/vitis-ai-caffe/bin/get_image_size'
    get_image_size_cmd = caffe_exec_path + ' ' + './' + ' '  + args.val_image2anno + ' ' + '../../data/test_name_size.txt'

    print(get_image_size_cmd)
    print(commands.getoutput(get_image_size_cmd))
