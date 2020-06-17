# (c) Copyright 2019 Xilinx, Inc. All rights reserved.
#
# This file contains confidential and proprietary information
# of Xilinx, Inc. and is protected under U.S. and
# international copyright and other intellectual property
# laws.
#
# DISCLAIMER
# This disclaimer is not a license and does not grant any
# rights to the materials distributed herewith. Except as
# otherwise provided in a valid license issued to you by
# Xilinx, and to the maximum extent permitted by applicable
# law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
# WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
# AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
# BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
# INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
# (2) Xilinx shall not be liable (whether in contract or tort,
# including negligence, or under any other theory of
# liability) for any loss or damage of any kind or nature
# related to, arising under or in connection with these
# materials, including for any direct, or any indirect,
# special, incidental, or consequential loss or damage
# (including loss of data, profits, goodwill, or any type of
# loss or damage suffered as a result of any action brought
# by a third party) even if such damage or loss was
# reasonably foreseeable or Xilinx had been advised of the
# possibility of the same.
#
# CRITICAL APPLICATIONS
# Xilinx products are not designed or intended to be fail-
# safe, or for use in any application requiring fail-safe
# performance, such as life-support or safety devices or
# systems, Class III medical devices, nuclear facilities,
# applications related to the deployment of airbags, or any
# other applications that could lead to death, personal
# injury, or severe property or environmental damage
# (individually and collectively, "Critical
# Applications"). Customer assumes the sole risk and
# liability of any use of Xilinx products in Critical
# Applications, subject only to applicable laws and
# regulations governing limitations on product liability.
#
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
# PART OF THIS FILE AT ALL TIMES.

# Copyright (c) 2019, Xilinx, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import json
import argparse
import shutil

arg_parser = argparse.ArgumentParser(description="This is a script to convert coco anntations to voc-like annotations.")
arg_parser.add_argument('-ti', '--train_images', type=str, default="./coco2014/train2014", help='where to put coco2014 train images.')
arg_parser.add_argument('-vi', '--val_images', type=str, default='./coco2014/val2014', help='where to put coco2014 val images.')
arg_parser.add_argument('-ta', '--train_anno', type=str, default='./coco2014/instances_train2014.json', help='where to put cooc2014 train set annotations.')
arg_parser.add_argument('-va', '--val_anno', type=str, default='./coco2014/instances_val2014.json', help='where to put coco2014 val set annotations')
arg_parser.add_argument('-tlf', '--tran_list_file', type=str, default='./coco2014/train2014.txt', help='image list for training')
arg_parser.add_argument('-vlf', '--val_list_file', type=str, default='./coco2014/val2014.txt', help='image list for evalution.')
arg_parser.add_argument('-ai', '--all_images', type=str, default='./coco2014/Images', help='where to put all images.')
arg_parser.add_argument('-aa', '--all_anno', type=str, default='./coco2014/Annotations', help='where to put all annotations.')
args = arg_parser.parse_args()

'''How to organize coco dataset folder:
 inputs:
 coco2014/
       |->train2014/
       |->val2014/
       |->instances_train2014.json
       |->instances_val2014.json

outputs:
 coco2014/
       |->Annotations/
       |->Images/
       |->train2014.txt
       |->val2014.txt
'''

def convert_images_coco2voc(args):
    assert os.path.exists(args.train_images)
    assert os.path.exists(args.val_images)
    os.system('mv ' + args.train_images + ' ' + args.all_images)
    imagename_list = os.listdir(args.val_images)
    for imagename in imagename_list:
       shutil.copy(os.path.join(args.val_images, imagename), args.all_images)  
    os.system('rm -r ' + args.val_images)

def generate_cid_name(json_object):
    id2name_dict = {}
    for ind, category_info in enumerate(json_object['categories']):
        id2name_dict[category_info['id']] = category_info['name']
    return id2name_dict

def generate_image_dict(json_object): 
    id2image_dict = {}
    for ind, image_info in enumerate(json_object['images']):
        id2image_dict[image_info['id']] = image_info['file_name']
    return id2image_dict

def generate_annotation_files(json_object, annotation_path, id2image_dict, id2name, image_list_file):
    if not os.path.exists(annotation_path):
        os.mkdir(annotation_path)
    f_image = open(image_list_file, 'w')
    all_images_name = []
    for ind, anno_info in enumerate(json_object['annotations']):
        print('preprocess: {}'.format(ind))
        category_id = anno_info['category_id']
        cls_name = id2name[category_id]
        if cls_name != "person":
            continue       
        image_id = anno_info['image_id']
        image_name = id2image_dict[image_id]
        bbox = anno_info['bbox']
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[3] + bbox[1]
        bbox_str = ' '.join([str(int(x)) for x in bbox])
        with open(os.path.join(annotation_path, image_name.split('.')[0] + '.txt'), 'a') as f_anno:
            f_anno.writelines(image_name.split('.')[0] + " " + cls_name + " " + bbox_str + "\n")
        if image_name not in all_images_name:
            all_images_name.append(image_name)
    for image_name in all_images_name:  
        f_image.writelines(image_name.split('.')[0] + "\n")
    f_image.close() 
                    
def convert_anno_coco2voc(coco_anno_file, image_list_file, all_anno_path):
    with open(coco_anno_file, 'r') as f_ann:
         line = f_ann.readlines()
    json_object = json.loads(line[0])
    id2name = generate_cid_name(json_object)
    id2image_dict = generate_image_dict(json_object)
    generate_annotation_files(json_object, all_anno_path, id2image_dict, id2name, image_list_file)

def convert_anno_all(args):
    convert_anno_coco2voc(args.train_anno, args.tran_list_file, args.all_anno)
    convert_anno_coco2voc(args.val_anno, args.val_list_file, args.all_anno)

if __name__  == "__main__":
    convert_anno_all(args)
    convert_images_coco2voc(args)
