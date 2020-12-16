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

# PART OF THIS FILE AT ALL TIMES.

import json
import numpy as np
import cv2
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--anno', help='anno file path')
parser.add_argument('--data', help='image data path')
parser.add_argument('--output', default='./', help='output path')
args = parser.parse_args()


IMAGE_PATH = args.data
ANNO_PATH = args.anno

with open(ANNO_PATH, "r") as f:
    anno = json.load(f)

a = np.array([])
ai = 0
wc = 0
twc = len(anno)
err = 0
names = ''
max_size = 0
print(twc)
for i in range(len(anno)):
    offset = len(a)
    im = cv2.imread(os.path.join(IMAGE_PATH, anno[i]['image_id']+'.jpg'))
    if im is None:        
        err += 1
        print(('Failed(%d):'%err)+os.path.join(IMAGE_PATH, anno[i]['image_id']+'.jpg'))
        continue
    names += os.path.join(IMAGE_PATH, anno[i]['image_id']+'.jpg') + '\n'
    a.resize(len(a)+52)
    sz = im.shape
    max_size = max(max_size,max(sz))
    idx = 0
    a[offset+0] = sz[1]
    a[offset+1] = sz[0]
    a[offset+2] = len(anno[i]['human_annotations']) - 1
    a[offset+3] = 1
    a[offset+4] = wc+1
    a[offset+5] = wc
    a[offset+6] = twc
    rect = anno[i]['human_annotations']['human1']
    a[offset+7] = (rect[2]+rect[0])/2.0
    a[offset+8] = (rect[3]+rect[1])/2.0
    a[offset+9] = (rect[3]-rect[1]) / 368.0
    a[offset+10:offset+52] = anno[i]['keypoint_annotations']['human1']
    for j in range(int(a[offset+2])):
        a.resize(len(a)+45)
        rect = anno[i]['human_annotations']['human%d'%(j+2)]
        a[offset+52+j*45+0] = (rect[2]+rect[0])/2.0
        a[offset+52+j*45+1] = (rect[3]+rect[1])/2.0
        a[offset+52+j*45+2] = (rect[3]-rect[1])/368.0
        a[offset+52+j*45+3:offset+52+j*45+45] = anno[i]['keypoint_annotations']['human%d'%(j+2)]


if not os.path.exists(args.output):
    os.makedirs(args.output)

path_path = os.path.join(args.output, 'path.txt')
meta_path = os.path.join(args.output, 'meta.bin')
fp = open(path_path,'w')
fp.writelines(names)
fp.close()
fp = open(meta_path,'wb')
b = a.astype(dtype='float32')
fp.write(b.data)
fp.close()
