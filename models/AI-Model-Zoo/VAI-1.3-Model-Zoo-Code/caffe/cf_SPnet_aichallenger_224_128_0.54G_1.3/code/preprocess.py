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
parser.add_argument('--source', help='source images path')
parser.add_argument('--output', help='output images path')
parser.add_argument('--image', type=bool, default=True, help='whether output the cropped image or not')
args = parser.parse_args()
print(args)
anno_path = args.anno
source_path = args.source
save_path = args.output
output_image = args.image

with open(anno_path, "r") as f:
    anno = json.load(f)
    
a = np.array([])
ai = 0
wc = 0
twc = len(anno)
err = 0
names = ''
max_size = 0
save_id = 0


if not os.path.exists(save_path + 'images'):
    os.makedirs(os.path.join(save_path, 'images'))

fp_128_224 = open(os.path.join(save_path,'label_size_w128_h224.txt'),'w')
fp_224_224 = open(os.path.join(save_path,'label_size_w224_h224.txt'),'w')
fp_256_256 = open(os.path.join(save_path,'label_size_w256_h256.txt'),'w')
print("There are {} images in given source.".format(twc))

for i in range(twc):
    offset = len(a)
    im = cv2.imread(os.path.join(source_path,anno[i]['image_id']+'.jpg'))
    if im is None:        
        err += 1
        print(('Failed(%d):'%err)+os.path.join(source_path,anno[i]['image_id']+'.jpg'))
        continue
    names += os.path.join(source_path,anno[i]['image_id']+'.jpg')+'\n'
    if (i+1) % 10000 == 0:
        print('Processing %d/%d'%(i+1,twc))
    sz = im.shape
    anno_num = len(anno[i]['human_annotations'])
    for j in range(anno_num):
        rect = anno[i]['human_annotations']['human%d'%(j+1)]
        if rect[0] < 0:
            rect[0] = 0
        if rect[1] < 0: 
            rect[1] = 0
        if rect[2] >= sz[1]:
            rect[2] = sz[1] 
        if rect[3] >= sz[0]:
            rect[3] = sz[0] 
        crop_im = im[rect[1]:rect[3],rect[0]:rect[2],:]
        new_sz = crop_im.shape
        if new_sz[0] < 2 or new_sz[1] < 2:
            continue
        kp = anno[i]['keypoint_annotations']['human%d'%(j+1)]
        
        weight = np.ones([14*2])
        true_id = 14
        for k in range(14):
            kp[k*3] -= rect[0]
            kp[k*3+1] -= rect[1]
            
            if kp[k*3] < 0 or kp[k*3+1] < 0 or kp[k*3] >= new_sz[1] or kp[k*3+1] >= new_sz[0]:
                kp[k*3+2] = 3
                kp[k*3] = 0
                kp[k*3+1] = 0
            kp[k*3+2] -= 1
            if kp[k*3+2] == 2:
                weight[k*2] = 0
                weight[k*2+1] = 0 
                true_id -= 1
        if true_id < 7:
            continue
        save_id += 1
        if output_image:
            cv2.imwrite(os.path.join(save_path,'images','%06d.png'%save_id),crop_im)
        fp_128_224.write('%06d.png'%save_id)
        fp_224_224.write('%06d.png'%save_id)
        fp_256_256.write('%06d.png'%save_id)
        ##write location
        for k in range(14):
            fp_128_224.write(' %d %d'%(np.floor(kp[k*3] * (128.0 / new_sz[1])),np.floor(kp[k*3+1]*(224.0/new_sz[0]))))
            fp_224_224.write(' %d %d'%(np.floor(kp[k*3] * (224.0 / new_sz[1])),np.floor(kp[k*3+1]*(224.0/new_sz[0]))))
            fp_256_256.write(' %d %d'%(np.floor(kp[k*3] * (256.0 / new_sz[1])),np.floor(kp[k*3+1]*(256.0/new_sz[0]))))
        ##write weight
        for k in range(14):
            fp_128_224.write(' %d %d'%(weight[k*2],weight[k*2+1]))
            fp_224_224.write(' %d %d'%(weight[k*2],weight[k*2+1]))
            fp_256_256.write(' %d %d'%(weight[k*2],weight[k*2+1]))
        ##write label
        for k in range(14):
            fp_128_224.write(' %d'%(kp[k*3+2]))
            fp_224_224.write(' %d'%(kp[k*3+2]))
            fp_256_256.write(' %d'%(kp[k*3+2]))
        fp_128_224.write('\n')
        fp_224_224.write('\n')
        fp_256_256.write('\n')
            ####
fp_128_224.close()
fp_224_224.close()
fp_256_256.close()

print("finished, it generates {} cropped images, and {} images are wrong".format(save_id, err))
