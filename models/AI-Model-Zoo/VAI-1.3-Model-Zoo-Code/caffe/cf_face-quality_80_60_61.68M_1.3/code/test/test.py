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

import cv2
import numpy as np
import os
import math
import sys

import argparse

parser = argparse.ArgumentParser(description='Face Recognition Test')
parser.add_argument('--root_imgs', type=str, default='../data/imgs',
                    help='your testset path')
parser.add_argument('--caffe_path', type=str, default='../../../../../caffe-xilinx',
                    help='your caffe path')
parser.add_argument('--model', type=str, default='../float/trainval.caffemodel',
                    help='caffe model for test')
parser.add_argument('--prototxt', type=str, default='../float/test.prototxt',
                    help='caffe model prototxt for test')
parser.add_argument('--testset_list', type=str, default='../data/face_quality/test_list.txt',
                    help='ID list file')
parser.add_argument('--gpu', type=int, default=0,
                    help='gpu card for test')
parser.add_argument('--points_output_name', type=str, default='Addmm_2',
                    help='points layer name')
parser.add_argument('--quality_output_name', type=str, default='Addmm_4',
                    help='quality layer name')
args = parser.parse_args()
sys.path.insert(0,args.caffe_path+'/python')
sys.path.insert(0,os.path.join(args.caffe_path,'python/caffe'))

import caffe

def eval_net(net, annoinfo):

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.array([127.5, 127.5, 127.5]))
    transformer.set_input_scale('data', 0.0078125)  
    transformer.set_channel_swap('data', (2,1,0))
    points_distance = 0.
    quality_distance = 0.
    points_total = 0
    quality_total = 0
    for i in range(len(annoinfo)):
             
        img = cv2.imread(annoinfo[i][0])
        img_tmp = cv2.resize(img, (60, 80))
        assert img is not None
        image = transformer.preprocess('data', img_tmp)
        net.blobs['data'].data[...] = np.expand_dims(image, 0)
        out = net.forward()
        points_outputs = net.blobs[args.points_output_name].data[0]
        quality_outputs = net.blobs[args.quality_output_name].data[0][0]
        points_labels = annoinfo[i][1] 
        quality_labels = annoinfo[i][2] 
        flags = annoinfo[i][3]
        # 80x60
        points_outputs[0:5] = points_outputs[0:5] * img.shape[1] / 60.0
        points_outputs[5:10] = points_outputs[5:10]*img.shape[0] / 80.0
        #points_outputs = points_outputs * 6./5.
        if flags == 0:
            points_distance += np.sum(np.abs(points_outputs - points_labels))
            points_total += 1
        elif flags == 1:
            quality_distance += np.sum(np.abs(quality_outputs -quality_labels))
            quality_total += 1
    points_total = max(points_total, 1)
    quality_total = max(quality_total, 1)
    return points_distance / float((10*points_total)), quality_distance / float(quality_total)

def get_anno(root, listFile):
    listname = listFile
    imgnameList = open(listname, 'r')
    lines = imgnameList.readlines()

    annotation_list = []
    flag = 0
    for line in lines:
        line = line.strip('\n').split(' ')
        image_name = os.path.join(root, line[0])
        #assert os.path.exists(image_name)
        if len(line[1:]) == 10:
            points = np.array(line[1:], dtype = np.float)
            quality = np.array([0.])
            flag = 0
        elif len(line[1:]) == 1:
            points = np.zeros(10)
            quality = np.array(line[1:], dtype = np.float)
            flag = 1
        else:
            assert False, 'wrong label {}'.format(line)
        points = points.tolist()
        quality = quality.tolist()
        annotation_list.append((image_name, points, quality, flag))
    return annotation_list
if __name__ == '__main__':
    deploy = args.prototxt
    model = args.model
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)
    net = caffe.Net(deploy, model, caffe.TEST)

    root = args.root_imgs
    listFile = args.testset_list

    anno_info = get_anno(root, listFile)
    l1_loss_points, l1_loss_quality = eval_net(net, anno_info)
    print(">>>>>> face quality l1_loss is: %.4f"%(l1_loss_quality))
    
	
