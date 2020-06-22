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

#coding=utf-8
import os
import sys
import cv2
from argparse import ArgumentParser
import numpy as np

def parse_args():
    parser = ArgumentParser(description="Moblinet-v2 evaluation on Imagenet")
    parser.add_argument('--caffe-root', '-c', type=str, 
            default='../../../caffe-xilinx/',
            help='path to caffe root')
    parser.add_argument('--data-root', '-d', type=str,
            default="../../data/Imagenet/val_dataset/" ,
            help='path to validation images')
    parser.add_argument('--image-list', '-i', type=str,
            default="../../data/Imagenet/val.txt",
            help='path to image list file')
    parser.add_argument('--prototxt', '-p', type=str,
            default='../../float/test.prototxt',
            help='path to caffemodel prototxt')
    parser.add_argument('--weights','-w', type=str,
            default='../../float/trainval.caffemodel',
            help='path to caffemodel weights')
    parser.add_argument('--labelmap', '-l', type=str,
            default='dataset_config/labelmap.txt',
            help='path to imagenet labelmap file')
    return parser.parse_args()

args = parse_args()

# repalce user's caffe path
caffe_root = args.caffe_root
sys.path.insert(0, caffe_root + 'python')
import caffe

USE_GPU = True

class ModelConfig:
    '''
    param config of cnn recognition model.
    '''
    def __init__(self):
        self.model_def = args.prototxt
        self.model_weights = args.weights 
        self.mean_value = [104, 117, 123]
        self.scale = 0.00390625
        self.short_side_resize = 256
        self.input_height = 224
        self.input_width = 224
        self.labelmap = args.labelmap 

class CnnRecognitionModel:
    '''
    cnn recognition model class.     
    '''

    def __init__(self, config):
        '''
        init VehicleAttributionRecognitionModel.
        :param config: config class instance.
        :return: None.
        '''
        if USE_GPU:
            caffe.set_device(0)
            caffe.set_mode_gpu()  
        else:
            caffe.set_mode_cpu()
        self.__cnn_recognition_net = caffe.Net(config.model_def, config.model_weights, caffe.TEST)
        self.__transformer = self.__transformer(config) 
        self.__input_height = config.input_height
        self.__input_width = config.input_width
        self.__short_side_resize = config.short_side_resize
        self.__scale = config.scale

    def __transformer(self, config):
        '''
        consturct data transformer with config class instance.
        :param config: config class instance.
        :return: transformer.
        ''' 
        transformer = caffe.io.Transformer({'data': (1, 3, config.input_height, config.input_width)})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array(config.mean_value))
  
        return transformer
    
    def __resize_shortest_edge(self, image, size):
        H, W = image.shape[:2]
        if H >= W:
            nW = size
            nH = int(float(H)/W * size)
        else:
            nH = size
            nW = int(float(W)/H * size)
        return cv2.resize(image,(nW,nH))

    def __central_crop(self, image, crop_height, crop_width):
        image_height = image.shape[0]
        image_width = image.shape[1]
        offset_height = (image_height - crop_height) // 2
        offset_width = (image_width - crop_width) // 2
        return image[offset_height:offset_height + crop_height, offset_width:
                     offset_width + crop_width, :]

    def predict_single_image(self, image):
        '''
        predict single image.
        :param image: image matrix.
        :return: index of sort prob.
        '''
        assert image is not None, "Image cannot be none!"
        image_resize = self.__resize_shortest_edge(image, self.__short_side_resize)
        image_resize = self.__central_crop(image_resize, self.__input_height, self.__input_width)
        image_resize = np.array(image_resize, dtype=np.float32)
        #print(image_resize[0:1])   
        transform_image = self.__transformer.preprocess('data', image_resize)
        transform_image = transform_image * self.__scale
        self.__cnn_recognition_net.blobs['data'].reshape(1, 3, self.__input_height, self.__input_width)
        self.__cnn_recognition_net.blobs['data'].data[0, ...] = transform_image
        prediction = self.__cnn_recognition_net.forward()
        prob_data = self.__cnn_recognition_net.blobs['prob'].data[0]
        return np.argsort(prob_data)[::-1]

    def compute_acc_of_dataset(self, image_root, gt_file):                                     
        '''
        compute acc of dataset
        :param image_root: path of image_root.
        :param gt_file: ground truth file
        :return: None
        '''
        assert os.path.exists(image_root)
        assert os.path.exists(gt_file)
        f_gt = open(gt_file, 'r')
        gt_lines = f_gt.readlines()
        f_gt.close()
        image_count = len(gt_lines)
        count_top_1 = 0
        count_top_5 = 0
        count = 0
        for line in gt_lines:
            count += 1
            if count % 50 == 0 or count == image_count:
                print("preprocess %d / %d"%(count, image_count))
            line = line.strip()
            items = line.split(" ")
            image_path = os.path.join(image_root, items[0])
            label = int(items[1])
            image = cv2.imread(image_path)
            preidct = self.predict_single_image(image)
            if label == preidct[0]:
                count_top_1 += 1
            if label in preidct[0:5]:
                count_top_5 += 1
        print ("acc top_1: ", float(count_top_1) / image_count)
        print ("acc top_5: ", float(count_top_5) / image_count)

if __name__ == "__main__":
    model_config = ModelConfig()
    cnn_recognition_model = CnnRecognitionModel(model_config)
    cnn_recognition_model.compute_acc_of_dataset(args.data_root, args.image_list)
