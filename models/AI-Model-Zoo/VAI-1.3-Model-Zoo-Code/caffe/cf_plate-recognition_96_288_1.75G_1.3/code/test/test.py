#coding=utf-8

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
import cv2
import numpy as np
from argparse import ArgumentParser

if sys.version > '3':
    from imp import reload
reload(sys)
if sys.version < '3':
    sys.setdefaultencoding('utf-8')


USE_GPU = True
font = cv2.FONT_HERSHEY_SIMPLEX

def parse_args():
    parser = ArgumentParser(description="Densebox evaluation on plate detection dataset.")
    parser.add_argument('--caffe-root', '-c', type=str, 
            default='../../../caffe-xilinx/',
            help='path to caffe root')
    parser.add_argument('--image-root', '-d', type=str,
            default='../../data/plate_recognition_val/plate_val/',
            help='path to validation images')
    parser.add_argument('--gt-file', '-gt', type=str,
            default='../../data/plate_recognition_val/plate_val.txt',
            help='file record test image annotations.')
    parser.add_argument('--prototxt', '-p', type=str,
            default='../../float/test.prototxt',
            help='path to caffemodel prototxt')
    parser.add_argument('--weights','-w', type=str,
            default='../../float/trainval.caffemodel',
            help='path to caffemodel weights')
    return parser.parse_args()

args = parse_args()

sys.path.insert(0, args.caffe_root + 'python')
import caffe

class ModelConfig:
    '''
    param config of plate recognition model.
    '''
    def __init__(self):
        self.model_def = args.prototxt
        self.model_weights = args.weights
        self.mean_value = [104, 117, 123]
        self.input_height = 96
        self.input_width = 288
        self.chars_dict = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A","B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X","Y", "Z"]
        self.colors_dict = ["blue", "yellow"]

class PlateRecognitionModel:
    '''
    plate recognition model class.     
    '''

    def __init__(self, config):
        '''
        init PlateRecognitionModel.
        :param config: config class instance.
        :return: None.
        '''
        if USE_GPU:
            caffe.set_mode_gpu()  
            caffe.set_device(0)
        else:
            caffe.set_mode_cpu()
        self.__plate_recognition_net = caffe.Net(config.model_def, config.model_weights, caffe.TEST)
        self.__transformer = self.__transformer(config) 
        self.__input_height = config.input_height
        self.__input_width = config.input_width
        self.__chars_dict = config.chars_dict
        self.__colors_dict = config.colors_dict
 
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

    def predict_single_image(self, image):
        '''
        predict single image.
        :param image: image matrix.
        :return: plate recognition prediction.
        '''
        assert image is not None
        image_resize = cv2.resize(image, (self.__input_width, self.__input_height))     
        self.__plate_recognition_net.blobs['data'].reshape(1, 3, self.__input_height, self.__input_width)
        transformed_image = self.__transformer.preprocess('data', image_resize)
        self.__plate_recognition_net.blobs['data'].data[...] = transformed_image
        self.__plate_recognition_net.forward()
        pred_first_char = self.__plate_recognition_net.blobs['prob'+ str(1)].data[0].flatten().argsort()[-1]
        chars_of_plate = []
        if pred_first_char == 0:
            chars_of_plate = ["*"] 
        else:
            chars_of_plate = [self.__chars_dict[pred_first_char - 1]]
        pred_second_char = self.__plate_recognition_net.blobs['prob'+ str(2)].data[0].flatten().argsort()[-1]
        if pred_second_char == 0:
            chars_of_plate.append("*")
        else:
            chars_of_plate.append(self.__chars_dict[pred_second_char + 41 - 1])  
        for ind in range(3, 8):
            pred_char = self.__plate_recognition_net.blobs['prob'+ str(ind)].data[0].flatten().argsort()[-1] 
            if pred_char == 0:
                chars_of_plate.append("*")
            else:
                pred_char = pred_char - 1
                if pred_char >= 23:
                    pred_char = pred_char + 32
                else:
                    pred_char = pred_char + 31
                chars_of_plate.append(self.__chars_dict[pred_char]) 
        pred_color = self.__colors_dict[self.__plate_recognition_net.blobs['prob8'].data[0].flatten().argsort()[-1]]
        if sys.version < '3':
            pred_chars = "".join(chars_of_plate).decode('string_escape')
        else:
            pred_chars = "".join(chars_of_plate)
        return [pred_chars, pred_color]

    def compute_acc_of_dataset(self, image_root, gt_file):                                     
        '''
        compute acc of dataset
        :param image_root: path of image_root.
        :param gt_file: ground truth file
        :return: None
        '''
        assert os.path.exists(gt_file)
        if sys.version > '3':
            f_gt = open(gt_file, 'rb')
        else:
            f_gt = open(gt_file, 'r')
        gt_lines = f_gt.readlines()
        f_gt.close()
        vaild_image_count = 0
        hit_plate_number_count = 0.0
        hit_color_count = 0.0 
        for line in gt_lines:
            if sys.version > '3':
                line = line.decode()
            line = line.strip()
            items = line.split(" ")
            image_path = os.path.join(image_root, items[0])
            plate_number = items[1]
            plate_color = items[2]
            image = cv2.imread(image_path)
            preidct_res = self.predict_single_image(image)
            if len(preidct_res) > 0:
                vaild_image_count += 1
                if plate_number == preidct_res[0]:
                    hit_plate_number_count += 1
                if plate_color == preidct_res[1]:
                    hit_color_count += 1
        print("plate_number acc: ", hit_plate_number_count / vaild_image_count)
        print("plate_color acc: ", hit_color_count / vaild_image_count)

if __name__ == "__main__":
    model_config = ModelConfig()
    plate_recognition_model = PlateRecognitionModel(model_config)
    plate_recognition_model.compute_acc_of_dataset(args.image_root, args.gt_file)

 
