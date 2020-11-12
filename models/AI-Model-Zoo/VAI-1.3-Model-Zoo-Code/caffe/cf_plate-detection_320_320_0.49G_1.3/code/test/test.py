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
import math
import numpy as np
from argparse import ArgumentParser
if sys.version > '3':
    from imp import reload
reload(sys)
if sys.version < '3':
    sys.setdefaultencoding('utf-8')

USE_GPU = True
#font = cv2.FONT_HERSHEY_SIMPLEX

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="Densebox evaluation on plate detection dataset.")
    parser.add_argument('--caffe-root', '-c', type=str, 
            default='../../../caffe-xilinx/',
            help='path to caffe root')
    parser.add_argument('--image-root', '-d', type=str,
            default='../../data/plate_detection_val/val_plate/',
            help='path to validation images')
    parser.add_argument('--gt-file', '-gt', type=str,
            default='../../data/plate_detection_val/landmark_gt.txt',
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
from comput_iou import comput_iou

class ModelConfig:
    '''
    param config of plate detection model.
    '''
    def __init__(self):
        self.model_def = args.prototxt
        self.model_weights = args.weights 
        self.mean_value = [128, 128, 128]
        self.input_height = 320
        self.input_width = 320
        self.conf_threshold = 0.2
        self.iou_threshold = 0.5

class PlateDetectionModel:
    '''
    plate detection model class.     
    '''

    def __init__(self, config):
        '''
        init PlateDetectionModel.
        :param config: config class instance.
        :return: None.
        '''
        if USE_GPU:
            caffe.set_mode_gpu()  
            caffe.set_device(1)
        else:
            caffe.set_mode_cpu()
        self.plate_detection_net = caffe.Net(config.model_def, config.model_weights, caffe.TEST)
        self.transformer = self.__transformer(config) 
        self.conf_threshold = config.conf_threshold
        self.iou_threshold = config.iou_threshold
        self.input_height = config.input_height
        self.input_width = config.input_width
 
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

    def detect_single_image(self, image):
        '''
        detect single image.
        :param image: image matrix.
        :return: None or 4 point cordinates.
        '''
        assert image is not None
        height, width = image.shape[0:2]
        height_ratio = float(height) / self.input_height
        width_ratio = float(width) / self.input_width
        image_resize = cv2.resize(image, (self.input_width, self.input_height))     
        self.plate_detection_net.blobs['data'].reshape(1, 3, self.input_height, self.input_width)
        transformed_image = self.transformer.preprocess('data', image_resize)
        self.plate_detection_net.blobs['data'].data[...] = transformed_image
        plate_detection_output = self.plate_detection_net.forward()
        prob = plate_detection_output['pixel-conv'][0, 64:128, ...]    
        prob_ne = plate_detection_output['pixel-conv'][0, 0:64, ...]   
        bb = plate_detection_output['bb-output'][0, ...]              
        c_max = 0                                             
        hei_max = 0                                           
        wid_max = 0                                           
        maxvalue = prob[0, 0, 0]                              
        for c in range(prob.shape[0]):                                                                        
            for hei in range(prob.shape[1]):                                                                  
                for wid in range(prob.shape[2]):                                                              
                    if (prob[c, hei, wid] > maxvalue):                                                        
                        c_max = c                                                                             
                        hei_max = hei                                                                         
                        wid_max = wid                                                                         
                        maxvalue = prob[c, hei, wid]                                                                      
        ne_maxvalue = prob_ne[c_max, hei_max, wid_max]                                                        
        prob_score = math.exp(maxvalue) /(math.exp(maxvalue) + math.exp(ne_maxvalue))                         
        if prob_score > self.conf_threshold:                                                                  
            r_ind = c_max / 8                                                                                 
            c_ind = c_max % 8                                                                                 
            row = 8 * hei_max + r_ind                                                                         
            col = 8 * wid_max + c_ind                                                                        
            x_1 = int((col * 4 + bb[c_max, hei_max, wid_max]) * width_ratio)                 
            y_1 = int((row * 4 + bb[64 + c_max, hei_max, wid_max]) * height_ratio)            
            x_2 = int((col * 4 + bb[128 + c_max, hei_max, wid_max]) * width_ratio)                
            y_2 = int((row * 4 + bb[192 + c_max, hei_max, wid_max]) * height_ratio)                          
            x_3 = int((col * 4 + bb[256 + c_max, hei_max, wid_max]) * width_ratio)                           
            y_3 = int((row * 4 + bb[320 + c_max, hei_max, wid_max]) * height_ratio)                           
            x_4 = int((col * 4 + bb[384 + c_max, hei_max, wid_max]) * width_ratio)                            
            y_4 = int((row * 4 + bb[448 + c_max, hei_max, wid_max]) * height_ratio)                                       
            return [x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4, maxvalue] 
        else:                                                                                                             
            return None

    def compute_acc_of_dataset(self, image_root, gt_file):                                     
        '''
        compute acc of dataset
        :param image_root: path of image_root.
        :param gt_file: ground truth file
        :return: None
        '''
        if sys.version > '3':
            f_gt = open(gt_file, 'rb')
        else:
            f_gt = open(gt_file, 'r')
        gt_lines = f_gt.readlines()
        f_gt.close()
        sample_size = int(len(gt_lines) / 3)
        vaild_count_image = 0
        hit = 0.0
        for i in range(sample_size):
            if sys.version > '3':
                gt_lines[3*i] = gt_lines[3*i].decode()
                gt_lines[3 * i + 2] = gt_lines[3 * i + 2].decode()
            image_name = gt_lines[3 * i].strip()
            landmark_cords = gt_lines[3 * i + 2].strip().split(" ") 
            gt_landmark = []
            for cord in landmark_cords:
                gt_landmark.append(int(cord))
            image = cv2.imread(os.path.join(image_root, image_name))
            if image is None:
                continue
            vaild_count_image += 1
            pred_landmark = self.detect_single_image(image)
            if pred_landmark is None:
                continue
            pred_landmark = pred_landmark[0:8]
            iou = comput_iou(image.shape[0], image.shape[1], gt_landmark, pred_landmark)
            print (iou)
            if iou >= self.iou_threshold:
                hit += 1.0
        print ("acc: ", hit / vaild_count_image)
   
    def dispay_detection(self, image):
        vehicle_plate_pred = self.detect_single_image(image)
        if vehicle_plate_pred is not None:      
            plate_xmin = min(vehicle_plate_pred[0], vehicle_plate_pred[6])
            plate_ymin = min(vehicle_plate_pred[1], vehicle_plate_pred[3])
            plate_xmax = max(vehicle_plate_pred[2], vehicle_plate_pred[4]) 
            plate_ymax = max(vehicle_plate_pred[5], vehicle_plate_pred[7]) 
            plate_width = plate_xmax - plate_xmin + 1
            plate_height = plate_ymax - plate_ymin + 1
            cv2.rectangle(image, (plate_xmin, plate_ymin), (plate_xmax, plate_ymax), (0, 0, 255), 1)
        return image

if __name__ == "__main__":
    model_config = ModelConfig()
    plate_detect_model = PlateDetectionModel(model_config)
    plate_detect_model.compute_acc_of_dataset(args.image_root, args.gt_file)

