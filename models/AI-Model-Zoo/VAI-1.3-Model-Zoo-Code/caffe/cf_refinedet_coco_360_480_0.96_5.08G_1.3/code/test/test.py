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


#coding=utf_8
import os
import sys
from argparse import ArgumentParser
import numpy as np
import cv2

def parse_args():
    parser = ArgumentParser(description="ssd evaluation on coco2014-person")
    parser.add_argument('--caffe-root', '-c', type=str, 
            default='../../../caffe-xilinx/',
            help='path to caffe root')
    parser.add_argument('--data-root', '-d', type=str,
            default='../../data/coco2014/Images/',
            help='path to validation images')
    parser.add_argument('--image-list', '-i', type=str,
            default='../../data/coco2014/val2014.txt',
            help='path to image list file')
    parser.add_argument('--anno-root', '-a', type=str,
            default='../../data/coco2014/Annotations',
            help='path to annotations')
    parser.add_argument('--gt-file', '-gt', type=str,
            default='../../data/gt_file.txt',
            help='file record test image annotations.')
    parser.add_argument('--det-file', '-det', type=str,
            default='../../data/det_file.txt',
            help='file record test image detection results.')
    parser.add_argument('--prototxt', '-p', type=str,
            default='../../float/test.prototxt',
            help='path to caffemodel prototxt')
    parser.add_argument('--weights','-w', type=str,
            default='../../float/trainval.caffemodel',
            help='path to caffemodel weights')
    parser.add_argument('--labelmap', '-l', type=str,
            default='../../data/labelmap.prototxt',
            help='path to labelmap file')
    parser.add_argument('--eval-script-path', '-e', type=str,
            default='./evaluation_py2.py',
            help='path to eval map script')
    return parser.parse_args()

args = parse_args()

caffe_root = args.caffe_root
sys.path.insert(0, caffe_root + 'python')
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2

font = cv2.FONT_HERSHEY_SIMPLEX
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

class ModelConfig:
    '''
    param config of cnn recognition model.
    '''
    def __init__(self):
        self.model_def = args.prototxt
        self.model_weights = args.weights 
        self.mean_value = [104, 117, 123]
        self.scale = 1
        self.input_height = 360
        self.input_width = 480
        self.labelmap = args.labelmap 
        self.use_gpu = True
        self.conf_threshold = 0.01
        self.dispy_threshold = 0.3

def get_labelmap(labelmap_file):
    with open(labelmap_file, 'r') as fr_labelmap:
        labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(fr_labelmap.read()), labelmap)
    return labelmap

def get_labelname(labelmap, labels):
    '''
    get labelname from lablemap and lables
    :param labelmap: map of label to name
    :param labels: label list
    :return: labelname list 
    '''
    num_labels = len(labelmap.item)
    labelnames = []
    if not isinstance(labels, list):
        labels = [labels]
    for label in labels:
        found = False
        for i in range(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

def generate_gt_file(anno_root, image_list, gt_file):
    with open(gt_file, 'w') as fw_gt:
         with open(image_list, 'r') as fr_image:
             imagename_list = fr_image.readlines()
             for imagename in imagename_list:
                 anno_path = os.path.join(anno_root, imagename.strip() + '.txt')
                 fr_anno = open(anno_path, 'r')
                 anno_lines = fr_anno.readlines()
                 fr_anno.close()
                 for anno in anno_lines:
                    fw_gt.writelines(anno)                          

class SSDModel:
    '''
    SSD model class.     
    '''

    def __init__(self, config):
        '''
        init SSDModel.
        :param config: config class instance.
        :return: None.
        '''
        if config.use_gpu:
            caffe.set_mode_gpu()  
            caffe.set_device(0)
        else:
            caffe.set_mode_cpu()
        self.__ssd_net = caffe.Net(config.model_def, config.model_weights, caffe.TEST)
        self.__transformer = self.__transformer(config) 
        self.__input_height = config.input_height
        self.__input_width = config.input_width
        self.__scale = config.scale
        self.__labelmap = get_labelmap(config.labelmap)
        self.__conf_threshold = config.conf_threshold
        self.__dispy_threshold = config.dispy_threshold


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
    

    def detect_single_image(self, image_path, is_display=False, image_name=None, fp=None):
        '''
        detect single image
        :param image: image matrix
        :param image_name: image name
        :param fp: handle of results record file
        :return: None
        '''
        assert os.path.exists(image_path), 'image path doesnot exists!'
        image = cv2.imread(image_path)
        assert image is not None, 'image cannot be None!' 
        height, width = image.shape[0:2]
        image_resize = cv2.resize(image, (self.__input_width, self.__input_height))
        self.__ssd_net.blobs['data'].reshape(1, 3, self.__input_height, self.__input_width)
        transformed_image = self.__transformer.preprocess('data', image_resize)
        self.__ssd_net.blobs['data'].data[...] = transformed_image
        detections = self.__ssd_net.forward()['detection_out']
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]
        top_indices = [ind for ind, conf in enumerate(det_conf) if conf >= self.__conf_threshold]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(self.__labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]
        if top_conf.size > 0:    
            det_num = len(top_labels)
            for ind in range(det_num):
                f_xmin = width * top_xmin[ind]
                int_xmin = int(f_xmin)
                f_ymin = height * top_ymin[ind]
                int_ymin = int(f_ymin)
                f_xmax = width * top_xmax[ind]
                int_xmax = int(f_xmax)
                f_ymax = height * top_ymax[ind]
                int_ymax = int(f_ymax)
                if is_display and top_conf[ind] >= self.__dispy_threshold:
                    color = colors_tableau[int(top_label_indices[ind])]
                    cv2.rectangle(image, (int_xmin, int_ymin), (int_xmax, int_ymax), color, 1)
                    cv2.putText(image, str(top_labels[ind]), (int_xmin, int_ymin + 10), font, 0.4, (255, 255, 255), 1) 
                    cv2.putText(image, str(top_conf[ind]), (int_xmin, int_ymin - 10), font, 0.4, (255, 255, 255), 1)
                if image_name is not None and fp is not None:
                    fp.writelines(image_name + " " + str(top_labels[ind]) + " " + str(top_conf[ind]) + " " \
                                  + str(f_xmin) + " " + str(f_ymin) + " " + str(f_xmax) + " " + str(f_ymax) + "\n")  
        if is_display:
            if not os.path.exists('output'):
                os.system('mkdir output')
            cv2.imwrite("output/res_output.jpg", image)     

    def compute_map_of_datset(self, image_root, image_list_file, eval_script_path, det_file, gt_file):
        '''
        compute map of dataset
        :param image_list_file:
        :param det_res_file:
        :param gt_file:
        :return: None
        '''   
        assert os.path.exists(image_list_file), 'image_list_file doesnot exists!'
        fr_image_list = open(image_list_file, 'r')
        imagename_list = fr_image_list.readlines()
        fr_image_list.close()
        fw_det = open(det_file, 'w')
        for imagename in imagename_list:
            image_name = imagename.strip()
            image_path = image_root + image_name + '.jpg'
            self.detect_single_image(image_path, image_name=image_name, fp=fw_det)
        fw_det.close()
        os.system("python " +  eval_script_path + " -mode detection " +  \
                  " -gt_file " + gt_file  +  " -result_file " + det_file \
                  + " -detection_use_07_metric True")

if __name__ == "__main__":
    generate_gt_file(args.anno_root, args.image_list, args.gt_file)
    model_config = ModelConfig()
    ssd_model = SSDModel(model_config)
    ssd_model.compute_map_of_datset(args.data_root, args.image_list, args.eval_script_path, args.det_file, args.gt_file)
       
