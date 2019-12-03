#!/usr/bin/env python
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
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

IMAGE_LIST_FILE = "VOCdevkit/VOC2007/ImageSets/Main/test.txt"
ANNOTATION_PATH = "VOCdevkit/VOC2007/Annotations/"
GT_FILE = "./voc07_gt_file_19c.txt"

def generate_gt_file(image_list_file, annotation_path, gt_file):
    '''
    generate gt_file for map computation with voc2007 test dataset.
    :param image_list_file: file record test images
    :param annotation_path: annotation file path
    :param gt_file: gt files for map computation
    :return: None
    '''
    assert os.path.exists(image_list_file)
    f_image_list = open(image_list_file, 'r')
    lines = f_image_list.readlines()
    f_image_list.close()
    f_gt = open(gt_file, "w")
    for line in lines:
        image_name = line.strip()
        image_annotation_path = annotation_path + image_name + ".xml"
        tree = ET.parse(image_annotation_path)
        root = tree.getroot()
        objects = root.findall('object')
        for obj in objects:
            class_name = obj.find("name").text
            difficult = obj.find("difficult").text
            bndbox_obj = obj.find('bndbox')
            xmin = bndbox_obj.find('xmin').text
            ymin = bndbox_obj.find('ymin').text
            xmax = bndbox_obj.find('xmax').text
            ymax = bndbox_obj.find('ymax').text
            if ((not class_name =='train')  and (not class_name == 'diningtable')):
                f_gt.writelines(" ".join([image_name, class_name, xmin, ymin, xmax, ymax, difficult]) + "\n")
    f_gt.close()

if __name__ == "__main__":
    generate_gt_file(IMAGE_LIST_FILE, ANNOTATION_PATH, GT_FILE)
