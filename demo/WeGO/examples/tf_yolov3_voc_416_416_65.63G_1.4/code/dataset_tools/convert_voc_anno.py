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
import argparse
import xml.etree.ElementTree as ET

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def voc_xml2lines(xml_path):
    tree = ET.parse(open(xml_path))
    root = tree.getroot()

    lines = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        # if cls not in classes or int(difficult)==1:
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
             int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        lines.append(classes[cls_id] + " " + " ".join([str(a)
                     for a in b]) + " " + difficult)
    return lines


def convert_voc_anno(anno_dir, list_file, dst_file):
    with open(list_file) as reader:
        ids = reader.readlines()
    with open(dst_file, 'w') as writer:
        for idx in ids:
            idx = idx.strip()
            xml_path = os.path.join(anno_dir, idx + ".xml")
            lines = voc_xml2lines(xml_path)
            for line in lines:
                writer.write(idx + " " + line + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Script to convert annotation from .xml to .txt')
    parser.add_argument('-anno_dir', type=str)
    parser.add_argument('-list_file', type=str)
    parser.add_argument('-dst_file', type=str)
    args = parser.parse_args()

    convert_voc_anno(args.anno_dir, args.list_file, args.dst_file)
