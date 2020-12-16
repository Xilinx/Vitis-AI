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

from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import math
import pdb

DET_CLASSES = (  # always index 0
    'car', 'cycle', 'person')
'''
ORI_ROOT = osp.join("/scratch/workspace/data/multi_task_det5_seg16/kitti")
'''
CAR_INCLUDE = ('car', 'truck', 'bus','van', 'tram')
MOTOR_INCLUDE = ('motor', 'bike','cyclist')
PERSON_INCLUDE = ('person', 'rider','pedestrian','person_sitting')


class DetOriAnnotationTransform(object):

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = dict(zip(DET_CLASSES, range(len(DET_CLASSES))))

    def __call__(self, target, width, height):
        res = []
        oris = []
        for line in target:
            component = line.strip().split(' ')
            if float(component[2]) >= float(component[4]) or float(component[3]) >= float(component[5]):
                continue
            name = component[1].lower()
            if name in CAR_INCLUDE:
                name = 'car'
            elif name in MOTOR_INCLUDE:
                name = 'cycle'
            elif name in PERSON_INCLUDE:
                name = 'person'
            else:
                print(name)
                continue
            bndbox = []
            bndbox.append(float(component[2]) / width)
            bndbox.append(float(component[3]) / height)
            bndbox.append(float(component[4]) / width)
            bndbox.append(float(component[5]) / height)

            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]
            
            ori = []
            ori.append(math.sin(float(component[6]) / 180 * math.pi))
            ori.append(math.cos(float(component[6]) / 180 * math.pi))
            oris += [ori]
        target.close()

        return res, oris  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class OriDetection(data.Dataset):
    #def __init__(self, root='/home/caiyi/data/bdd100k/',
    def __init__(self, ORI_ROOT,image_sets='train',transform=None, target_transform=DetOriAnnotationTransform()):
        self.root = ORI_ROOT
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self._annopath = osp.join('%s', 'labels', '%s.txt')
        self._imgpath = osp.join('%s', 'images', '%s.png')
        self.ids = list()
        rootpath = osp.join(self.root, self.image_set)
        for line in open(osp.join(rootpath, self.image_set + '.txt')):
            self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w, oris = self.pull_item(index)

        return im, gt, oris

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = open(self._annopath % img_id)
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target, oris = self.target_transform(target, width, height)
        
        if self.transform is not None:
            target = np.array(target)
            if len(target.shape) != 2:
                print("img_id: {}, target.shape: ()".format(self._annopath % img_id, target.shape))
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width, oris

    def pull_image(self, index):
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        img_id = self.ids[index]
        anno = open(self._annopath % img_id)
        gt, oris = self.target_transform(anno, 1, 1)
        return img_id[1], gt, oris

    def pull_tensor(self, index):
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
        

