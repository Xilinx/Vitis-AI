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

DET_CLASSES = (  # always index 0
    'car', 'sign', 'person')
'''
DET_ROOT = "/scratch/workspace/data/multi_task_det5_seg16/detection/Waymo_bdd_txt"
'''

CAR_INCLUDE = ('car', 'truck', 'bus', 'vehicle')
MOTOR_INCLUDE = ('motor', 'bike')
PERSON_INCLUDE = ('person', 'rider', 'cyclist')


class DetAnnotationTransform(object):

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = dict(zip(DET_CLASSES, range(len(DET_CLASSES))))

    def __call__(self, target, width, height):
        res = []
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
                continue
            bndbox = []
            bndbox.append(float(component[2]) / width)
            bndbox.append(float(component[3]) / height)
            bndbox.append(float(component[4]) / width)
            bndbox.append(float(component[5]) / height)

            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox] 
        target.close()
	
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class Detection(data.Dataset):
    #def __init__(self, root='/home/caiyi/data/bdd100k/',
    def __init__(self, DET_ROOT,image_sets='train',transform=None, target_transform=DetAnnotationTransform()):
        self.root = DET_ROOT
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self._annopath = osp.join('%s', 'detection', '%s.txt')
        self._imgpath = osp.join('%s', 'images', '%s.jpg')
        self.ids = list()
        rootpath = osp.join(self.root, self.image_set)
        for line in open(osp.join(rootpath, self.image_set + '.txt')):
            self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = open(self._annopath % img_id)
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        
        if self.transform is not None:
            target = np.array(target)
            if len(target.shape) != 2:
                print("img_id: {}, target.shape: ()".format(self._annopath % img_id, target.shape))
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        img_id = self.ids[index]
        anno = open(self._annopath % img_id)
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
        

