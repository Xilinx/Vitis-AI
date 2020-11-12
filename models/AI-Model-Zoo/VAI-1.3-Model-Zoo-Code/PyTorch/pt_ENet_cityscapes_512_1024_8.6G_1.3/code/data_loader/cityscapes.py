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
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from code.utils.transforms import RandomFlip, Normalize, Resize, Resize_eval, Compose

CITYSCAPE_CLASS_LIST = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
                        'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                        'bicycle', 'background']

num_classes = len(CITYSCAPE_CLASS_LIST) - 1
ignore_label=255

def make_dataset(root, quality, mode):
    assert (quality == 'fine' and mode in ['train', 'val', 'test']) or \
           (quality == 'coarse' and mode in ['train', 'train_extra', 'val'])

    if quality == 'coarse':
        img_dir_name = 'leftImg8bit' if mode == 'train_extra' else 'leftImg8bit_trainvaltest'
        mask_path = os.path.join(root, 'gtCoarse', mode)
        mask_postfix = '_gtCoarse_labelIds.png'
    else:
        img_dir_name = 'leftImg8bit'
        mask_path = os.path.join(root, 'gtFine', mode)
        mask_postfix = '_gtFine_labelIds.png'
    img_path = os.path.join(root, img_dir_name , mode)
    items = []
    categories = os.listdir(img_path)
    for c in categories:
        c_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(img_path, c))]
        for it in c_items:
            item = (os.path.join(img_path, c, it + '_leftImg8bit.png'), os.path.join(mask_path, c, it + mask_postfix))
            items.append(item)
    print('Found {} images numbers: {}'.format(mode, len(items)))
    return items

class CityScapes(data.Dataset):
    def __init__(self, root, quality='fine', mode='train', size=(1024, 512)):

        self.root = root
        self.quality = quality
        self.mode = mode
        self.size=size
        self.items = make_dataset(root, quality, mode)
        if len(self.items) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.train_transforms, self.val_transforms = self.transforms()

        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def transforms(self):
        train_transforms = Compose(
            [
                Resize(size=self.size),
                RandomFlip(),
                Normalize()
            ]
        )
        val_transforms = Compose(
            [
                Resize_eval(size=self.size),
                Normalize()
            ]
        )
        return train_transforms, val_transforms

    def __getitem__(self, index):
        img_path, mask_path = self.items[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        
        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in self.id_to_trainid.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        if self.mode == 'train':
            img, mask = self.train_transforms(img, mask)
        else:
            img, mask = self.val_transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.items)





