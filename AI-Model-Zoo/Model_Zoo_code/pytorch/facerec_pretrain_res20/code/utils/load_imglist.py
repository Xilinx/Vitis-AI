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

import torch
import torch.utils.data as data
from torch.utils.data.sampler import BatchSampler

from PIL import Image
import os
import os.path

def default_loader(path):
    img = Image.open(path)
    w, h = img.size
    if w == 112:
        return img.crop((8, 0, 104, 112))
    return img

def default_list_reader(fileList):
    imgList = []
    labelList = []
    with open(fileList, 'r') as file:
        for index, line in enumerate(file.readlines()):
            splits = line.strip().split()
            if len(splits) > 2:
                label = line.strip().split(' ')[-1]
                name = line.replace(label, '').strip()
            elif len(splits) == 2:
                name = splits[0]
                label = splits[1]
            elif len(splits) == 1:
                name = splits[0]
                label = 0
            imgList.append((name, int(label)))
            labelList.append(int(label))
            assert '.' in name, index

    return imgList, labelList

class ImageList(data.Dataset):
    def __init__(self, root, fileList, train=True, flip=False, transform=None, list_reader=default_list_reader, loader=default_loader):
        self.root = root
        self.imgList, self.labelList = list_reader(fileList)
        self.transform = transform
        self.loader = loader
        self.flip = flip
        self.train = train
        self.labels = torch.Tensor(self.labelList)

    def __getitem__(self, index):
        newindex = index
        if self.flip:
            newindex = newindex // 2
        imgPath, target = self.imgList[newindex]
        img = self.loader(os.path.join(self.root, imgPath))
        if self.flip and index % 2 == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return 2 * len(self.imgList) if self.flip else len(self.imgList)

