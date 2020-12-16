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

#-*-coding:utf-8-*-
import torch
import torchvision
import torchvision.transforms as Transforms
import torch.utils.data as Data
from PIL import Image
import os
import numpy as np


class PointsDataset(Data.Dataset):
    def __init__(self, root, annotation_file, transform):
        os.path.exists(root)
        fp = open(annotation_file, 'r')
        lines = fp.readlines()
        fp.close()
        self.annotation_list = []
        flag = 0
        for line in lines:
            line = line.strip('\n').split(' ')
            image_name = os.path.join(root, line[0])
            #assert os.path.exists(image_name)
            if len(line[1:]) == 10:
                points = np.array(line[1:], dtype = np.float)
                quality = np.array([0.])
                flag = 0
            elif len(line[1:]) == 1:
                points = np.zeros(10)
                quality = np.array(line[1:], dtype = np.float)
                flag = 1
            else:
                assert False, 'wrong label {}'.format(line)
            # normalize the label
            # points[:5] = points[:5] / 72.
            # points[5:] = points[5:] / 96.
            # quality = quality / 300.
            # points = points * 5. / 6 # resize to (80, 60)
            points = points.tolist()
            quality = quality.tolist()
            self.annotation_list.append((image_name, points, quality, flag))
        self.transform = transform
    def __getitem__(self, index):
        annotation = self.annotation_list[index]
        image = Image.open(annotation[0])
        image = self.transform(image)
        points = torch.tensor(annotation[1], dtype = torch.float)
        quality = torch.tensor(annotation[2], dtype = torch.float)
        flag = torch.tensor(annotation[3], dtype=torch.int)
        return image, (points, quality, flag)
    def __len__(self):
        return len(self.annotation_list)

def get_data(args):
    #train data
    train_dataset = PointsDataset(root = args.dataset_root,
                                  annotation_file = args.anno_train_list,
                                  transform = Transforms.Compose([Transforms.Resize(size = args.size),
                                                                  Transforms.Grayscale(3),
                                                                  Transforms.ToTensor(),
                                                                  Transforms.Normalize(mean = args.mean, std = args.std),
                                                                  ]),
                                  )
    train_loader = Data.DataLoader(dataset = train_dataset,
                                   batch_size = args.train_batch_size,
                                   shuffle = True,
                                   num_workers = args.train_worker,
                                   drop_last = True,
                                   )
    #test data
    test_dataset = PointsDataset(root = args.dataset_root,
                                 annotation_file=args.anno_test_list,
                                 transform = Transforms.Compose([Transforms.Resize(size = args.size),
                                                                Transforms.Grayscale(3),
                                                                Transforms.ToTensor(),
                                                                Transforms.Normalize(mean = args.mean, std = args.std),
                                                                ]),
                                )
    test_loader = Data.DataLoader(dataset = test_dataset,
                                   batch_size = args.test_batch_size,
                                   shuffle = False,
                                   num_workers = args.test_worker,
                                   drop_last = False,
                                   )
    return train_loader, test_loader

