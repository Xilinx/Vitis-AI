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
import torchvision.transforms as Transforms
import os
from PIL import Image
import math
import cv2
import time
import numpy as np
import sys
sys.path.append('../models/')
sys.path.append('../utils/')
sys.path.append('../configs')
import model
from model_config import Configs

def test(args):
    net = model.get_model()
    net.load_state_dict(torch.load(args.pretrained))
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()
    # list
    fp = open(args.visual_test_list, 'r')
    lines = fp.readlines()
    fp.close()
    # transform
    args.mean = (0.5, 0.5, 0.5)
    args.std = (0.5, 0.5, 0.5)
    transform = Transforms.Compose([Transforms.Resize(size = args.size),
                                    Transforms.Grayscale(3), # gray
                                    Transforms.ToTensor(),
                                    Transforms.Normalize(mean = args.mean, std = args.std),
                                    ])
    # test
    points_distance = 0.
    test_total = 0
    img_d = {}
    with torch.no_grad():
        for line in lines:
            line = line.strip('\n')
            image_name = line
            img = cv2.imread(image_name)
            #img = cv2.resize(img, (72, 96))
            # image input
            image = Image.open(image_name)
            # image = to_gray(image) # for night
            image = transform(image)
            image = torch.unsqueeze(image, 0)
            # points input
            # points = np.array(line[1:], dtype = np.float).tolist()
            # points_label = torch.tensor([points], dtype = torch.float)
            # forward
            image = image.to(device)
            test_total += image.size(0)
            points_output, quality_output  = net(image)
            # print(points_output, quality_output)
            points = points_output[0].cpu().data.numpy().reshape((2, 5)).T
            ##compute points for the original image
            points[:,0] = points[:,0] * img.shape[1] / 60.0
            points[:,1] = points[:,1]*img.shape[0] / 80.0

            #compute points for the (72*96) image
            #points = points * 6./5.
            q = quality_output[0].data.cpu().numpy()
            for p in points:
                cv2.circle(img,(int(p[0]),int(p[1])),2,(55,255,155),1)
            img_d[line] = [q[0], img]
            # points
            # points_label = points_label.to(device)
            # points_distance += torch.sum(torch.abs(points_output - points_label)).item()
    img_d = sorted(img_d.items(), key=lambda item : item[1][0])
    saveroot = 'results/'
    if not os.path.exists(saveroot):
        os.makedirs(saveroot)
    for index, k in enumerate(img_d):
        qu = k[1][0]
        q = 1/(1.+math.e**-((3.*qu-600.)/150.))
        img = k[1][1]
        name = '{}_{}_{}.jpg'.format(img_d[index][0].split('/')[1], q, qu)
        cv2.imwrite(os.path.join(saveroot, name), img)

def to_gray(img):
    img = img.convert('L')
    np_img = np.array(img, dtype=np.uint8)
    np_img = np.dstack([np_img, np_img, np_img])
    img = Image.fromarray(np_img, 'RGB')
    return img

if __name__ == '__main__':
    args = Configs().parse()
    test(args)
