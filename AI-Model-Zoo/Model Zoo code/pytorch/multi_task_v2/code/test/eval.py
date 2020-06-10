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
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
from model_res18 import build_model
from layers import *
from collections import OrderedDict
import sys
import os
import time
import argparse
import numpy as np
import cv2
import math
labelmap_det = (  # always index 0
    'car', 'sign', 'person')
solver = {
    'k1': 8,
    'k2': 8,
    'act_clip_val': 8,
    ' warmup': False,
    'det_classes': 4,
    'seg_classes': 16,
    'lr_steps': (12000, 18000),
    #'lr_steps': (5, 10),
    'max_iter': 20010,
    'feature_maps': [(80,128), (40,64), (20,32), (10,16), (5,8), (3,6), (1,4)],
    'resize': (320,512),
    'steps': [4, 8, 16, 32, 64, 128, 256],
    'min_sizes': [10, 30, 60, 100, 160, 220, 280],
    'max_sizes': [30, 60, 100, 160, 220, 280, 340],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2]],
    'variance': [0.1, 0.2],
    'clip': False,
}
MEANS = (104, 117, 123)
def base_transform(image, size, mean):
    x = cv2.resize(image, (size[1], size[0])).astype(np.float32)
    x -= mean
    x = x/256.0
    x = x.astype(np.float32)
    return x

class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image):
        return base_transform(image, self.size, self.mean)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='./weights/iter_v3_finetune_4000.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='result/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--image_root', default=None, type=str,
                    help='image_root_path')
parser.add_argument('--image_list', default=None, type=str,
                    help='image_list')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def get_voc_results_file_template(cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_test' + '_%s.txt' % (cls)
    filedir = os.path.join(args.save_folder, 'det')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, ids):
    for cls_ind, cls in enumerate(labelmap_det):
        print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(ids):
                dets = all_boxes[cls_ind+1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    if dets.shape[1] == 5: 
                        f.write('{:s} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\n'.
                                format(index[1], dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
                    elif dets.shape[1] == 7:
                        angle_sin = math.asin(dets[k, 4]) / math.pi * 180
                        angle_cos = math.acos(dets[k, 5]) / math.pi * 180
                        if dets[k, 4] >= 0 and dets[k, 5] >= 0:
                            angle = (angle_sin + angle_cos) / 2
                        elif dets[k, 4] < 0 and dets[k, 5] >= 0:
                            angle = (angle_sin - angle_cos) / 2
                        elif dets[k, 4] < 0 and dets[k, 5] < 0:
                            angle = (-180 - angle_sin - angle_cos) / 2
                        elif dets[k, 4] >= 0 and dets[k, 5] < 0:
                            angle = (angle_cos + 180 - angle_sin) / 2
                        f.write('{:s} {:s} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\n'.
                                format(index[1],cls, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
                                      # angle))

def test_net(save_folder, net, cuda, ids, detect, transform, thresh=0.01, has_ori=False):
    num_images = len(ids)
    softmax = nn.Softmax(dim=-1)
    img_path = os.path.join('%s', 'images', '%s.jpg')
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap_det)+1)]
    save_seg_root = os.path.join(save_folder, 'seg')
    if not os.path.exists(save_seg_root): 
        os.mkdir(save_seg_root)
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    for i in range(num_images):
        img_id = ids[i]
        file_path = img_path % img_id
        im = cv2.imread(file_path)
        im_size = (im.shape[1], im.shape[0])
        x = torch.from_numpy(transform(im)).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        loc_or, conf_dat, seg_data = net(x)
        priorbox = PriorBox(solver)
        priors = Variable(priorbox.forward(), volatile=True).cuda()
        loc_ori=list()
        conf_data=list()
        for loc in loc_or:
            loc_ori.append(loc.permute(0,2,3,1).contiguous().cuda())
        loc_ori = torch.cat([o.view(o.size(0), -1) for o in loc_ori],1).cuda()
        loc_ori=loc_ori.view(loc_ori.size(0), -1, 6).cuda()
        for conf in conf_dat:
            conf_data.append(conf.permute(0,2,3,1).contiguous().cuda())
        conf_data = torch.cat([o.view(o.size(0), -1) for o in conf_data],1).cuda()
        conf_data = conf_data.view(conf_data.size(0), -1, 4).cuda()
        pred = detect(loc_ori, softmax(conf_data), priors)
        detections = pred.data
        detect_time = _t['im_detect'].toc(average=False)
        
        seg_data = np.squeeze(seg_data.data.max(1)[1].cpu().numpy(), axis=0)
        seg_data = cv2.resize(seg_data, im_size, interpolation=cv2.INTER_NEAREST)
        save_seg_file = os.path.join(save_seg_root, img_id[1] + '.png')
        cv2.imwrite(save_seg_file, seg_data)
        # skip j = 0, because it's the background class
        
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            feature_dim = dets.size(1)
            mask = dets[:, 0].gt(0.).expand(feature_dim, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, feature_dim)
            if dets.dim() == 0:
                continue
            boxes = dets[:, 1:5]
            boxes[:, 0] *= im_size[0]
            boxes[:, 2] *= im_size[0]
            boxes[:, 1] *= im_size[1]
            boxes[:, 3] *= im_size[1]
            scores = dets[:, 0].cpu().numpy()
            if has_ori:
                cls_dets = np.hstack((dets[:, 1:feature_dim].cpu().numpy(),
                                      scores[:, np.newaxis])).astype(np.float32,
                                                                     copy=False)
            else:
                cls_dets = np.hstack((boxes.cpu().numpy(),
                                      scores[:, np.newaxis])).astype(np.float32,
                                                                     copy=False)
            all_boxes[j][i] = cls_dets
        
        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))
    
    print('Saveing the detection results...')
    write_voc_results_file(all_boxes, ids) 

if __name__ == '__main__':
    # load net
    det_classes = solver['det_classes']                      # +1 for background
    seg_classes = solver['seg_classes']
    print(seg_classes)
    net = build_model(det_classes, seg_classes)
    state_dict = torch.load(args.trained_model)
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        if k[0:7] == 'module.':
            name = k[7:]
            new_state_dict[name] = v
        else:
            name = k
            new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()
    print('Finished loading model!')
    # load data
    # image_root = '/scratch/workspace/multi-task_v2/'
    # image_list = 'det_test.txt'
    # image_root='/scratch/workspace/multi-task_v2/'
    # image_list='seg_val.txt'
    ids = list()
    for line in open(os.path.join(args.image_root, args.image_list)):
        ids.append((args.image_root, line.strip()))
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    detect = Detect(det_classes, 0, 200, 0.01, 0.45) #num_classes, backgroung_label, top_k, conf_threshold, nms_threshold
    test_net(args.save_folder, net, args.cuda, ids, detect,
             BaseTransform(solver['resize'], MEANS), 
             thresh=args.confidence_threshold, has_ori=True)
