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


# MIT License

# Copyright (c) 2019 Hengshuang Zhao

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import argparse
import copy
import json
import os
from collections import defaultdict

import os.path as osp

import numpy as np
from PIL import Image


def parse_args():
    """Use argparse to get command line arguments."""
    '''
    task: segmentation
    gt: path to ground truth
    result: path to results to be evaluated
    num_classes: dataset classes. #e.g. 19 for cityscapes
    ignore_label: label to be ignored when to do evalution. #e.g. 255 for cityscapes
    result_suffix: prediction result file suffix. #e.g. leftImg8bit.png
    gt_suffix: ground truth file suffix. #e.g. gtFine_trainIds.png
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='segmentation', help='evaluation task name')
    parser.add_argument('--gt', help='path to ground truth')
    parser.add_argument('--result', help='path to results to be evaluated')
    parser.add_argument('--result_suffix', type=str, default='leftImg8bit.png', help = 'prediction result file suffix')
    parser.add_argument('--gt_suffix',  type=str, default='gtFine_trainIds.png', help = 'groundtruth file suffix')
    parser.add_argument('--num_classes', type=int, default=19, help='dataset classes')
    parser.add_argument('--ignore_label', type=int, default=255, help='dataset classes')
    parser.add_argument('--result_file', type=str, default='accuracy.txt', help = 'save accuracy to file')
    args = parser.parse_args()

    return args


def fast_hist(gt, prediction, n):
    k = (gt >= 0) & (gt < n) 
    return np.bincount(
        n * gt[k].astype(int) + prediction[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    ious = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    ious[np.isnan(ious)] = 0
    return ious


def find_all_pred_png(folder):
    paths = []
    for root, dirs, files in os.walk(folder, topdown=True):
        paths.extend([osp.join(root, f)
                      for f in files if osp.splitext(f)[1] == '.png'])
    return paths

def find_all_gt_png(folder):
    paths = []
    for root, dirs, files in os.walk(folder, topdown=True):
        paths.extend([osp.join(root, f)
                      for f in files if osp.splitext(f)[1] == '.png'])
    return paths

def evaluate_segmentation(gt_dir, result_dir, num_classes, ignore_label, result_suffix, gt_suffix, result_file):
    gt_dict = dict([(osp.split(p)[1][:-len(gt_suffix)], p)
                    for p in find_all_gt_png(gt_dir)])
    
    result_dict = dict([(osp.split(p)[1][:-len(result_suffix)], p) # -16
                        for p in find_all_pred_png(result_dir)])
    result_gt_keys = set(gt_dict.keys()) & set(result_dict.keys())
    if len(result_gt_keys) != len(gt_dict):
        raise ValueError('Result folder only has {} of {} ground truth files.'
                         .format(len(result_gt_keys), len(gt_dict)))
    hist = np.zeros((num_classes, num_classes))
    i = 0
    gt_id_set = set()
    for key in sorted(gt_dict.keys()):
        gt_path = gt_dict[key]
        result_path = result_dict[key]
        gt = np.asarray(Image.open(gt_path, 'r'))
        gt_id_set.update(np.unique(gt).tolist())
        prediction = np.asanyarray(Image.open(result_path, 'r'))
        hist += fast_hist(gt.flatten(), prediction.flatten(), num_classes)
        i += 1
    if ignore_label is not None:
        gt_id_set.remove(ignore_label)
        print("Ignore the label id {} ".format(ignore_label))
    else:
       gt_id_set.remove(255)
    print('GT id set', gt_id_set)
    ious = per_class_iu(hist) * 100
    miou = np.mean(ious[list(gt_id_set)])

    with open(result_file, 'w') as f:
        f.write('mean IoU(%): ' + str(format(miou,'.2f')) + '\n')
        f.write('per-class IoU(%): ' + str(ious)+ '\n\n')
    return miou, list(ious)


def main():
    """
    usage:
    python evaluate_miou.py --task segmentation --gt $GT_PATH --result $PREDICTION_PATH
    """
    args = parse_args()
    assert args.task == 'segmentation'
    mean, breakdown = evaluate_segmentation(args.gt, args.result, args.num_classes, args.ignore_label, args.result_suffix, args.gt_suffix, args.result_file)

    print('mIoU(%): {:.2f}'.format(mean))
    print(','.join(['{:.2f}'.format(n) for n in breakdown]))


if __name__ == '__main__':
    main()
