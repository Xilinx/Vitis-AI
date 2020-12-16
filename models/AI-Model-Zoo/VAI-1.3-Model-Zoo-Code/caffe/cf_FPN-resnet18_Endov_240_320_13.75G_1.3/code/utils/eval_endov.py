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
import cv2
import numpy as np


def general_dice(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for instrument_id in set(y_true.flatten()):
        #if instrument_id == 0:
        #    continue
        result += [dice(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)


def general_jaccard(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for instrument_id in set(y_true.flatten()):
        #if instrument_id == 0:
        #    continue
        result += [jaccard(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)


def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection ) / (union )


def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() ) / (y_true.sum() + y_pred.sum())

def id2trainId(label, id_to_trainid, reverse=False):
    label_copy = label.copy()
    if reverse:
        for v, k in id_to_trainid.items():
            label_copy[label == k] = v
    else:
        for k, v in id_to_trainid.items():
            label_copy[label == k] = v
    return label_copy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--predpath', type=str, default='results',help='path where train images with ground truth are located')
    arg('--gtpath', type=str, default='data/Endov/val/', help='path with predictions')
    arg('--type', type=str, default='class', choices=['class', 'instruments'])
    args = parser.parse_args()

    result_dice = []
    result_jaccard = []

    assert args.type == 'class'
    id_to_trainid = {0:0, 70: 1, 160:2}
    categories = os.listdir(args.predpath)
    for c in categories:
        mask_path = os.path.join(args.predpath, c)
        c_items = [name.split('_raw.png')[0] for name in os.listdir(mask_path)]
        for it in c_items:
            print(c, it)
            gt_file = os.path.join(args.gtpath, c, 'Masks', it+'_class.png')
            label = cv2.imread(gt_file, 0)
            y_true = id2trainId(label, id_to_trainid)   
            pred_file = os.path.join(args.predpath, c, it+'_raw.png')

            y_pred = cv2.imread(str(pred_file), 0)
            result_dice += [general_dice(y_true, y_pred)]
            result_jaccard += [general_jaccard(y_true, y_pred)]
    print('Dice = ', np.mean(result_dice))
    print('Jaccard = ', np.mean(result_jaccard))
