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

#!/usr/bin/env python

import argparse
import logging
import os
import sys
import time

import cv2
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

logger = logging.getLogger()
logger.setLevel(logging.INFO)

USE_L2_METRIC = False


def load_pairs(pairs_path):
    # print("...Reading pairs.")
    pairs = []
    with open(pairs_path, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    assert(len(pairs) == 6000)
    return np.array(pairs)

def pairs_info(pair):
    suffix = 'jpg'
    if len(pair) == 3:
        name1 = "{}_{}.{}".format(pair[0], pair[1].zfill(4), suffix)
        name2 = "{}_{}.{}".format(pair[0], pair[2].zfill(4), suffix)
        same = 1
    elif len(pair) == 4:
        name1 = "{}_{}.{}".format(pair[0], pair[1].zfill(4), suffix)
        name2 = "{}_{}.{}".format(pair[2], pair[3].zfill(4), suffix)
        same = 0
    else:
        raise Exception(
            "Unexpected pair length: {}".format(len(pair)))
    return (name1, name2, same)

def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = accuracy_score(y_true, y_predict)
    return accuracy

def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold

def acc(predicts):
    # print("...Computing accuracy.")
    #folds = KFold(n=6000, n_folds=10, shuffle=False)
    folds = KFold(10, False)
    if USE_L2_METRIC:
        thresholds = np.arange(170, 180, 0.5)
    else:
        thresholds = np.arange(-1.0, 1.0, 0.005)
    accuracy = []
    thd = []
    # print(predicts)
    for idx, (train, test) in enumerate(folds.split(predicts)):
        # logging.info("processing fold {}...".format(idx))
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)
    return accuracy,thd

def get_predict_file(features, pair_path, write=False):
    pairs = load_pairs(pair_path)
    predicts = []
    for pair in pairs:
        name1, name2, same = pairs_info(pair)
        # logging.info("processing name1:{} <---> name2:{}".format(name1, name2))
        f1 = features[name1]
        f2 = features[name2]
        dis = np.dot(f1, f2)/np.linalg.norm(f1)/np.linalg.norm(f2)
        predicts.append([name1, name2, str(dis), str(same)])
    # print 'Done generate predict file!'
    return np.array(predicts)

def print_result(predicts):
    accuracy, threshold = acc(predicts)
    # logging.info("10-fold accuracy is:\n{}\n".format(accuracy))
    # logging.info("10-fold threshold is:\n{}\n".format(threshold))
    # print("mean threshold is {:.4f}".format(np.mean(threshold)))
    print("mean is {:.4f}, var is {:.4f}".format(np.mean(accuracy), np.std(accuracy)))
    return np.mean(accuracy)

def test(features, pair_path, write=False):
    start = time.time()
    predicts = get_predict_file(features, pair_path, write)
    accuracy, threshold = acc(predicts)
    print("mean is {:.4f}, var is {:.4f}, time: {}s".format(np.mean(accuracy), np.std(accuracy), time.time()-start))
    return np.mean(accuracy)

