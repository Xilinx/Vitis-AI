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



#coding=utf-8

import tensorflow as tf

from utils.prior_box import *

def transform_yx2xy(bboxes):
    ymin, xmin, ymax, xmax = tf.split(bboxes, 4, axis=-1)
    return tf.concat([xmin, ymin, xmax, ymax], axis=-1)

def encode(gt_bboxes, priorboxes, varinaces):
    '''
    Args: 
      gt_loc: xmin, ymin, xmax, ymax
      priorboxes: cx, cy, w, h 
      varinaces: scale for cx, cy, w, h
    return: groundtruth after encode.   
    '''
    gt_xmin, gt_ymin, gt_xmax, gt_ymax = tf.split(gt_bboxes, 4, axis=-1)
    gt_cx, gt_cy, gt_w, gt_h = point2center(gt_xmin, gt_ymin, gt_xmax, gt_ymax)
    cx, cy, w, h = tf.split(priorboxes, 4, axis=-1)
    en_cx = (gt_cx - cx) / w / varinaces[0]
    en_cy = (gt_cy - cy) / h / varinaces[1]
    en_w = tf.log(gt_w / w) / varinaces[-2]
    en_h = tf.log(gt_h / h) / varinaces[-1]
    return tf.concat([en_cx, en_cy, en_w, en_h], axis=-1)

def decode(pred_loc, priorboxes, varinaces):
    cx, cy, w, h = tf.split(priorboxes, 4, axis=-1)
    pred_w = tf.exp(tf.expand_dims(pred_loc[:, -2], axis=-1) * varinaces[-2]) * w
    pred_h = tf.exp(tf.expand_dims(pred_loc[:, -1], axis=-1) * varinaces[-1]) * h
    pred_cx = tf.expand_dims(pred_loc[:, 0], axis=-1) * varinaces[0] * w + cx
    pred_cy = tf.expand_dims(pred_loc[:, 1], axis=-1) * varinaces[1] * h + cy
    return to_ltrb_mi(pred_cx, pred_cy, pred_w, pred_h)
