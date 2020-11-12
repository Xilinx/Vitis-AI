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
from utils.en_decode import *

def cal_areas(bboxes):
    xmin, ymin, xmax, ymax = tf.split(bboxes, 4, axis=-1)
    return (xmax - xmin) * (ymax - ymin)

def cal_intersections(gt_bboxes, prior_bboxes_ltrb):
    gt_xmin, gt_ymin, gt_xmax, gt_ymax = tf.split(gt_bboxes, 4, axis=-1)
    p_xmin, p_ymin, p_xmax, p_ymax = [tf.transpose(p, perm=[1, 0]) for p in tf.split(prior_bboxes_ltrb, 4, axis=-1)]

    inter_xmin = tf.maximum(gt_xmin, p_xmin)
    inter_ymin = tf.maximum(gt_ymin, p_ymin)
    inter_xmax = tf.minimum(gt_xmax, p_xmax)
    inter_ymax = tf.minimum(gt_ymax, p_ymax)
    inter_w = tf.maximum(inter_xmax - inter_xmin, 0.)
    inter_h = tf.maximum(inter_ymax - inter_ymin, 0.)
    return inter_w * inter_h

def cal_ious(gt_bboxes, prior_bboxes):
    prior_bboxes_ltrb = to_ltrb(prior_bboxes)
    inter_mat = cal_intersections(gt_bboxes, prior_bboxes_ltrb)
    union_mat = cal_areas(gt_bboxes) + tf.transpose(cal_areas(prior_bboxes_ltrb), perm=[1, 0]) - inter_mat
    return tf.where(tf.equal(union_mat, 0.0),
                    tf.zeros_like(inter_mat), tf.truediv(inter_mat, union_mat))

def match(gt_labels, gt_bboxes, prior_bboxes, variances, threshold=0.5):
    overlap_mat = cal_ious(gt_bboxes, prior_bboxes)

    best_gt_idxs = tf.argmax(overlap_mat, axis=0)
    best_gt_overlaps = tf.reduce_max(overlap_mat, axis=0)    

    best_prior_idxs = tf.argmax(overlap_mat, axis=1)
    best_prior_overlaps = tf.reduce_max(overlap_mat, axis=1)

    len_best_prior_idxs = tf.shape(best_prior_idxs)[0]
    len_best_gt_idxs = tf.shape(best_gt_idxs)[0]

    def condition(idx, best_gt_idxs, best_gt_overlaps):
        return tf.less(idx, len_best_prior_idxs)

    def body(idx, best_gt_idxs, best_gt_overlaps):
        one_hot = tf.one_hot([best_prior_idxs[idx]], depth=len_best_gt_idxs) 
        one_hot_mask = one_hot > 0
        one_hot_mask = tf.squeeze(one_hot_mask, axis=0)

        best_gt_idxs = tf.where(one_hot_mask, tf.ones_like(best_gt_idxs, dtype=tf.int64) * tf.cast(idx, dtype=tf.int64), best_gt_idxs)
        best_gt_overlaps = tf.where(one_hot_mask, tf.ones_like(best_gt_idxs, dtype=tf.float32), best_gt_overlaps)
        #best_gt_overlaps = tf.where(one_hot_mask, tf.ones_like(best_gt_idxs, dtype=tf.float32) * best_prior_overlaps[idx], best_gt_overlaps)

        idx = idx + 1
        return idx, best_gt_idxs, best_gt_overlaps

    idx = 0
    [idx, best_gt_idxs, best_gt_overlaps] = tf.while_loop(condition, body, [idx, best_gt_idxs, best_gt_overlaps], parallel_iterations=1, back_prop=False, swap_memory=True)
  
    gather_labels = tf.gather(gt_labels, best_gt_idxs)
    gather_bboxes = tf.gather_nd(gt_bboxes, tf.expand_dims(best_gt_idxs, axis=-1))
    loc_t = encode(gather_bboxes, prior_bboxes, variances)

    select_mask = tf.greater_equal(best_gt_overlaps, threshold)
    conf_t = tf.where(select_mask, gather_labels, tf.zeros_like(gather_labels)) 
    #loc_t = gather_bboxes#gather_bboxes_en#tf.where(select_mask, tf.zeros_like(gather_bboxes_en), gather_bboxes_en)

    return conf_t, loc_t

def filter_by_score(conf_pred, filter_score):
    filter_mask = conf_pred > filter_score
    return tf.where(filter_mask, conf_pred, tf.zeros_like(conf_pred))

def clip_bboxes(decode_bboxes):
    xmin, ymin, xmax, ymax = tf.split(decode_bboxes, 4, axis=-1)
    xmin = tf.maximum(xmin, 0.)
    ymin = tf.maximum(ymin, 0.)
    xmax = tf.minimum(xmax, 1.)
    ymax = tf.minimum(ymax, 1.)
    xmin = tf.minimum(xmin, xmax)
    ymin = tf.minimum(ymin, ymax)
    return tf.concat([xmin, ymin, xmax, ymax], axis=-1)

def filter_bboxes_by_size(conf_pred, decode_bboxes, min_size):
    xmin, ymin, xmax, ymax = tf.split(decode_bboxes, 4, axis=-1)
    width, height = xmax - xmin, ymax - ymin
    filter_mask = tf.squeeze(tf.logical_and(width > min_size, height > min_size), axis=-1)
    return tf.where(filter_mask, conf_pred, tf.zeros_like(conf_pred))

def sort_bboxes(conf_pred, decode_bboxes, keep_topk):
    scores, idxs = tf.nn.top_k(conf_pred, k=keep_topk, sorted=True)
    decode_bboxes = tf.gather_nd(decode_bboxes, tf.expand_dims(idxs, axis=-1))
    return scores, decode_bboxes

def nms_bboxes(scores, decode_bboxes, keep_nms_maxnum, nms_threshold):
    xmin, ymin, xmax, ymax = tf.split(decode_bboxes, 4, axis=-1)
    decode_bboxes_yx = tf.concat([ymin, xmin, ymax, xmax], axis=-1)
    idxs = tf.image.non_max_suppression(decode_bboxes_yx, scores, keep_nms_maxnum, nms_threshold)
    return tf.gather(scores, idxs), tf.gather_nd(decode_bboxes, tf.expand_dims(idxs, axis=-1))

def padding_tensor(tensor, real_len, max_len, begin, size, padding_shape, dtype):
    tensor = tf.cond(real_len <= max_len, lambda: tensor, lambda: tf.slice(tensor, begin, size))
    tensor = tf.cond(real_len < max_len, lambda: tf.concat([tensor, tf.zeros(shape=padding_shape, dtype=dtype)], axis=0), lambda: tensor)
    return tensor

def detect_post_process(conf_pred, decode_bboxes, num_class, config):
    decode_bboxes = clip_bboxes(decode_bboxes) 
    conf_pred = filter_bboxes_by_size(conf_pred, decode_bboxes, config.min_size)

    pred_bboxes = []
    pred_scores = []
    pred_classes = []

    for idx in range(num_class - 1):
        scores, bboxes = sort_bboxes(conf_pred[:, idx], decode_bboxes, config.keep_topk)
        scores, bboxes = nms_bboxes(scores, bboxes, config.keep_nms_maxnum, config.nms_threshold)
        real_len = tf.shape(scores)[0] 
        scores = padding_tensor(scores, real_len, config.max_num_bboxes, [0], [config.max_num_bboxes], (config.max_num_bboxes - real_len), tf.float32) 
        bboxes = padding_tensor(bboxes, real_len, config.max_num_bboxes, [0, 0], [config.max_num_bboxes, 4], (config.max_num_bboxes - real_len, 4), tf.float32)
        class_ids = tf.cast((idx + 1), dtype=tf.int64) * tf.ones_like(scores, dtype=tf.int64) 
        pred_classes.append(class_ids)
        pred_scores.append(scores)
        pred_bboxes.append(bboxes)  
  
    detection_scores = tf.concat(pred_scores, axis=0)
    detection_classes = tf.concat(pred_classes, axis=0)
    detection_bboxes = tf.concat(pred_bboxes, axis=0)

    detection_scores, idxes = tf.nn.top_k(detection_scores, k=config.max_num_bboxes, sorted=True)
    detection_classes = tf.gather(detection_classes, idxes)
    detection_bboxes = tf.gather_nd(detection_bboxes, tf.expand_dims(idxes, axis=-1))

    return detection_scores, detection_classes, detection_bboxes
