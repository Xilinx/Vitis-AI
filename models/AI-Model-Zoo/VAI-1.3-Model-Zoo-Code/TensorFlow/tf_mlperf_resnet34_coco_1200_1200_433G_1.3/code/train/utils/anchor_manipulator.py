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


# Copyright 2018 Changan Wang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import math

import tensorflow as tf
import numpy as np
from tensorflow.contrib.image.python.ops import image_ops

def center2point(center_x, center_y, width, height):
    return center_x - width / 2., center_y - height / 2., center_x + width / 2., center_y + height / 2.

def point2center(xmin, ymin, xmax, ymax):
    width, height = (xmax - xmin), (ymax - ymin)
    return xmin + width / 2., ymin + height / 2., width, height

def areas(bboxes):
    with tf.name_scope('bboxes_areas', values=[bboxes]):
        xmin, ymin, xmax, ymax = tf.split(bboxes, 4, axis=1)
        return (xmax - xmin) * (ymax - ymin)

def calc_intersection(gt_bboxes, default_bboxes):
    with tf.name_scope('bboxes_intersection', values=[gt_bboxes, default_bboxes]):
        gt_xmin, gt_ymin, gt_xmax, gt_ymax = tf.split(gt_bboxes, 4, axis=1)
        anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax = [tf.transpose(b, perm=[1, 0]) for b in tf.split(default_bboxes, 4, axis=1)]
        inter_xmin = tf.maximum(gt_xmin, anchor_xmin)
        inter_ymin = tf.maximum(gt_ymin, anchor_ymin)
        inter_xmax = tf.minimum(gt_xmax, anchor_xmax)
        inter_ymax = tf.minimum(gt_ymax, anchor_ymax)
        inter_w = tf.maximum(inter_xmax - inter_xmin, 0.)
        inter_h = tf.maximum(inter_ymax - inter_ymin, 0.)
        return inter_w * inter_h

def calc_iou_matrix(gt_bboxes, default_bboxes):
    with tf.name_scope('iou_matrix', values=[gt_bboxes, default_bboxes]):
        inter_matrix = calc_intersection(gt_bboxes, default_bboxes)
        union_matrix = areas(gt_bboxes) + tf.transpose(areas(default_bboxes), perm=[1, 0]) - inter_matrix
        return tf.where(tf.equal(union_matrix, 0.0),
                        tf.zeros_like(inter_matrix), tf.truediv(inter_matrix, union_matrix))

def do_dual_max_match(overlap_matrix, low_thres, high_thres, ignore_between=True, gt_max_first=True):
    '''
    overlap_matrix: num_gt * num_anchors
    '''
    with tf.name_scope('dual_max_match', values=[overlap_matrix]):
        anchors_to_gt = tf.argmax(overlap_matrix, axis=0)
        match_values = tf.reduce_max(overlap_matrix, axis=0)
        less_mask = tf.less(match_values, low_thres)
        between_mask = tf.logical_and(tf.less(match_values, high_thres), tf.greater_equal(match_values, low_thres))
        negative_mask = less_mask if ignore_between else tf.logical_or(less_mask, between_mask)
        ignore_mask = between_mask if ignore_between else tf.less(tf.zeros_like(less_mask), -1)
        match_indices = tf.where(negative_mask, -1 * tf.ones_like(anchors_to_gt), anchors_to_gt)
        match_indices = tf.where(ignore_mask, -2 * tf.ones_like(match_indices), match_indices)

        anchors_to_gt_mask = tf.one_hot(tf.clip_by_value(match_indices, -1, tf.cast(tf.shape(overlap_matrix)[0], tf.int64)),
                                        tf.shape(overlap_matrix)[0], on_value=1, off_value=0, axis=0, dtype=tf.int32)
        gt_to_anchors = tf.argmax(overlap_matrix, axis=1)
        if gt_max_first:
            left_gt_to_anchors_mask = tf.one_hot(gt_to_anchors, tf.shape(overlap_matrix)[1], on_value=1, off_value=0, axis=1, dtype=tf.int32)
        else:
            left_gt_to_anchors_mask = tf.cast(tf.logical_and(tf.reduce_max(anchors_to_gt_mask, axis=1, keep_dims=True) < 1,
                                                             tf.one_hot(gt_to_anchors, tf.shape(overlap_matrix)[1],
                                                                        on_value=True, off_value=False, axis=1, dtype=tf.bool)
                                                             ), tf.int64)
        left_gt_to_anchors_scores = overlap_matrix * tf.to_float(left_gt_to_anchors_mask)
        selected_scores = tf.gather_nd(overlap_matrix,  tf.stack([tf.where(tf.reduce_max(left_gt_to_anchors_mask, axis=0) > 0,
                                                                           tf.argmax(left_gt_to_anchors_scores, axis=0),
                                                                           anchors_to_gt),
                                                                  tf.range(tf.cast(tf.shape(overlap_matrix)[1], tf.int64))], axis=1))
        return tf.where(tf.reduce_max(left_gt_to_anchors_mask, axis=0) > 0,
                        tf.argmax(left_gt_to_anchors_scores, axis=0),
                        match_indices), selected_scores


class En_Decoder(object):
    def __init__(self, allowed_borders, positive_threshold, ignore_threshold, prior_scaling):
        super(En_Decoder, self).__init__()
        self._allowed_borders = allowed_borders
        self._positive_threshold = positive_threshold
        self._ignore_threshold = ignore_threshold
        self._prior_scaling = prior_scaling

    def encode_all_anchors(self, labels, bboxes, default_bboxes_ltrb, all_num_anchors_depth, all_num_anchors_spatial):
        assert len(all_num_anchors_depth) == len(all_num_anchors_spatial), 'inconsist num layers for anchors.'
        gt_ymin, gt_xmin, gt_ymax, gt_xmax = tf.unstack(bboxes, 4, axis=-1)
        bboxes = tf.stack([gt_xmin, gt_ymin, gt_xmax, gt_ymax], axis=-1)
        with tf.name_scope('encode_all_anchors'):
            tiled_allowed_borders = []
            [tiled_allowed_borders.extend([self._allowed_borders[ind]] * all_num_anchors_depth[ind] * all_num_anchors_spatial[ind]) for ind in range(len(all_num_anchors_depth))]
            anchor_allowed_borders = tf.stack(tiled_allowed_borders, 0, name='concat_allowed_borders')
            anchors_xmin, anchors_ymin, anchors_xmax, anchors_ymax = default_bboxes_ltrb
            inside_mask = tf.logical_and(tf.logical_and(anchors_xmin > -anchor_allowed_borders * 1.,
                                                        anchors_ymin > -anchor_allowed_borders * 1.),
                                         tf.logical_and(anchors_xmax < (1. + anchor_allowed_borders * 1.),
                                                        anchors_ymax < (1. + anchor_allowed_borders * 1.)))

            default_bboxes_ltrb = tf.stack(default_bboxes_ltrb, axis=-1)
            overlap_matrix = calc_iou_matrix(bboxes, default_bboxes_ltrb) * tf.cast(tf.expand_dims(inside_mask, 0), tf.float32)
            matched_gt, gt_scores = do_dual_max_match(overlap_matrix, self._ignore_threshold, self._positive_threshold)
            matched_gt_mask = matched_gt > -1
            matched_indices = tf.clip_by_value(matched_gt, 0, tf.int64.max)
            gt_labels = tf.gather(labels, matched_indices)
            gt_labels = gt_labels * tf.cast(matched_gt_mask, tf.int64)
            gt_labels = gt_labels + (-1 * tf.cast(matched_gt < -1, tf.int64))
            gt_xmin, gt_ymin, gt_xmax, gt_ymax = tf.unstack(tf.gather(bboxes, matched_indices), 4, axis=-1)
            gt_cx, gt_cy, gt_w, gt_h = point2center(gt_xmin, gt_ymin, gt_xmax, gt_ymax)
            anchor_cx, anchor_cy, anchor_w, anchor_h = point2center(anchors_xmin, anchors_ymin, anchors_xmax, anchors_ymax)
            gt_cx = (gt_cx - anchor_cx) / anchor_w / self._prior_scaling[0]
            gt_cy = (gt_cy - anchor_cy) / anchor_h / self._prior_scaling[1]
            gt_w = tf.log(gt_w / anchor_w) / self._prior_scaling[2]
            gt_h = tf.log(gt_h / anchor_h) / self._prior_scaling[3]
            gt_targets = tf.stack([gt_cx, gt_cy, gt_w, gt_h], axis=-1)
            gt_targets = tf.expand_dims(tf.cast(matched_gt_mask, tf.float32), -1) * gt_targets

            return gt_targets, gt_labels, gt_scores

    def decode_all_anchors(self, pred_location, default_bboxes, num_anchors_per_layer):
        with tf.name_scope('decode_all_anchors', values=[pred_location]):
            anchor_cx, anchor_cy, anchor_w, anchor_h = default_bboxes
            pred_cx = pred_location[:, 0] * self._prior_scaling[0] * anchor_w + anchor_cx
            pred_cy = pred_location[:, 1] * self._prior_scaling[1] * anchor_h + anchor_cy
            pred_w = tf.exp(pred_location[:, 2] * self._prior_scaling[2]) * anchor_w
            pred_h = tf.exp(pred_location[:, 3] * self._prior_scaling[3]) * anchor_h

            return tf.stack(center2point(pred_cx, pred_cy, pred_w, pred_h), axis=-1)


class DefaultBoxes(object):
    def __init__(self, img_shape, layers_shapes, anchor_scales, extra_anchor_scales, anchor_ratios, layer_steps, clip=False):
        super(DefaultBoxes, self).__init__()
        self._img_shape = img_shape
        self._layers_shapes = layers_shapes
        self._anchor_scales = anchor_scales
        self._extra_anchor_scales = extra_anchor_scales
        self._anchor_ratios = anchor_ratios
        self._layer_steps = layer_steps
        self._anchor_offset = [0.5] * len(self._layers_shapes)
        self._clip = clip

    def get_layer_anchors(self, layer_shape, anchor_scale, extra_anchor_scale, anchor_ratio, layer_step, offset = 0.5):
        ''' assume layer_shape[0] = 6, layer_shape[1] = 5
        x_on_layer = [[0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4]]
        y_on_layer = [[0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1],
                       [2, 2, 2, 2, 2],
                       [3, 3, 3, 3, 3],
                       [4, 4, 4, 4, 4],
                       [5, 5, 5, 5, 5]]
        '''
        with tf.name_scope('get_layer_anchors'):
            x_on_layer, y_on_layer = tf.meshgrid(tf.range(layer_shape[1]), tf.range(layer_shape[0]))
            x_on_image = (tf.cast(x_on_layer, tf.float32) + offset) * layer_step / self._img_shape[1]
            y_on_image = (tf.cast(y_on_layer, tf.float32) + offset) * layer_step / self._img_shape[0]
            num_anchors_along_depth = 2 * len(anchor_scale) * len(anchor_ratio) + len(anchor_scale) + len(extra_anchor_scale)
            num_anchors_along_spatial = layer_shape[0] * layer_shape[1]
            list_w_on_image = []
            list_h_on_image = []
            for _, scales in enumerate(zip(anchor_scale, extra_anchor_scale)):
                list_w_on_image.append(scales[0])
                list_h_on_image.append(scales[0])
                list_w_on_image.append(math.sqrt(scales[0] * scales[1]))
                list_h_on_image.append(math.sqrt(scales[0] * scales[1]))
            for _, scale in enumerate(anchor_scale):
                for _, ratio in enumerate(anchor_ratio):
                    w, h = scale * math.sqrt(ratio), scale / math.sqrt(ratio), 
                    list_w_on_image.append(w)
                    list_h_on_image.append(h)
                    list_w_on_image.append(h)
                    list_h_on_image.append(w)

            return  tf.expand_dims(x_on_image, axis=-1), tf.expand_dims(y_on_image, axis=-1), \
                    tf.constant(list_w_on_image, dtype=tf.float32), tf.constant(list_h_on_image, dtype=tf.float32), \
                    num_anchors_along_depth, num_anchors_along_spatial

    def get_all_anchors(self):
        all_num_anchors_depth = []
        all_num_anchors_spatial = []
        list_anchors_xmin = []
        list_anchors_ymin = []
        list_anchors_xmax = []
        list_anchors_ymax = []
        for ind, layer_shape in enumerate(self._layers_shapes):
            anchors_in_layer = self.get_layer_anchors(layer_shape,
                                                      self._anchor_scales[ind],
                                                      self._extra_anchor_scales[ind],
                                                      self._anchor_ratios[ind],
                                                      self._layer_steps[ind],
                                                      self._anchor_offset[ind])
            anchors_xmin, anchors_ymin, anchors_xmax, anchors_ymax = center2point(anchors_in_layer[0], anchors_in_layer[1], \
                                                                                  anchors_in_layer[2], anchors_in_layer[3])
            anchors_xmin = tf.transpose(anchors_xmin, perm=[2, 0, 1])
            anchors_ymin = tf.transpose(anchors_ymin, perm=[2, 0, 1])
            anchors_xmax = tf.transpose(anchors_xmax, perm=[2, 0, 1])
            anchors_ymax = tf.transpose(anchors_ymax, perm=[2, 0, 1])
            list_anchors_xmin.append(tf.reshape(anchors_xmin, [-1]))
            list_anchors_ymin.append(tf.reshape(anchors_ymin, [-1]))
            list_anchors_xmax.append(tf.reshape(anchors_xmax, [-1]))
            list_anchors_ymax.append(tf.reshape(anchors_ymax, [-1]))
            all_num_anchors_depth.append(anchors_in_layer[-2])
            all_num_anchors_spatial.append(anchors_in_layer[-1])
        anchors_xmin = tf.concat(list_anchors_xmin, 0, name='concat_xmin')
        anchors_ymin = tf.concat(list_anchors_ymin, 0, name='concat_ymin')
        anchors_xmax = tf.concat(list_anchors_xmax, 0, name='concat_xmax')
        anchors_ymax = tf.concat(list_anchors_ymax, 0, name='concat_ymax')
        if self._clip:
            anchors_xmin = tf.clip_by_value(anchors_xmin, 0., 1.)
            anchors_ymin = tf.clip_by_value(anchors_ymin, 0., 1.)
            anchors_xmax = tf.clip_by_value(anchors_xmax, 0., 1.)
            anchors_ymax = tf.clip_by_value(anchors_ymax, 0., 1.)
        default_bboxes_ltrb = (anchors_xmin, anchors_ymin, anchors_xmax, anchors_ymax)
        anchor_cx, anchor_cy, anchor_w, anchor_h = point2center(anchors_xmin, anchors_ymin, anchors_xmax, anchors_ymax)
        default_bboxes = (anchor_cx, anchor_cy, anchor_w, anchor_h)  

        return default_bboxes, default_bboxes_ltrb, all_num_anchors_depth, all_num_anchors_spatial


if __name__ == "__main__":
    import os
    out_shape = [1200] * 2
    defaultbox_creator = DefaultBoxes(out_shape,
                                      layers_shapes = [(50, 50), (25, 25), (13, 13), (7, 7), (3, 3), (3, 3)],
                                      anchor_scales = [(0.07,), (0.15,), (0.33,), (0.51,), (0.69,), (0.87,)],
                                      extra_anchor_scales = [(0.15,), (0.33,), (0.51,), (0.69,), (0.87,), (1.05,)],
                                      anchor_ratios = [(2, ), (2., 3.), (2., 3), (2., 3.), (2.,), (2.,)],
                                      layer_steps = [24, 48, 92, 171, 400, 400])
    default_bboxes, default_bboxes_ltrb, all_num_anchors_spatial, all_num_anchors_spatial = defaultbox_creator.get_all_anchors()
    default_bboxes_ltrb = tf.stack(default_bboxes_ltrb, axis=-1)  
    sess = tf.Session()
    anchor_bboxes = sess.run(default_bboxes_ltrb)
