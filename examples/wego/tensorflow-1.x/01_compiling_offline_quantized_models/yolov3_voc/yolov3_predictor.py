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


# This file is modified from other's code from github.
# For more details, please refer to https://github.com/aloyschen/tensorflow-yolo3.git


import os
import sys
import numpy as np
import tensorflow as tf


class yolo_predictor:
    def __init__(self, config):
        self.obj_threshold = config.score_thresh
        self.nms_threshold = config.nms_thresh
        self.class_names = config.classes
        self.anchors = np.array(config.anchors)

    def eval(self, yolo_outputs, image_shape, max_boxes=20):
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes = []
        box_scores = []
        input_shape = tf.shape(yolo_outputs[0])[1: 3] * 32
        input_shape = tf.expand_dims(input_shape, 0)
        for i in range(len(yolo_outputs)):
            _boxes, _box_scores = self.boxes_and_scores(
                yolo_outputs[i], self.anchors[anchor_mask[i]], len(self.class_names), input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = tf.concat(boxes, axis=1)
        boxes = tf.expand_dims(boxes, 2)
        box_scores = tf.concat(box_scores, axis=1)

        max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)
        nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes,
            box_scores,
            max_boxes_tensor,
            max_boxes,
            iou_threshold=self.nms_threshold,
            score_threshold=self.obj_threshold,
            pad_per_class=False,
            clip_boxes=False,
            name=None
        )
        return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections

    def boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(
            feats, anchors, classes_num, input_shape)
        boxes = self.correct_boxes(box_xy, box_wh, input_shape, image_shape)
        batch = tf.shape(boxes)[0]
        boxes = tf.reshape(boxes, [batch, -1, 4])
        box_scores = box_confidence * box_class_probs
        box_scores = tf.reshape(box_scores, [batch, -1, classes_num])
        return boxes, box_scores

    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = tf.cast(input_shape, dtype=tf.float32)
        image_shape = tf.cast(image_shape, dtype=tf.float32)
        new_shape = tf.round(
            image_shape * tf.reduce_min(input_shape / image_shape))
        #[batch, 2]
        offset = (input_shape - new_shape) / 2. / input_shape
        offset = tf.expand_dims(offset, 1)
        offset = tf.expand_dims(offset, 1)
        offset = tf.expand_dims(offset, 1)
        #[batch, 2]
        scale = input_shape / new_shape
        scale = tf.expand_dims(scale, 1)
        scale = tf.expand_dims(scale, 1)
        scale = tf.expand_dims(scale, 1)
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = tf.concat([
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2]
        ], axis=-1)
        image_shape = tf.expand_dims(image_shape, 1)
        image_shape = tf.expand_dims(image_shape, 1)
        image_shape = tf.expand_dims(image_shape, 1)
        boxes *= tf.concat([image_shape, image_shape], axis=-1)
        return boxes

    def _get_feats(self, feats, anchors, num_classes, input_shape):
        num_anchors = len(anchors)
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [
                                    1, 1, 1, num_anchors, 2])

        grid_size = tf.shape(feats)[1:3]
        predictions = tf.reshape(
            feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])
        box_xy, box_wh, box_confidence, box_class_probs = tf.split(
            predictions, (2, 2, 1, num_classes), axis=-1
        )
        grid_y = tf.tile(tf.reshape(
            tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [
                         1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis=-1)
        grid = tf.cast(grid, tf.float32)
        box_xy = (tf.sigmoid(box_xy) + grid) / tf.cast(grid_size, tf.float32)

        box_wh = tf.exp(box_wh) * anchors_tensor / \
            tf.cast(input_shape[..., ::-1], tf.float32)
        box_confidence = tf.sigmoid(box_confidence)
        box_class_probs = tf.sigmoid(box_class_probs)
        return box_xy, box_wh, box_confidence, box_class_probs

    def predict(self, output, image_shape):
        boxes, scores, classes, valid_detections = self.eval(
            output, image_shape, max_boxes=20)
        return boxes, scores, classes, valid_detections
