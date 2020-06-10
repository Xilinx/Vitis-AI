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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile

from eval_ssd_large import *

tf.app.flags.DEFINE_string(
    'output_graph', 'resnet34_ssd.pbtxt',
    'exported pbtxt file.') 
global_anchor_info = {}

def ssd_model_fn(features, labels, mode, params):
    filename = features['filename']
    shape = features['shape']
    loc_targets = features['loc_targets']
    cls_targets = features['cls_targets']
    match_scores = features['match_scores']
    features = features['image']
    global global_anchor_info
    decode_fn = global_anchor_info['decode_fn']
    num_anchors_per_layer = global_anchor_info['num_anchors_per_layer']
    all_num_anchors_depth = global_anchor_info['all_num_anchors_depth']
    with tf.variable_scope(params['model_scope'], default_name=None, values=[features], reuse=tf.AUTO_REUSE):
        backbone = ssd_net_resnet34_large.Resnet34Backbone(params['data_format'])
        feature_layers = backbone.forward(features, training=(mode == tf.estimator.ModeKeys.TRAIN))
        location_pred, cls_pred = ssd_net_resnet34_large.multibox_head(feature_layers, params['num_classes'], all_num_anchors_depth, data_format=params['data_format'], strides=(3, 3))
        if params['data_format'] == 'channels_last':
            cls_pred = [tf.transpose(pred, [0, 3, 1, 2]) for pred in cls_pred]
            location_pred = [tf.transpose(pred, [0, 3, 1, 2]) for pred in location_pred]

        cls_pred = [tf.reshape(pred, [tf.shape(features)[0], params['num_classes'], -1]) for pred in cls_pred]
        location_pred = [tf.reshape(pred, [tf.shape(features)[0], 4, -1]) for pred in location_pred]
        cls_pred = tf.concat(cls_pred, axis=2)
        location_pred = tf.concat(location_pred, axis=2)
        tf.identity(cls_pred, name='py_cls_pred')
        tf.identity(location_pred, name='py_location_pred')
        cls_pred = tf.transpose(cls_pred, [0, 2, 1])
        location_pred = tf.transpose(location_pred, [0, 2, 1])

    with tf.device('/cpu:0'):
        bboxes_pred = tf.map_fn(lambda _preds : decode_fn(_preds),
                                location_pred,
                                dtype=tf.float32, back_prop=False)
        #bboxes_pred = tf.concat(bboxes_pred, axis=1)
        parse_bboxes_fn = lambda x: parse_by_class_fixed_bboxes(x[0], x[1], params)
        pred_results = tf.map_fn(parse_bboxes_fn, (cls_pred, bboxes_pred), dtype=(tf.float32, tf.float32, tf.float32), back_prop=False)     
 
    predictions = {'filename': filename, 'shape': shape }
    detection_bboxes = tf.concat(pred_results[0], axis=0)
    detection_scores = tf.concat(pred_results[1], axis=0)
    detection_classes = tf.concat(pred_results[2], axis=0)
    xmin, ymin, xmax, ymax = tf.unstack(detection_bboxes, axis=-1)
    detection_bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    predictions['detection_classes'] = detection_classes
    predictions['detection_scores'] = detection_scores
    predictions['detection_bboxes'] = detection_bboxes
    tf.identity(detection_bboxes, name='detection_bboxes')
    tf.identity(detection_scores, name='detection_scores')
    tf.identity(detection_classes, name='detection_classes')
    if mode == tf.estimator.ModeKeys.PREDICT:
        est = tf.estimator.EstimatorSpec(mode=mode,
                                         predictions=predictions,
                                         prediction_hooks=None,
                                         loss=None, train_op=None)
        return est
    else:
        raise ValueError('This script only support "PREDICT" mode!')


def export_graph(args):
    with tf.Graph().as_default() as graph:
        out_shape = [args.train_image_size, args.train_image_size]
        defaultboxes_creator = anchor_manipulator.DefaultBoxes(out_shape,
                                                               layers_shapes = [(50, 50), (25, 25), (13, 13), (7, 7), (3, 3), (3, 3)],
                                                               anchor_scales = [(0.07,), (0.15,), (0.33,), (0.51,), (0.69,), (0.87,)],
                                                               extra_anchor_scales = [(0.15,), (0.33,), (0.51,), (0.69,), (0.87,), (1.05,)],
                                                               anchor_ratios = [(2,), ( 2., 3.,), (2., 3.,), (2., 3.,), (2.,), (2.,)],
                                                               layer_steps = [24, 48, 92, 171, 400, 400])
        defaultboxes, defaultboxes_ltrb, all_num_anchors_depth, all_num_anchors_spatial = defaultboxes_creator.get_all_anchors()
        num_anchors_per_layer = []
        for ind in range(len(all_num_anchors_depth)):
            num_anchors_per_layer.append(all_num_anchors_depth[ind] * all_num_anchors_spatial[ind])
        en_decoder = anchor_manipulator.En_Decoder(allowed_borders = [1.0] * 6,
                                                   positive_threshold = args.match_threshold,
                                                   ignore_threshold = args.neg_threshold,
                                                   prior_scaling=[0.1, 0.1, 0.2, 0.2])
        global global_anchor_info
        global_anchor_info = {'decode_fn': lambda pred : en_decoder.decode_all_anchors(pred, defaultboxes, num_anchors_per_layer),
                              'num_anchors_per_layer': num_anchors_per_layer,
                              'all_num_anchors_depth': all_num_anchors_depth}
        encoder_fn = lambda glabels, gbboxes: en_decoder.encode_all_anchors(glabels, gbboxes, defaultboxes_ltrb, all_num_anchors_depth, all_num_anchors_spatial)
        glabels = [1]
        glabels = tf.cast(glabels, tf.int64)
        gbboxes = [[10., 10., 200., 200.]] 
        gt_targets, gt_labels, gt_scores = encoder_fn(glabels, gbboxes)

        if args.data_format == "channels_first":
            image = tf.placeholder(name='image', dtype=tf.float32, shape=[None, 3, args.train_image_size, args.train_image_size])
        elif args.data_format == "channels_last":
            image = tf.placeholder(name='image', dtype=tf.float32, shape=[None, args.train_image_size, args.train_image_size, 3])
        filename = tf.placeholder(name='filename', dtype=tf.string, shape=[None,])
        shape = tf.placeholder(name='shape', dtype=tf.int32, shape=[None, 3])
        input_ = {'image': image, 'filename': filename, 'shape': shape, 'loc_targets': [gt_targets], 'cls_targets': [gt_labels], 'match_scores': [gt_scores]}
        ssd_model_fn(input_, None, tf.estimator.ModeKeys.PREDICT, { 'select_threshold': args.select_threshold,
                                                                    'min_size': args.min_size,
                                                                    'nms_threshold': args.nms_threshold,
                                                                    'nms_topk': args.nms_topk,
                                                                    'keep_topk': args.keep_topk,
                                                                    'data_format': args.data_format,
                                                                    'batch_size': args.batch_size_mine,
                                                                    'model_scope': args.model_scope,
                                                                    'save_summary_steps': args.save_summary_steps,
                                                                    'summary_dir': None,
                                                                    'num_classes': args.num_classes,
                                                                    'negative_ratio': args.negative_ratio,
                                                                    'match_threshold': args.match_threshold,
                                                                    'neg_threshold': args.neg_threshold,
                                                                    'weight_decay': args.weight_decay,
                                                                    'keep_max_boxes': args.keep_max_boxes})
 
        graph_def = graph.as_graph_def()
        with gfile.GFile(args.output_graph, 'w') as f:
            f.write(text_format.MessageToString(graph_def))
        print("Finish export inference graph")

def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = tf.app.flags.FLAGS  
    export_graph(args)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
