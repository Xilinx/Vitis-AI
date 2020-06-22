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
from google.protobuf import text_format
from tensorflow.python.ops import gen_image_ops
tf.image.non_max_suppression = gen_image_ops.non_max_suppression_v2

from model import ssd_net_resnet34_large
from dataset import dataset_common
from utils import ssd_preprocessing
from utils import anchor_manipulator
from utils import scaffolds
from utils.coco_eval import eval_on_coco

tf.app.flags.DEFINE_integer(
    'num_readers', 8,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads_mine', 24,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
    'num_cpu_threads', 2,
    'The number of cpu cores used to train.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1., 'GPU memory fraction to use.')
tf.app.flags.DEFINE_string(
    'data_dir', '../../data/tfrecords/',
    'The directory where the dataset input data is stored.')
tf.app.flags.DEFINE_integer(
    'num_classes', 81, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'model_dir', './logs/',
    'The directory where the model will be stored.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are printed.')
tf.app.flags.DEFINE_integer(
    'save_summary_steps', 500,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'train_image_size', 1200,
    'The size of the input image for the model to use.')
tf.app.flags.DEFINE_integer(
    'train_epochs', 1,
    'The number of epochs to use for training.')
tf.app.flags.DEFINE_integer(
    'batch_size_mine', 4,
    'Batch size for training and evaluation.')
tf.app.flags.DEFINE_string(
    'data_format', 'channels_first',
    'A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible '
    'with CPU. If left unspecified, the data format will be chosen '
    'automatically based on whether TensorFlow was built for CPU or GPU.')
tf.app.flags.DEFINE_float(
    'negative_ratio', 3., 'Negative ratio in the loss function.')
tf.app.flags.DEFINE_float(
    'match_threshold', 0.5, 'Matching threshold in the loss function.')
tf.app.flags.DEFINE_float(
    'neg_threshold', 0.5, 'Matching threshold for the negtive examples in the loss function.')
tf.app.flags.DEFINE_float(
    'select_threshold', 0.05, 'Class-specific confidence score threshold for selecting a box.')
tf.app.flags.DEFINE_float(
    'min_size', 0.003, 'The min size of bboxes to keep.')
tf.app.flags.DEFINE_float(
    'nms_threshold', 0.5, 'Matching threshold in NMS algorithm.')
tf.app.flags.DEFINE_integer(
    'nms_topk', 200, 'Number of total object to keep after NMS.')
tf.app.flags.DEFINE_integer(
    'keep_topk', 200, 'Number of total object to keep for each image before nms.')
tf.app.flags.DEFINE_integer(
    'keep_max_boxes', 200, 'Max number of total prediect bboxes to keep for each image after nms.')
tf.app.flags.DEFINE_float(
    'weight_decay', 5e-4, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', './model',
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'model_scope', 'ssd1200',
    'Model scope name used to replace the name_scope in checkpoint.')  

def get_checkpoint(args):
    if tf.train.latest_checkpoint(args.model_dir):
        tf.logging.info('Ignoring --checkpoint_path because a checkpoint already exists in %s' % args.model_dir)
        return None
    if tf.gfile.IsDirectory(args.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(args.checkpoint_path)
    else:
        checkpoint_path = args.checkpoint_path
    return checkpoint_path

global_anchor_info = dict()
def input_pipeline(args, dataset_pattern='train-*', is_training=True):
    batch_size = args.batch_size_mine
    def input_fn():
        out_shape = [args.train_image_size] * 2
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
        image_preprocessing_fn = lambda image_, labels_, bboxes_ : ssd_preprocessing.preprocess_image(image_, labels_, bboxes_, out_shape, is_training=is_training, data_format=args.data_format, output_rgb=True)
        encoder_fn = lambda glabels_, gbboxes_: en_decoder.encode_all_anchors(glabels_, gbboxes_, defaultboxes_ltrb, all_num_anchors_depth, all_num_anchors_spatial)
        image, filename, shape, loc_targets, cls_targets, match_scores = dataset_common.slim_get_batch(args.num_classes,
                                                                                                       batch_size,
                                                                                                       ('train' if is_training else 'val'),
                                                                                                       os.path.join(args.data_dir, dataset_pattern),
                                                                                                       args.num_readers,
                                                                                                       args.num_preprocessing_threads_mine,
                                                                                                       image_preprocessing_fn,
                                                                                                       encoder_fn,
                                                                                                       num_epochs=args.train_epochs,
                                                                                                       is_training=is_training)
        global global_anchor_info
        global_anchor_info = {'decode_fn': lambda pred : en_decoder.decode_all_anchors(pred, defaultboxes, num_anchors_per_layer),
                              'num_anchors_per_layer': num_anchors_per_layer,
                              'all_num_anchors_depth': all_num_anchors_depth }
        return {'image': image, 'filename': filename, 'shape': shape, 'loc_targets': loc_targets, 'cls_targets': cls_targets, 'match_scores': match_scores}, None
    return input_fn

def modified_smooth_l1(bbox_pred, bbox_targets, bbox_inside_weights=1., bbox_outside_weights=1., sigma=1.):
    with tf.name_scope('smooth_l1', [bbox_pred, bbox_targets]):
        sigma2 = sigma * sigma
        inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))
        smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))
        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)
        return outside_mul

def select_bboxes(scores_pred, bboxes_pred, num_classes, select_threshold):
    selected_bboxes = {}
    selected_scores = {}
    with tf.name_scope('select_bboxes', [scores_pred, bboxes_pred]):
        for class_ind in range(1, num_classes):
            class_scores = scores_pred[:, class_ind]
            select_mask = class_scores > select_threshold
            select_mask = tf.cast(select_mask, tf.float32)
            selected_bboxes[class_ind] = tf.multiply(bboxes_pred, tf.expand_dims(select_mask, axis=-1))
            selected_scores[class_ind] = tf.multiply(class_scores, select_mask)
    return selected_bboxes, selected_scores

def clip_bboxes(xmin, ymin, xmax, ymax, name):
    with tf.name_scope(name, 'clip_bboxes', [xmin, ymin, xmax, ymax]):
        xmin = tf.maximum(xmin, 0.)
        ymin = tf.maximum(ymin, 0.)
        xmax = tf.minimum(xmax, 1.)
        ymax = tf.minimum(ymax, 1.)
        xmin = tf.minimum(xmin, xmax)
        ymin = tf.minimum(ymin, ymax)
        return xmin, ymin, xmax, ymax

def filter_bboxes(scores_pred, xmin, ymin, xmax, ymax, min_size, name):
    with tf.name_scope(name, 'filter_bboxes', [scores_pred, xmin, ymin, xmax, ymax]):
        width = xmax - xmin
        height = ymax - ymin
        filter_mask = tf.logical_and(width > min_size, height > min_size)
        filter_mask = tf.cast(filter_mask, tf.float32)
        return tf.multiply(xmin, filter_mask), tf.multiply(ymin, filter_mask),\
               tf.multiply(xmax, filter_mask), tf.multiply(ymax, filter_mask), tf.multiply(scores_pred, filter_mask)

def sort_bboxes(scores_pred, xmin, ymin, xmax, ymax, keep_topk, name):
    with tf.name_scope(name, 'sort_bboxes', [scores_pred, xmin, ymin, xmax, ymax]):
        cur_bboxes = tf.shape(scores_pred)[0]
        scores, idxes = tf.nn.top_k(scores_pred, k=tf.minimum(keep_topk, cur_bboxes), sorted=True)
        xmin, ymin, xmax, ymax = tf.gather(xmin, idxes), tf.gather(ymin, idxes), tf.gather(xmax, idxes), tf.gather(ymax, idxes)
        paddings_scores = tf.expand_dims(tf.stack([0, tf.maximum(keep_topk-cur_bboxes, 0)], axis=0), axis=0)
        return tf.pad(xmin, paddings_scores, "CONSTANT"), tf.pad(ymin, paddings_scores, "CONSTANT"),\
               tf.pad(xmax, paddings_scores, "CONSTANT"), tf.pad(ymax, paddings_scores, "CONSTANT"),\
               tf.pad(scores, paddings_scores, "CONSTANT")

def nms_bboxes(scores_pred, bboxes_pred, nms_topk, nms_threshold, name):
    with tf.name_scope(name, 'nms_bboxes', [scores_pred, bboxes_pred]):
        xmin, ymin, xmax, ymax = tf.unstack(bboxes_pred, 4, axis=-1)
        bboxes_pred = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
        idxes = tf.image.non_max_suppression(bboxes_pred, scores_pred, nms_topk, nms_threshold)
        ymin, xmin, ymax, xmax = tf.unstack(bboxes_pred, 4, axis=-1)
        bboxes_pred = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
        return tf.gather(scores_pred, idxes), tf.gather(bboxes_pred, idxes)

def parse_by_class_fixed_bboxes(cls_pred, bboxes_pred, params):
    selected_bboxes, selected_scores = parse_by_class(cls_pred, bboxes_pred, params['num_classes'], params['select_threshold'], params['min_size'], params['keep_topk'], params['nms_topk'], params['nms_threshold'])
    pred_bboxes = []
    pred_scores = []
    pred_classes = []
    predictions = {}
    for class_ind in range(1, params['num_classes']):
        predictions['scores_{}'.format(class_ind)] = tf.expand_dims(selected_scores[class_ind], axis=0)
        predictions['bboxes_{}'.format(class_ind)] = tf.expand_dims(selected_bboxes[class_ind], axis=0)
        labels_mask = selected_scores[class_ind] > -0.5
        labels_mask = tf.cast(labels_mask, tf.float32)
        selected_labels = labels_mask * class_ind
        pred_bboxes.append(selected_bboxes[class_ind])
        pred_scores.append(selected_scores[class_ind])
        pred_classes.append(selected_labels)
    detection_bboxes = tf.concat(pred_bboxes, axis=0)
    detection_scores = tf.concat(pred_scores, axis=0)
    detection_classes = tf.concat(pred_classes, axis=0)
    num_bboxes = tf.shape(detection_bboxes)[0] 
    detection_scores, idxes = tf.nn.top_k(detection_scores, k=tf.minimum(params['keep_max_boxes'], num_bboxes), sorted=True)
    detection_bboxes = tf.gather(detection_bboxes, idxes)
    detection_classes = tf.gather(detection_classes, idxes)
    keep_max_boxes = tf.convert_to_tensor(params['keep_max_boxes']) 
    cur_num = tf.shape(detection_classes)[0]
    detection_bboxes = tf.cond(cur_num < keep_max_boxes, lambda: tf.concat([detection_bboxes, tf.zeros(shape=(params['keep_max_boxes'] - cur_num, 4), dtype=tf.float32)], axis=0), lambda: detection_bboxes)
    detection_scores = tf.cond(cur_num < keep_max_boxes, lambda: tf.concat([detection_scores, tf.zeros(shape=(params['keep_max_boxes'] - cur_num,), dtype=tf.float32)], axis=0), lambda: detection_scores)
    detection_classes = tf.cond(cur_num < keep_max_boxes, lambda: tf.concat([detection_classes, tf.zeros(shape=(params['keep_max_boxes'] - cur_num,), dtype=tf.float32)], axis=0), lambda: detection_classes)   
    return detection_bboxes, detection_scores, detection_classes

def parse_by_class(cls_pred, bboxes_pred, num_classes, select_threshold, min_size, keep_topk, nms_topk, nms_threshold):
    with tf.name_scope('select_bboxes', [cls_pred, bboxes_pred]):
        scores_pred = tf.nn.softmax(cls_pred)
        selected_bboxes, selected_scores = select_bboxes(scores_pred, bboxes_pred, num_classes, select_threshold)
        for class_ind in range(1, num_classes):
            xmin, ymin, xmax, ymax = tf.unstack(selected_bboxes[class_ind], 4, axis=-1)
            #ymin, xmin, ymax, xmax = clip_bboxes(ymin, xmin, ymax, xmax, 'clip_bboxes_{}'.format(class_ind))
            #ymin, xmin, ymax, xmax, selected_scores[class_ind] = filter_bboxes(selected_scores[class_ind],
            #                                    ymin, xmin, ymax, xmax, min_size, 'filter_bboxes_{}'.format(class_ind))
            xmin, ymin, xmax, ymax, selected_scores[class_ind] = sort_bboxes(selected_scores[class_ind],
                                                                             xmin, ymin, xmax, ymax, keep_topk, 'sort_bboxes_{}'.format(class_ind))
            selected_bboxes[class_ind] = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
            selected_scores[class_ind], selected_bboxes[class_ind] = nms_bboxes(selected_scores[class_ind], selected_bboxes[class_ind], nms_topk, nms_threshold, 'nms_bboxes_{}'.format(class_ind))
        return selected_bboxes, selected_scores

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
    predictions['detection_classes'] = detection_classes
    predictions['detection_scores'] = detection_scores
    predictions['detection_bboxes'] = detection_bboxes
    tf.identity(detection_bboxes, name='detection_bboxes')
    tf.identity(detection_scores, name='detection_scores')
    tf.identity(detection_classes, name='detection_classes')
    tf.identity(tf.shape(features)[0], name='eval_images_per_bacth')  
    tf.summary.scalar('eval_images', params['batch_size'])
    summary_hook = tf.train.SummarySaverHook(save_steps=params['save_summary_steps'],
                                             output_dir=params['summary_dir'],
                                             summary_op=tf.summary.merge_all())
    if mode == tf.estimator.ModeKeys.PREDICT:
        est = tf.estimator.EstimatorSpec(mode=mode,
                                         predictions=predictions,
                                         prediction_hooks=None,
                                         loss=None, train_op=None)
        return est
    else:
        raise ValueError('This script only support "PREDICT" mode!')

def parse_comma_list(args):
    return [float(s.strip()) for s in args.split(',')]

def evaluate(args):
    checkpoint_path = get_checkpoint(args)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, intra_op_parallelism_threads=args.num_cpu_threads, inter_op_parallelism_threads=args.num_cpu_threads, gpu_options=gpu_options)
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=None).replace(
                                                  save_checkpoints_steps=None).replace(
                                                  save_summary_steps=args.save_summary_steps).replace(
                                                  keep_checkpoint_max=5).replace(
                                                  log_step_count_steps=args.log_every_n_steps).replace(
                                                  session_config=config)
    summary_dir = os.path.join(args.model_dir, 'predict')
    ssd_detector = tf.estimator.Estimator(
        model_fn=ssd_model_fn, model_dir=args.model_dir, config=run_config,
        params={'select_threshold': args.select_threshold,
                'min_size': args.min_size,
                'nms_threshold': args.nms_threshold,
                'nms_topk': args.nms_topk,
                'keep_topk': args.keep_topk,
                'data_format': args.data_format,
                'batch_size': args.batch_size_mine,
                'model_scope': args.model_scope,
                'save_summary_steps': args.save_summary_steps,
                'summary_dir': summary_dir,
                'num_classes': args.num_classes,
                'negative_ratio': args.negative_ratio,
                'match_threshold': args.match_threshold,
                'neg_threshold': args.neg_threshold,
                'weight_decay': args.weight_decay,
                'keep_max_boxes': args.keep_max_boxes,})
    tensors_to_log = {'eval_images_per_bacth': 'eval_images_per_bacth',}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=args.log_every_n_steps,
                                              formatter=lambda dicts: (', '.join(['%s=%s' % (k, v) for k, v in dicts.items()])))
    print('Starting a predict cycle.')
    pred_results = ssd_detector.predict(input_fn=input_pipeline(args, dataset_pattern='coco_2017_val-*', is_training=False), hooks=[logging_hook], checkpoint_path=checkpoint_path)
    det_results = list(pred_results)
    predict_by_class_dict = {}
    for image_ind, pred in enumerate(det_results):
        filename = pred['filename']
        shape = pred['shape']
        scores = pred['detection_scores']
        bboxes = pred['detection_bboxes']
        labels = pred['detection_classes']
        bboxes[:, 0] = (bboxes[:, 0] * shape[1]).astype(np.float32, copy=False)
        bboxes[:, 1] = (bboxes[:, 1] * shape[0]).astype(np.float32, copy=False)
        bboxes[:, 2] = (bboxes[:, 2] * shape[1]).astype(np.float32, copy=False)
        bboxes[:, 3] = (bboxes[:, 3] * shape[0]).astype(np.float32, copy=False)
        valid_mask = np.logical_and((bboxes[:, 2] - bboxes[:, 0] > 0), (bboxes[:, 3] - bboxes[:, 1] > 0))
        for det_ind in range(valid_mask.shape[0]):
            if not valid_mask[det_ind]:
                continue
            class_ind = int(labels[det_ind])  
            if class_ind in predict_by_class_dict.keys():
                predict_by_class_dict[class_ind].append([filename.decode('utf8')[:-4], scores[det_ind],
                                                         bboxes[det_ind, 0], bboxes[det_ind, 1],
                                                         bboxes[det_ind, 2], bboxes[det_ind, 3]])
            else:
                predict_by_class_dict[class_ind] = [[filename.decode('utf8')[:-4], scores[det_ind],
                                                     bboxes[det_ind, 0], bboxes[det_ind, 1],
                                                     bboxes[det_ind, 2], bboxes[det_ind, 3]]] 
    return eval_on_coco(predict_by_class_dict)

def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = tf.app.flags.FLAGS
    evaluate(args)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
