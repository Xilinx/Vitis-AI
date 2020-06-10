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

import os
import sys
import cv2
import json
import functools
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from model_utils.preprocess import PREPROCESS_FUNC
from model_utils.anchor import BUILD_ANCHOR_FUNC
from model_utils.decoder import batch_decode
from model_utils.postprocess import batch_multiclass_non_max_suppression
from model_utils.postprocess import compute_clip_window
from model_utils.postprocess import score_converter_fn_with_logit_scale
from dataset_tools.cocoval import cocoval
from dataset_tools.json_encoder import MyEncoder

from configs.model_config import SSDMobilenetV1Config as Config


class SSD(object):
  ''' Construct the SSD model trained on COCO
  '''
  def __init__(self, params):
    self.frozon_graph = params.pb_file
    self.new_height = params.height
    self.new_width = params.width
    self.preprocess_fn = PREPROCESS_FUNC[params.feature_extractor_type]
    self.build_anchor_fn = BUILD_ANCHOR_FUNC[params.anchor_type]
    self.anchor_config = params.anchor_config
    self.nms_config = params.nms_config
    self.score_fn = params.score_fn
    self.logit_scale = params.logit_scale
    self.input_tensor = params.input_tensor
    self.box_encoding_tensor = params.box_encoding_tensor
    self.class_score_tensor = params.class_score_tensor
    self.scale_factors = params.scale_factors
    self.feature_map_spatial_dims = params.feature_map_spatial_dims

    self.graph = tf.Graph()
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(self.frozon_graph, 'rb') as fid:
      graph_def.ParseFromString(fid.read())
    with self.graph.as_default():
      raw_image = tf.placeholder(tf.float32, shape=(None, None, None, 3), name="raw_image")
      preprocessed_image = self.preprocess(raw_image)
      tf.import_graph_def(graph_def, name='', input_map={self.input_tensor: preprocessed_image})
      self.anchors = self.build_anchors()
      prediction_dict = { "box_encodings": tf.get_default_graph().get_tensor_by_name(self.box_encoding_tensor),
                          "class_predictions_with_background": tf.get_default_graph().get_tensor_by_name(self.class_score_tensor),
                          "anchors": self.anchors}
      self.detections = self.postprocess(prediction_dict)

    self.sess = tf.Session(graph=self.graph)

  def run_single_image(self, image):
    ''' Conduct preprocess, network process and postprocess for detection
    Args:
        image: image of shape [height, width, 3] with RGB channel order,
               pixel value in [0.0, 255.0]
    '''
    image_tensor = self.graph.get_tensor_by_name("raw_image:0")
    feed_dict = {image_tensor: np.expand_dims(image, 0)}
    out_dict = self.detections
    pred_dict = self.sess.run(out_dict, feed_dict=feed_dict)
    return pred_dict

  def preprocess(self, image):
    return self.preprocess_fn(image, self.new_height, self.new_width)

  def build_anchors(self):
    return self.build_anchor_fn(self.anchor_config, self.feature_map_spatial_dims, im_height=self.new_height, im_width=self.new_width)

  def postprocess(self, prediction_dict):
    ''' Decode box_encodings and class_predictions_with_background, apply multi-class NMS
    Args:
        prediction_dict: dict with these keys - "box_encodings", "class_predictions_with_background", "anchors"
    '''
    with tf.name_scope('Postprocessor'):
      box_encodings = prediction_dict['box_encodings']
      class_predictions = prediction_dict['class_predictions_with_background']
      anchors = prediction_dict['anchors']

      # decode bounding box
      detection_boxes = batch_decode(box_encodings, anchors, self.scale_factors)
      detection_boxes = tf.expand_dims(detection_boxes, axis=2)

      # map score with conversion_fn
      score_conversion_fn = score_converter_fn_with_logit_scale(self.score_fn, self.logit_scale)
      detection_scores = score_conversion_fn(class_predictions)
      detection_scores = tf.slice(detection_scores, [0, 0, 1], [-1, -1, -1])

      # multi-class nms
      resized_inputs_shape = [1, self.new_height, self.new_height, 3]
      true_image_shapes = np.array([[self.new_height, self.new_width, 3]])
      non_max_suppression_fn = functools.partial(
        batch_multiclass_non_max_suppression,
        score_thresh=self.nms_config.score_threshold,
        iou_thresh=self.nms_config.iou_threshold,
        max_size_per_class=self.nms_config.max_detections_per_class,
        max_total_size=self.nms_config.max_total_detections)
      (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks, nmsed_additional_fields, num_detections) = non_max_suppression_fn(
        detection_boxes,
        detection_scores,
        clip_window=compute_clip_window(resized_inputs_shape, true_image_shapes))

      label_id_offset = 1
      nmsed_classes += label_id_offset

      detection_dict = {
        "detection_boxes": nmsed_boxes,
        "detection_scores": nmsed_scores,
        "detection_classes": nmsed_classes,
        "num_detections": tf.to_float(num_detections),
      }
      return detection_dict

#######################################################

def run_SSD_for_eval(model, image_root, image_list_file):

  with open(image_list_file, 'r') as f_image:
    image_lines = f_image.readlines()
  coco_records = []
  for image_line in tqdm(image_lines):
    image_name = image_line.strip()
    image_path = os.path.join(image_root, image_name + ".jpg")
    image = cv2.imread(image_path)
    height, width = image.shape[0:2]
    image = image[:,:,::-1] # BGR to RGB
    image = np.array(image, dtype=np.float32)

    output_dict = model.run_single_image(image)

    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    for i in range(output_dict['detection_classes'].shape[0]):
      record = {}
      ymin = output_dict['detection_boxes'][i][0] * height
      xmin = output_dict['detection_boxes'][i][1] * width
      ymax = output_dict['detection_boxes'][i][2] * height
      xmax = output_dict['detection_boxes'][i][3] * width
      score = output_dict['detection_scores'][i]
      class_id = output_dict['detection_classes'][i]
      record['image_id'] = int(image_name.split('_')[-1])
      record['category_id'] = class_id
      record['score'] = score
      record['bbox'] = [xmin, ymin, xmax - xmin + 1, ymax - ymin + 1]
      # if score < 0.005:
      #   break
      coco_records.append(record)
  return coco_records


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Script for predicting the detection results')
  parser.add_argument('-weights', default='float/float.pb')
  parser.add_argument('-image_list', default='data/coco2014_minival_8059/minival2014_8059.txt')
  parser.add_argument('-image_root', default='data/coco2014_minival_8059/image')
  parser.add_argument('-gt_json', default='data/coco2014_minival_8059/minival2014_8059.json')
  parser.add_argument('-dt_json', default='data/ssd_prediction.json')
  args = parser.parse_args()


  params = Config(args.weights)
  model = SSD(params)
  coco_records = run_SSD_for_eval(model, args.image_root, args.image_list)
  with open(args.dt_json, 'w') as f_det:
    f_det.write(json.dumps(coco_records, cls=MyEncoder))
  cocoval(args.dt_json, args.gt_json)
