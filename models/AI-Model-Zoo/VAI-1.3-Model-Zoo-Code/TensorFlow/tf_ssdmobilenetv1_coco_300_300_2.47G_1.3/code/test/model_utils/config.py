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


from easydict import EasyDict
import tensorflow as tf

class SSDInceptionV2Config(object):
  def __init__(self):
    self.height = 300 
    self.width = 300 
    self.input_tensor = "image_tensor:0"
    self.box_encoding_tensor = "concat:0"
    self.class_score_tensor = "concat_1:0"

    self.feature_extractor_type = "ssd_inception_v2"

    self.anchor_type = "ssd_anchor_generator"
    self.anchor_config = EasyDict()
    self.anchor_config.num_layers = 6 
    self.anchor_config.min_scale = 0.2 
    self.anchor_config.max_scale = 0.95
    self.anchor_config.scales = []
    self.anchor_config.aspect_ratios = [1.0, 2.0, 0.5, 3.0, 0.3333]
    self.anchor_config.interpolated_scale_aspect_ratio=1.0
    self.anchor_config.base_anchor_size=[1.0, 1.0]
    self.anchor_config.anchor_strides=None
    self.anchor_config.anchor_offsets=None
    self.anchor_config.reduce_boxes_in_lowest_layer=True

    self.scale_factors = [10.0, 10.0, 5.0, 5.0]
    self.feature_map_spatial_dims = [(19, 19), (10, 10), (5, 5), (3, 3), (2, 2), (1, 1)] 

    self.nms_config = EasyDict()
    self.nms_config.score_threshold = 0.005 
    self.nms_config.iou_threshold = 0.6 
    self.nms_config.max_detections_per_class = 100 
    self.nms_config.max_total_detections = 100 

    self.score_fn = tf.sigmoid
    self.logit_scale = 1.0 


class SSDMobilenetV1Config(object):
  def __init__(self):
    self.height = 300 
    self.width = 300 
    self.input_tensor = "image_tensor:0"
    self.box_encoding_tensor = "concat:0"
    self.class_score_tensor = "concat_1:0"

    self.feature_extractor_type = "ssd_mobilenet_v1"

    self.anchor_type = "ssd_anchor_generator"
    self.anchor_config = EasyDict()
    self.anchor_config.num_layers = 6 
    self.anchor_config.min_scale = 0.2 
    self.anchor_config.max_scale = 0.95
    self.anchor_config.scales = []
    self.anchor_config.aspect_ratios = [1.0, 2.0, 0.5, 3.0, 0.3333]
    self.anchor_config.interpolated_scale_aspect_ratio=1.0
    self.anchor_config.base_anchor_size=[1.0, 1.0]
    self.anchor_config.anchor_strides=None
    self.anchor_config.anchor_offsets=None
    self.anchor_config.reduce_boxes_in_lowest_layer=True

    self.scale_factors = [10.0, 10.0, 5.0, 5.0]
    self.feature_map_spatial_dims = [(19, 19), (10, 10), (5, 5), (3, 3), (2, 2), (1, 1)] 

    self.nms_config = EasyDict()
    self.nms_config.score_threshold = 0.005 
    self.nms_config.iou_threshold = 0.6 
    self.nms_config.max_detections_per_class = 100 
    self.nms_config.max_total_detections = 100 

    self.score_fn = tf.sigmoid
    self.logit_scale = 1.0 


class SSDMobilenetV2Config(object):
  def __init__(self):
    self.height = 300 
    self.width = 300 
    self.input_tensor = "image_tensor:0"
    self.box_encoding_tensor = "concat:0"
    self.class_score_tensor = "concat_1:0"

    self.feature_extractor_type = "ssd_mobilenet_v2"

    self.anchor_type = "ssd_anchor_generator"
    self.anchor_config = EasyDict()
    self.anchor_config.num_layers = 6 
    self.anchor_config.min_scale = 0.2 
    self.anchor_config.max_scale = 0.95
    self.anchor_config.scales = []
    self.anchor_config.aspect_ratios = [1.0, 2.0, 0.5, 3.0, 0.3333]
    self.anchor_config.interpolated_scale_aspect_ratio=1.0
    self.anchor_config.base_anchor_size=[1.0, 1.0]
    self.anchor_config.anchor_strides=None
    self.anchor_config.anchor_offsets=None
    self.anchor_config.reduce_boxes_in_lowest_layer=True

    self.scale_factors = [10.0, 10.0, 5.0, 5.0]
    self.feature_map_spatial_dims = [(19, 19), (10, 10), (5, 5), (3, 3), (2, 2), (1, 1)] 

    self.nms_config = EasyDict()
    self.nms_config.score_threshold = 0.005 
    self.nms_config.iou_threshold = 0.6 
    self.nms_config.max_detections_per_class = 100 
    self.nms_config.max_total_detections = 100 

    self.score_fn = tf.sigmoid
    self.logit_scale = 1.0 


class SSDliteMobilenetV2Config(object):
  def __init__(self):
    self.height = 300 
    self.width = 300 
    self.input_tensor = "image_tensor:0"
    self.box_encoding_tensor = "concat:0"
    self.class_score_tensor = "concat_1:0"

    self.feature_extractor_type = "ssd_mobilenet_v2"

    self.anchor_type = "ssd_anchor_generator"
    self.anchor_config = EasyDict()
    self.anchor_config.num_layers = 6 
    self.anchor_config.min_scale = 0.2 
    self.anchor_config.max_scale = 0.95
    self.anchor_config.scales = []
    self.anchor_config.aspect_ratios = [1.0, 2.0, 0.5, 3.0, 0.3333]
    self.anchor_config.interpolated_scale_aspect_ratio=1.0
    self.anchor_config.base_anchor_size=[1.0, 1.0]
    self.anchor_config.anchor_strides=None
    self.anchor_config.anchor_offsets=None
    self.anchor_config.reduce_boxes_in_lowest_layer=True

    self.scale_factors = [10.0, 10.0, 5.0, 5.0]
    self.feature_map_spatial_dims = [(19, 19), (10, 10), (5, 5), (3, 3), (2, 2), (1, 1)] 

    self.nms_config = EasyDict()
    self.nms_config.score_threshold = 0.005
    self.nms_config.iou_threshold = 0.6 
    self.nms_config.max_detections_per_class = 100 
    self.nms_config.max_total_detections = 100 

    self.score_fn = tf.sigmoid
    self.logit_scale = 1.0 


class SSDMobilenetV1FpnConfig(object):
  def __init__(self):
    self.height = 640 
    self.width = 640 
    self.input_tensor = "image_tensor:0"
    self.box_encoding_tensor = "concat:0"
    self.class_score_tensor = "concat_1:0"

    self.feature_extractor_type = "ssd_mobilenet_v1_fpn"

    self.anchor_type = "multiscale_anchor_generator"
    self.anchor_config = EasyDict()
    self.anchor_config.min_level = 3 
    self.anchor_config.max_level = 7 
    self.anchor_config.anchor_scale = 4.0 
    self.anchor_config.aspect_ratios = [1.0, 2.0, 0.5]
    self.anchor_config.scales_per_octave = 2 
    self.anchor_config.normalize_coordinates = True

    self.scale_factors = [10.0, 10.0, 5.0, 5.0]
    self.feature_map_spatial_dims = [(80, 80), (40, 40), (20, 20), (10, 10), (5, 5)] 

    self.nms_config = EasyDict()
    self.nms_config.score_threshold = 0.005
    self.nms_config.iou_threshold = 0.6 
    self.nms_config.max_detections_per_class = 100 
    self.nms_config.max_total_detections = 100 

    self.score_fn = tf.sigmoid
    self.logit_scale = 1.0 


class SSDResnet50V1FpnConfig(object):
  def __init__(self):
    self.height = 640
    self.width = 640
    self.input_tensor = "image_tensor:0"
    self.box_encoding_tensor = "concat:0"
    self.class_score_tensor = "concat_1:0"

    self.feature_extractor_type = "ssd_resnet50_v1_fpn"

    self.anchor_type = "multiscale_anchor_generator"
    self.anchor_config = EasyDict()
    self.anchor_config.min_level = 3
    self.anchor_config.max_level = 7
    self.anchor_config.anchor_scale = 4.0
    self.anchor_config.aspect_ratios = [1.0, 2.0, 0.5]
    self.anchor_config.scales_per_octave = 2
    self.anchor_config.normalize_coordinates = True

    self.scale_factors = [10.0, 10.0, 5.0, 5.0]
    self.feature_map_spatial_dims = [(80, 80), (40, 40), (20, 20), (10, 10), (5, 5)]

    self.nms_config = EasyDict()
    self.nms_config.score_threshold = 0.005
    self.nms_config.iou_threshold = 0.6
    self.nms_config.max_detections_per_class = 100
    self.nms_config.max_total_detections = 100

    self.score_fn = tf.sigmoid
    self.logit_scale = 1.0


CONFIG_MAP = {"ssd_inception_v2": SSDInceptionV2Config(),
              "ssd_mobilenet_v2": SSDMobilenetV2Config(),
              "ssd_mobilenet_v1": SSDMobilenetV1Config(),
              "ssdlite_mobilenet_v2": SSDliteMobilenetV2Config(),
              "ssd_mobilenet_v1_fpn": SSDMobilenetV1FpnConfig(),
              "ssd_resnet50_v1_fpn": SSDResnet50V1FpnConfig(),
             }
