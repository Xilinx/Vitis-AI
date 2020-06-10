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

from easydict import EasyDict
import tensorflow as tf

class SSDMobilenetV2Config(object):
  def __init__(self, weights_path):
    self.pb_file = weights_path
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


