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

class SSDResnet50V1FpnConfig(object):
  def __init__(self, weights_path):
    self.pb_file = weights_path
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


