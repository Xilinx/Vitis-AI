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
# ==============================================================================
"""Parser functions for VAI_UTF."""

import enum
import copy

from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_layer_limits
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.vai_utf import conv2d_limits
from tensorflow_model_optimization.python.core.quantization.keras.vitis.vai_utf import depthwise_conv2d_limits
from tensorflow_model_optimization.python.core.quantization.keras.vitis.vai_utf import conv2d_transpose_limits
from tensorflow_model_optimization.python.core.quantization.keras.vitis.vai_utf import dense_limits
from tensorflow_model_optimization.python.core.quantization.keras.vitis.vai_utf import max_pool2d_limits
from tensorflow_model_optimization.python.core.quantization.keras.vitis.vai_utf import avg_pool2d_limits
from tensorflow_model_optimization.python.core.quantization.keras.vitis.vai_utf import global_avg_pool2d_limits
from tensorflow_model_optimization.python.core.quantization.keras.vitis.vai_utf import add_limits
from tensorflow_model_optimization.python.core.quantization.keras.vitis.vai_utf import multiply_limits

logger = common_utils.VAILogger


class VAIUTFParser(object):

  @classmethod
  def parse_legacy_dpu_target(cls, target):
    dpu_target = target.devices[0].legacy_dpu_target

    layer_limits_map = {}

    # Conv2d limits
    conv2d_limits_map = cls.parse_conv2d_limits(dpu_target)
    layer_limits_map.update(copy.deepcopy(conv2d_limits_map))
    # Pool2d limits
    pool2d_limits_map = cls.parse_pool2d_limits(dpu_target)
    layer_limits_map.update(copy.deepcopy(pool2d_limits_map))
    # eltwise limits
    eltwise_limits_map = cls.parse_eltwise_limits(dpu_target)
    layer_limits_map.update(copy.deepcopy(eltwise_limits_map))
    return layer_limits_map

  @classmethod
  def parse_conv2d_limits(cls, dpu_target):
    conv2d_limits_ins = conv2d_limits.Conv2DLimits(dpu_target)
    depthwise_conv2d_limits_ins = depthwise_conv2d_limits.DepthwiseConv2DLimits(
        dpu_target)
    conv2d_transpose_limits_ins = conv2d_transpose_limits.Conv2DTransposeLimits(
        dpu_target)
    dense_limits_ins = dense_limits.DenseLimits(dpu_target)

    conv2d_limits_map = {
        'tensorflow.keras.layers.Conv2D': conv2d_limits_ins,
        'tensorflow.keras.layers.DepthwiseConv2D': depthwise_conv2d_limits_ins,
        'tensorflow.keras.layers.Conv2DTranspose': conv2d_transpose_limits_ins,
        'tensorflow.keras.layers.Dense': dense_limits_ins
    }
    return conv2d_limits_map

  @classmethod
  def parse_pool2d_limits(cls, dpu_target):
    avg_pool2d_limits_ins = avg_pool2d_limits.AvgPool2DLimits(dpu_target)
    global_avg_pool2d_limits_ins = global_avg_pool2d_limits.GlobalAvgPool2DLimits(
        dpu_target)
    max_pool2d_limits_ins = max_pool2d_limits.MaxPool2DLimits(dpu_target)

    pool2d_limits_map = {
        'tensorflow.keras.layers.MaxPooling2D':
            max_pool2d_limits_ins,
        'tensorflow.keras.layers.AveragePooling2D':
            avg_pool2d_limits_ins,
        'tensorflow.keras.layers.GlobalAveragePooling2D':
            global_avg_pool2d_limits_ins,
        'tensorflow_model_optimization.python.core.quantization.keras.vitis.layers.vitis_pooling.VitisAveragePooling2D':
            avg_pool2d_limits_ins,
        'tensorflow_model_optimization.python.core.quantization.keras.vitis.layers.vitis_pooling.VitisGlobalAveragePooling2D':
            global_avg_pool2d_limits_ins
    }
    return pool2d_limits_map

  @classmethod
  def parse_eltwise_limits(cls, dpu_target):
    add_limits_ins = add_limits.AddLimits(dpu_target)
    multiply_limits_ins = multiply_limits.MultiplyLimits(dpu_target)

    eltwise_limits_map = {
        'tensorflow.keras.layers.Add': add_limits_ins,
        'tensorflow.keras.layers.Multiply': multiply_limits_ins
    }
    return eltwise_limits_map
