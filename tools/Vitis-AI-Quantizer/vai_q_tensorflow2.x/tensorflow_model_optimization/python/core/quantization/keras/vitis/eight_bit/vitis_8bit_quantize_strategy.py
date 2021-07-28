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
"""Vitis 8-Bit Quantization Strategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras.vitis.eight_bit import vitis_8bit_transforms_pipeline
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_strategy
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_registry
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

VitisQuantizeStrategy = vitis_quantize_strategy.VitisQuantizeStrategy
VitisQuantizeRegistry = vitis_quantize_registry.VitisQuantizeRegistry
Vitis8BitOptimizeTransformsPipeline = vitis_8bit_transforms_pipeline.Vitis8BitOptimizeTransformsPipeline
Vitis8BitQuantizeTransformsPipeline = vitis_8bit_transforms_pipeline.Vitis8BitQuantizeTransformsPipeline
logger = common_utils.VAILogger

# vitis 8bit quantize strategy config
dirname = os.path.dirname(__file__)
vitis_8bit_qs_config = os.path.join(dirname,
                                    "vitis_8bit_quantize_strategy.json")


class Vitis8BitQuantizeStrategy(VitisQuantizeStrategy):
  """Vitis 8-Bit Quantize Strategy."""

  def __init__(self, qs_configs=vitis_8bit_qs_config):
    """Init

    Args:
      qs_config: A json file contains the quantize strategy configurations.
    """
    self._qs_configs = common_utils.load_json(qs_configs)

    self._optimize_pipeline = Vitis8BitOptimizeTransformsPipeline(
        self._qs_configs['optimize_pipeline_config'])
    self._quantize_pipeline = Vitis8BitQuantizeTransformsPipeline(
        self._qs_configs['quantize_pipeline_config'])
    self._quantize_registry = VitisQuantizeRegistry(
        self._qs_configs['quantize_registry_config'])
