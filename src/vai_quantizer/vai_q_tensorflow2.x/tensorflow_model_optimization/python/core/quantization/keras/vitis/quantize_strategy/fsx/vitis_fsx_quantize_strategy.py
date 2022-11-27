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
"""Vitis float scale (Xilinx Version) quantization strategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.fs import vitis_fs_transforms_pipeline
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_strategy
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_registry
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

VitisQuantizeStrategy = vitis_quantize_strategy.VitisQuantizeStrategy
VitisQuantizeRegistry = vitis_quantize_registry.VitisQuantizeRegistry
VitisFSOptimizeTransformsPipeline = vitis_fs_transforms_pipeline.VitisFSOptimizeTransformsPipeline
VitisFSQuantizeTransformsPipeline = vitis_fs_transforms_pipeline.VitisFSQuantizeTransformsPipeline
VitisFSRefineTransformsPipeline = vitis_fs_transforms_pipeline.VitisFSRefineTransformsPipeline
VitisFSFinalizeTransformsPipeline = vitis_fs_transforms_pipeline.VitisFSFinalizeTransformsPipeline
logger = common_utils.VAILogger

# vitis float_scale quantize strategy config
dirname = os.path.dirname(__file__)
vitis_fsx_qs_config = os.path.join(dirname, "vitis_fsx_quantize_strategy.json")


class VitisFSXQuantizeStrategy(VitisQuantizeStrategy):
  """Vitis 8-Bit Float Scale (Xilinx Version) Quantize Strategy."""

  def __init__(self, qs_configs=vitis_fsx_qs_config):
    """Init

    Args:
      qs_config: A json file contains the quantize strategy configurations.
    """
    self._qs_configs = common_utils.load_json(qs_configs)

    self._optimize_pipeline = VitisFSOptimizeTransformsPipeline(
        self._qs_configs['optimize_pipeline_config'])
    self._quantize_pipeline = VitisFSQuantizeTransformsPipeline(
        self._qs_configs['quantize_pipeline_config'])
    self._refine_pipeline = VitisFSRefineTransformsPipeline(
        self._qs_configs['refine_pipeline_config'])
    self._finalize_pipeline = VitisFSFinalizeTransformsPipeline(
        self._qs_configs['finalize_pipeline_config'])
    self._quantize_registry = VitisQuantizeRegistry(
        self._qs_configs['quantize_registry_config'])
