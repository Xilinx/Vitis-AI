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
"""Vitis trained quantization threshold quantization strategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.pof2s import vitis_pof2s_quantize_strategy
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.tqt import vitis_tqt_transforms_pipeline
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

VitisPof2SQuantizeStrategy = vitis_pof2s_quantize_strategy.VitisPof2SQuantizeStrategy
VitisTQTRefineTransformsPipeline = vitis_tqt_transforms_pipeline.VitisTQTRefineTransformsPipeline
VitisTQTFinalizeTransformsPipeline = vitis_tqt_transforms_pipeline.VitisTQTFinalizeTransformsPipeline
logger = common_utils.VAILogger

# vitis tqt quantize strategy config
dirname = os.path.dirname(__file__)
vitis_tqt_qs_config = os.path.join(dirname, "vitis_tqt_quantize_strategy.json")


class VitisTQTQuantizeStrategy(VitisPof2SQuantizeStrategy):
  """Vitis trained quantization threshold quantize strategy."""

  def __init__(self, qs_configs=vitis_tqt_qs_config):
    """Init

    Args:
      qs_config: A json file contains the quantize strategy configurations.
    """
    super(VitisTQTQuantizeStrategy, self).__init__(qs_configs)
    # Now the difference of tqt and pof2s is only in refine and finalize pipeline
    self._refine_pipeline = VitisTQTRefineTransformsPipeline(
        self._qs_configs['refine_pipeline_config'])
    self._finalize_pipeline = VitisTQTFinalizeTransformsPipeline(
        self._qs_configs['finalize_pipeline_config'])
