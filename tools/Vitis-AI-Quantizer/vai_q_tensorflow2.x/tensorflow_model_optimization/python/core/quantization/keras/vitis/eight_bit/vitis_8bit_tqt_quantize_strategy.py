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
"""Vitis 8-Bit Trained Quantization Threshold Quantization Strategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras.vitis.eight_bit import vitis_8bit_quantize_strategy
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

Vitis8BitQuantizeStrategy = vitis_8bit_quantize_strategy.Vitis8BitQuantizeStrategy
logger = common_utils.VAILogger

# vitis 8bit tqt quantize strategy config
dirname = os.path.dirname(__file__)
vitis_8bit_tqt_qs_config = os.path.join(
    dirname, "vitis_8bit_tqt_quantize_strategy.json")


class Vitis8BitTQTQuantizeStrategy(Vitis8BitQuantizeStrategy):
  """Vitis 8-Bit Trained Quantization Threshold Quantize Strategy."""

  def __init__(self, qs_configs=vitis_8bit_tqt_qs_config):
    """Init

    Args:
      qs_config: A json file contains the quantize strategy configurations.
    """
    super(Vitis8BitTQTQuantizeStrategy, self).__init__(qs_configs)
