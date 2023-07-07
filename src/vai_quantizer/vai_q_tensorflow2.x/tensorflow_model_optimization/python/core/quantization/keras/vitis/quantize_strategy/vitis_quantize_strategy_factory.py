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
"""Vitis quantize strategy factory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.pof2s import vitis_pof2s_quantize_strategy
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.tqt import vitis_tqt_quantize_strategy
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.fs import vitis_fs_quantize_strategy
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.fsx import vitis_fsx_quantize_strategy
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

logger = common_utils.VAILogger


def get_quantize_strategy(identifier):
  """Returns quantize strategy."""

  _quantize_strategy_dict = {
      '8bit': vitis_pof2s_quantize_strategy.VitisPof2SQuantizeStrategy(),
      '8bit_tqt': vitis_tqt_quantize_strategy.VitisTQTQuantizeStrategy(),
      'pof2s': vitis_pof2s_quantize_strategy.VitisPof2SQuantizeStrategy(),
      'pof2s_tqt': vitis_tqt_quantize_strategy.VitisTQTQuantizeStrategy(),
      'fs': vitis_fs_quantize_strategy.VitisFSQuantizeStrategy(),
      'fsx': vitis_fsx_quantize_strategy.VitisFSXQuantizeStrategy()
  }

  if isinstance(identifier, six.string_types):
    identifier = str(identifier)
    if identifier in _quantize_strategy_dict:
      return _quantize_strategy_dict.get(identifier)
    else:
      logger.error(
          'Unidentified quantize strategy: {}. Supported built-in quantize strategies are {}'
          .format(identifier, _quantize_strategy_dict.keys()), KeyError)
  else:
    logger.error('Invalid identifier type: {}'.format(type(identifier)))
