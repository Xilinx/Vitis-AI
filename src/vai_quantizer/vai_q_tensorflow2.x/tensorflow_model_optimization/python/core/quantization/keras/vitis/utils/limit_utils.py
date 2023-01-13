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
"""Utility function for layer limit."""

from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_layer_limits
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

logger = common_utils.VAILogger
Limit = vitis_layer_limits.Limit
LimitType = vitis_layer_limits.LimitType


def _is_leaky_relu_quantizable(alpha, alpha_target=0.1, threshold=1e-7):
  """Due to DPU constraints, only leaky_relu with alpha=0.1 is quantizable."""
  if abs(alpha - alpha_target) < threshold:
    return True
  else:
    return False


def str_to_pair_limit(in_str):
  """Convert string to pair limits. e.g. '1-6' to int range limit."""
  if '-' in in_str and ',' in in_str:
    tmp = in_str.split(',')
    in_str = [s for s in tmp if '-' not in s]
    for s in tmp:
      if '-' in s:
        start, end = s.split('-')
        li = range(int(start), int(end) + 1)
        for i in li:
          in_str.append(str(i))
    in_str = ','.join(in_str)

  h_str, w_str = in_str, in_str
  limit_str = '{};{}'.format(h_str, w_str)

  limit_type = None
  if '-' in in_str:
    limit_type = LimitType.INT_RANGE_PAIR
  else:
    limit_type = LimitType.INT_CHOICE_PAIR
  return Limit(limit_str, limit_type=limit_type)
