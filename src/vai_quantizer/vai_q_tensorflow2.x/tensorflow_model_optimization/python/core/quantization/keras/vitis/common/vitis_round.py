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
"""Round implementation of different rounding modes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

logger = common_utils.VAILogger


class RoundMode(enum.Enum):
  """Enum class for round modes."""
  # ROUND_HALF_TO_EVEN, used in py3 round, tf.round or numpy.round.
  HALF_TO_EVEN = 0

  # ROUND_HALF_UP, used in dpu round and tf.fake_quant.
  HALF_UP = 1

  # ROUND_HALF_AWAY_FROM_ZERO, used in std round/py2 round.
  HALF_AWAY_FROM_ZERO = 2


def round_half_to_even(x):
  """ROUND_HALF_TO_EVEN, used in py3 round, tf.round or numpy.round.
      f(x) = round(x)
      eg: f(2.3) = 2, f(1.5) = 2, f(-1.5) = -2, f(2.5) = 2, f(-2.5) = -2, f(-2.6) = -3
  """
  rounded = tf.math.round(x)
  return rounded


def round_half_up(x):
  """ROUND_HALF_UP, used in dpu round and tf.fake_quant
      f(x) = (x - floor(x) == 0.5) ? ceil(x) : round(x)
           = floor(x + 0.5)
      eg: f(2.3) = 2, f(1.5) = 2, f(-1.5) = -1, f(2.5) = 3, f(-2.5) = -2, f(-2.6) = -3
  """
  rounded = tf.math.floor(x + 0.5)
  return rounded


def round_half_away_from_zero(x):
  """ROUND_HALF_AWAY_FROM_ZERO, used in std round/py2 round.
      f(x) = std::round(x)
             ceil(x),   x - floor(x) == 0.5 && x > 0
           = round(x),  x - floor(x) != 0.5
             floor(x),  x - floor(x) == 0.5 && x < 0
      eg: f(2.3) = 2, f(1.5) = 2, f(-1.5) = -2, f(2.5) = 3, f(-2.5) = -3, f(-2.6) = -3
  """
  floored = tf.math.floor(x)
  ceiled = tf.math.ceil(x)
  rounded = tf.math.round(x)
  rounded_half = tf.where(x > 0, ceiled, floored)
  rounded = tf.where(tf.math.equal(x - floored, 0.5), rounded_half, rounded)
  return rounded


def round(x, round_mode):
  """Round with different modes."""
  if round_mode == RoundMode.HALF_TO_EVEN:
    return round_half_to_even(x)
  elif round_mode == RoundMode.HALF_UP:
    return round_half_up(x)
  elif round_mode == RoundMode.HALF_AWAY_FROM_ZERO:
    return round_half_away_from_zero(x)
  else:
    logger.error('Invalid round_mode: {}'.format(round_mode))
