# Copyright 2019, The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Misc."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_model_optimization.python.core.internal.tensor_encoding.core import encoding_stage


@encoding_stage.tf_style_encoding_stage
class SplitBySmallValueEncodingStage(encoding_stage.EncodingStageInterface):
  """Encoding stage splitting the input by small values.

  This encoding stage will split the input into two outputs: the value and the
  indices of the elements whose absolute value is larger than a certain
  threshold. The elements smaller than the threshold is then decoded to zero.
  """

  ENCODED_INDICES_KEY = 'indices'
  ENCODED_VALUES_KEY = 'non_zero_floats'
  THRESHOLD_PARAMS_KEY = 'threshold'

  def __init__(self, threshold=1e-8):
    """Initializer for the SplitBySmallValueEncodingStage.

    Args:
      threshold: The threshold of the small weights to be set to zero.
    """
    self._threshold = threshold

  @property
  def name(self):
    """See base class."""
    return 'split_by_small_value'

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return [
        self.ENCODED_VALUES_KEY,
        self.ENCODED_INDICES_KEY,
    ]

  @property
  def commutes_with_sum(self):
    """See base class."""
    return False

  @property
  def decode_needs_input_shape(self):
    """See base class."""
    return True

  def get_params(self):
    """See base class."""
    return {self.THRESHOLD_PARAMS_KEY: self._threshold}, {}

  def encode(self, x, encode_params):
    """See base class."""

    threshold = tf.cast(encode_params[self.THRESHOLD_PARAMS_KEY], x.dtype)
    indices = tf.cast(tf.compat.v2.where(tf.abs(x) > threshold), tf.int32)
    non_zero_x = tf.gather_nd(x, indices)
    indices = tf.squeeze(indices, axis=1)
    return {
        self.ENCODED_INDICES_KEY: indices,
        self.ENCODED_VALUES_KEY: non_zero_x,
    }

  def decode(self,
             encoded_tensors,
             decode_params,
             num_summands=None,
             shape=None):
    """See base class."""
    del decode_params, num_summands  # Unused.

    indices = encoded_tensors[self.ENCODED_INDICES_KEY]
    non_zero_x = encoded_tensors[self.ENCODED_VALUES_KEY]

    indices = tf.expand_dims(indices, 1)
    shape = tf.cast(shape, indices.dtype)
    decoded_x = tf.scatter_nd(indices=indices, updates=non_zero_x, shape=shape)

    return decoded_x


@encoding_stage.tf_style_encoding_stage
class DifferenceBetweenIntegersEncodingStage(
    encoding_stage.EncodingStageInterface):
  """Encoding stage taking the difference between a sequence of integers.

  This encoding stage can be useful when the original integers can be large, but
  the difference of the integers are much smaller values and have a more compact
  representation. For example, it can be combined with the
  `SplitBySmallValueEncodingStage` to further compress the increasing sequence
  of indices.

  The encode method expects a tensor with 1 dimension and with integer dtype.
  """

  ENCODED_VALUES_KEY = 'difference_between_integers'

  @property
  def name(self):
    """See base class."""
    return 'difference_between_integers'

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return [
        self.ENCODED_VALUES_KEY,
    ]

  @property
  def commutes_with_sum(self):
    """See base class."""
    return False

  @property
  def decode_needs_input_shape(self):
    """See base class."""
    return False

  def get_params(self):
    """See base class."""
    return {}, {}

  def encode(self, x, encode_params):
    """See base class."""
    del encode_params  # Unused.
    if x.shape.ndims != 1:
      raise ValueError('Number of dimensions must be 1. Shape of x: %s' %
                       x.shape)
    if not x.dtype.is_integer:
      raise TypeError(
          'Unsupported input type: %s. Support only integer types.' % x.dtype)

    diff_x = x - tf.concat([[0], x[:-1]], 0)
    return {
        self.ENCODED_VALUES_KEY: diff_x,
    }

  def decode(self,
             encoded_tensors,
             decode_params,
             num_summands=None,
             shape=None):
    """See base class."""
    del decode_params, num_summands, shape  # Unused
    return tf.cumsum(encoded_tensors[self.ENCODED_VALUES_KEY])
