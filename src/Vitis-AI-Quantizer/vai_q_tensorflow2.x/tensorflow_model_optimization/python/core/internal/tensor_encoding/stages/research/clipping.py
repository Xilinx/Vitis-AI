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
"""Encoding stages implementing various clipping strategies.

The base classes, `ClipByNormEncodingStage` and `ClipByValueEncodingStage`, are
expected to be subclassed as implementations of
`AdaptiveEncodingStageInterface`, to realize a variety of clipping strategies
that are adaptive to the data being processed in an iterative execution.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_model_optimization.python.core.internal.tensor_encoding.core import encoding_stage


@encoding_stage.tf_style_encoding_stage
class ClipByNormEncodingStage(encoding_stage.EncodingStageInterface):
  """Encoding stage applying clipping by norm (L-2 ball projection).

  See `tf.clip_by_norm` for more information.
  """

  ENCODED_VALUES_KEY = 'clipped_values'
  NORM_PARAMS_KEY = 'norm_param'

  def __init__(self, clip_norm):
    """Initializer for the `ClipByNormEncodingStage`.

    Args:
      clip_norm: A scalar, norm of the ball onto which to project.
    """
    self._clip_norm = clip_norm

  @property
  def name(self):
    """See base class."""
    return 'clip_by_norm'

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return [self.ENCODED_VALUES_KEY]

  @property
  def commutes_with_sum(self):
    """See base class."""
    return True

  @property
  def decode_needs_input_shape(self):
    """See base class."""
    return False

  def get_params(self):
    """See base class."""
    return {self.NORM_PARAMS_KEY: self._clip_norm}, {}

  def encode(self, x, encode_params):
    """See base class."""
    clipped_x = tf.clip_by_norm(
        x, tf.cast(encode_params[self.NORM_PARAMS_KEY], x.dtype))
    return {self.ENCODED_VALUES_KEY: clipped_x}

  def decode(self,
             encoded_tensors,
             decode_params,
             num_summands=None,
             shape=None):
    """See base class."""
    del decode_params, num_summands, shape  # Unused.
    return tf.identity(encoded_tensors[self.ENCODED_VALUES_KEY])


@encoding_stage.tf_style_encoding_stage
class ClipByValueEncodingStage(encoding_stage.EncodingStageInterface):
  """Encoding stage applying clipping by value (L-infinity ball projection).

  See `tf.clip_by_value` for more information.
  """

  ENCODED_VALUES_KEY = 'clipped_values'
  MIN_PARAMS_KEY = 'min_param'
  MAX_PARAMS_KEY = 'max_param'

  def __init__(self, clip_value_min, clip_value_max):
    """Initializer for the `ClipByValueEncodingStage`.

    Args:
      clip_value_min: A scalar, the minimum value to which to clip.
      clip_value_max: A scalar, the maximum value to which to clip.
    """
    self._clip_value_min = clip_value_min
    self._clip_value_max = clip_value_max

  @property
  def name(self):
    """See base class."""
    return 'clip_by_value'

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return [self.ENCODED_VALUES_KEY]

  @property
  def commutes_with_sum(self):
    """See base class."""
    return True

  @property
  def decode_needs_input_shape(self):
    """See base class."""
    return False

  def get_params(self):
    """See base class."""
    params = {
        self.MIN_PARAMS_KEY: self._clip_value_min,
        self.MAX_PARAMS_KEY: self._clip_value_max
    }
    return params, {}

  def encode(self, x, encode_params):
    """See base class."""
    clipped_x = tf.clip_by_value(
        x,
        tf.cast(encode_params[self.MIN_PARAMS_KEY], x.dtype),
        tf.cast(encode_params[self.MAX_PARAMS_KEY], x.dtype))
    return {self.ENCODED_VALUES_KEY: clipped_x}

  def decode(self,
             encoded_tensors,
             decode_params,
             num_summands=None,
             shape=None):
    """See base class."""
    del decode_params, num_summands, shape  # Unused.
    return tf.identity(encoded_tensors[self.ENCODED_VALUES_KEY])
