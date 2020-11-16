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
"""Implementations of ideas related to quantization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_model_optimization.python.core.internal.tensor_encoding.core import encoding_stage
from tensorflow_model_optimization.python.core.internal.tensor_encoding.utils import tf_utils


@encoding_stage.tf_style_encoding_stage
class PRNGUniformQuantizationEncodingStage(encoding_stage.EncodingStageInterface
                                          ):
  """Encoding stage performing uniform quantization using PRNG.

  Different from `UniformQuantizationEncodingStage`, which uses deterministic
  decoding, this stage uses a single integer seed as the same source of
  randomness in encode and decode methods, in order to, in the decode method,
  differently estimate the original values before encoding was applied.

  In particular, given a floating point input `x` to the `encode` method, the
  output will have the same `dtype` as `x`, with values being "floating point
  integers" in the range `[0, 2**bits-1]`.

  The shape of the input `x` to the `encode` method must be statically known.

  The extreme points of the quantized interval will correspond to the min and
  max values of the input `x`.

  The output of the `decode` method is an unbiased estimation of the original
  values before quantization, but note that it is possible for some of the
  decoded values to be outside of the original range.
  """

  ENCODED_VALUES_KEY = 'quantized_values'
  MIN_MAX_VALUES_KEY = 'min_max'
  MAX_INT_VALUE_PARAMS_KEY = 'max_value'
  SEED_PARAMS_KEY = 'seed'
  # The allowed values for `bits` argument to initializer.
  # We cap the allowed quantization bits at 16, as the randomized rounding could
  # otherwise be numerically unstable for float32 values.
  _ALLOWED_BITS_ARG = list(range(1, 17))

  def __init__(self, bits=8):
    """Initializer for the PRNGUniformQuantizationEncodingStage.

    Args:
      bits: The number of bits to quantize to. Must be an integer between 1 and
        16. Can be either a TensorFlow or a Python value.

    Raises:
      ValueError: The inputs do not satisfy the above constraints.
    """
    if (not tf.is_tensor(bits) and bits not in self._ALLOWED_BITS_ARG):
      raise ValueError('The bits argument must be an integer between 1 and 16.')
    self._bits = bits

  @property
  def name(self):
    """See base class."""
    return 'PRNG_uniform_quantization'

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return [self.ENCODED_VALUES_KEY]

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
    params = {self.MAX_INT_VALUE_PARAMS_KEY: 2**self._bits - 1}
    return params, params

  def encode(self, x, encode_params):
    """See base class."""
    min_x = tf.reduce_min(x)
    max_x = tf.reduce_max(x)

    max_value = tf.cast(encode_params[self.MAX_INT_VALUE_PARAMS_KEY], x.dtype)
    # Shift the values to range [0, max_value].
    # In the case of min_x == max_x, this will return all zeros.
    x = tf.compat.v1.div_no_nan(x - min_x, max_x - min_x) * max_value

    # Randomized rounding.
    floored_x = tf.floor(x)
    random_seed = tf.random.uniform((2,), maxval=tf.int64.max, dtype=tf.int64)
    num_elements = tf.reduce_prod(tf.shape(x))
    rounding_floats = tf.reshape(
        self._random_floats(num_elements, random_seed, x.dtype), tf.shape(x))

    bernoulli = rounding_floats < (x - floored_x)
    quantized_x = floored_x + tf.cast(bernoulli, x.dtype)

    # Include the random seed in the encoded tensors so that it can be used to
    # generate the same random sequence in the decode method.
    encoded_tensors = {
        self.ENCODED_VALUES_KEY: quantized_x,
        self.SEED_PARAMS_KEY: random_seed,
        self.MIN_MAX_VALUES_KEY: tf.stack([min_x, max_x])
    }

    return encoded_tensors

  def decode(self,
             encoded_tensors,
             decode_params,
             num_summands=None,
             shape=None):
    """See base class."""
    del num_summands, shape  # Unused.
    quantized_x = encoded_tensors[self.ENCODED_VALUES_KEY]
    random_seed = encoded_tensors[self.SEED_PARAMS_KEY]
    min_max = encoded_tensors[self.MIN_MAX_VALUES_KEY]
    min_x, max_x = min_max[0], min_max[1]
    max_value = tf.cast(decode_params[self.MAX_INT_VALUE_PARAMS_KEY],
                        quantized_x.dtype)

    num_elements = tf.reduce_prod(tf.shape(quantized_x))
    # The rounding_floats are identical to those used in the encode method.
    rounding_floats = tf.reshape(
        self._random_floats(num_elements, random_seed, min_x.dtype),
        tf.shape(quantized_x))

    # Regenerating the random values used in encode, enables us to determine a
    # narrower range of possible original values, before quantization was
    # applied. We shift the quantized values into the middle of this range,
    # corresponding to the intersection of
    # [quantized_x - 1 + rounding_floats, quantized_x + rounding_floats]
    # in the quantized range. This shifted value can be out of the range
    # [0, max_value] and therefore the decoded value can be out of the range
    # [min_x, max_x], which is impossible, but it ensures that the decoded x
    # is an unbiased estimator of the original values before quantization.
    q_shifted = quantized_x + rounding_floats - 0.5

    x = q_shifted / max_value * (max_x - min_x) + min_x
    return x

  def _random_floats(self, num_elements, seed, dtype):
    return tf_utils.random_floats(num_elements, seed, dtype)


@encoding_stage.tf_style_encoding_stage
class PerChannelUniformQuantizationEncodingStage(
    encoding_stage.EncodingStageInterface):
  """Encoding stage performing uniform quantization per channel in conv layers.

  This encoding stage will first reshape the input `x` to `(-1, dim)`, where
  `dim` is the size of the last dimension of `x`. Then each of the `dim`
  vectors will be quantized independently.

  This encoding stage does not require the shape of the input `x` to the
  `encode` method to be statically known.

  In particular, given a floating point input `x` to the `encode` method, the
  output will have the same `dtype` as `x`, with values being "floating point
  integers" in the range `[0, 2**bits-1]`.

  The extreme points of the quantized interval will correspond to the min and
  max values of the input `x`.
  """

  ENCODED_VALUES_KEY = 'quantized_values'
  MIN_MAX_VALUES_KEY = 'min_max'
  MAX_INT_VALUE_PARAMS_KEY = 'max_value'
  SEED_PARAMS_KEY = 'seed'
  # The allowed values for `bits` argument to initializer.
  # We cap the allowed quantization bits at 16, as the randomized rounding could
  # otherwise be numerically unstable for float32 values.
  _ALLOWED_BITS_ARG = list(range(1, 17))

  def __init__(self, bits=8, stochastic=True):
    """Initializer for the PerChannelUniformQuantizationEncodingStage.

    Args:
      bits: The number of bits to quantize to. Must be an integer between 1 and
        16. Can be either a TensorFlow or a Python value.
      stochastic: A Python bool, whether to use stochastic or deterministic
        rounding. If `True`, the encoding is randomized and on expectation
        unbiased. If `False`, the encoding is deterministic.

    Raises:
      ValueError: The inputs do not satisfy the above constraints.
    """
    if (not tf.is_tensor(bits) and bits not in self._ALLOWED_BITS_ARG):
      raise ValueError('The bits argument must be an integer between 1 and 16.')
    self._bits = bits

    if not isinstance(stochastic, bool):
      raise TypeError('The stochastic argument must be a bool.')
    self._stochastic = stochastic

  @property
  def name(self):
    """See base class."""
    return 'per_channel_uniform_quantization'

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return [self.ENCODED_VALUES_KEY]

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
    params = {self.MAX_INT_VALUE_PARAMS_KEY: 2**self._bits - 1}
    return params, params

  def encode(self, x, encode_params):
    """See base class."""
    dim = tf.shape(x)[-1]
    x = tf.reshape(x, [-1, dim])

    # Per-channel min and max.
    min_x = tf.reduce_min(x, axis=0)
    max_x = tf.reduce_max(x, axis=0)

    max_value = tf.cast(encode_params[self.MAX_INT_VALUE_PARAMS_KEY], x.dtype)
    # Shift the values to range [0, max_value].
    # In the case of min_x == max_x, this will return all zeros.
    x = tf.compat.v1.div_no_nan(x - min_x, max_x - min_x) * max_value
    if self._stochastic:  # Randomized rounding.
      floored_x = tf.floor(x)
      bernoulli = tf.random.uniform(tf.shape(x), dtype=x.dtype)
      bernoulli = bernoulli < (x - floored_x)
      quantized_x = floored_x + tf.cast(bernoulli, x.dtype)
    else:  # Deterministic rounding.
      quantized_x = tf.round(x)

    encoded_tensors = {
        self.ENCODED_VALUES_KEY: quantized_x,
        self.MIN_MAX_VALUES_KEY: tf.stack([min_x, max_x])
    }

    return encoded_tensors

  def decode(self,
             encoded_tensors,
             decode_params,
             num_summands=None,
             shape=None):
    """See base class."""
    del num_summands  # Unused.
    quantized_x = encoded_tensors[self.ENCODED_VALUES_KEY]
    min_max = encoded_tensors[self.MIN_MAX_VALUES_KEY]
    min_x, max_x = min_max[0], min_max[1]
    max_value = tf.cast(decode_params[self.MAX_INT_VALUE_PARAMS_KEY],
                        quantized_x.dtype)

    x = quantized_x / max_value * (max_x - min_x) + min_x

    x = tf.reshape(x, shape)

    return x


@encoding_stage.tf_style_encoding_stage
class PerChannelPRNGUniformQuantizationEncodingStage(
    encoding_stage.EncodingStageInterface):
  """Encoding stage performing uniform quantization per channel in conv layers.

  This encoding stage will first reshape the input `x` to `(-1, dim)`, where
  `dim` is the size of the last dimension of `x`. Then each of the `dim`
  vectors will be quantized independently, using PRNG uniform quantization.

  PRNG uniform quantization uses a single integer seed as the same source of
  randomness in encode and decode methods, in order to, in the decode method,
  differently estimate the original values before encoding was applied.

  This encoding stage does not require the shape of the input `x` to the
  `encode` method to be statically known.

  In particular, given a floating point input `x` to the `encode` method, the
  output will have the same `dtype` as `x`, with values being "floating point
  integers" in the range `[0, 2**bits-1]`.

  The extreme points of the quantized interval will correspond to the min and
  max values of the input `x`.

  The output of the `decode` method is an unbiased estimation of the original
  values before quantization, but note that it is possible for some of the
  decoded values to be outside of the original range.
  """

  ENCODED_VALUES_KEY = 'quantized_values'
  MIN_MAX_VALUES_KEY = 'min_max'
  MAX_INT_VALUE_PARAMS_KEY = 'max_value'
  SEED_PARAMS_KEY = 'seed'
  # The allowed values for `bits` argument to initializer.
  # We cap the allowed quantization bits at 16, as the randomized rounding could
  # otherwise be numerically unstable for float32 values.
  _ALLOWED_BITS_ARG = list(range(1, 17))

  def __init__(self, bits=8):
    """Initializer for the `PerChannelPRNGUniformQuantizationEncodingStage`.

    Args:
      bits: The number of bits to quantize to. Must be an integer between 1 and
        16. Can be either a TensorFlow or a Python value.

    Raises:
      ValueError: The inputs do not satisfy the above constraints.
    """
    if (not tf.is_tensor(bits) and bits not in self._ALLOWED_BITS_ARG):
      raise ValueError('The bits argument must be an integer between 1 and 16.')
    self._bits = bits

  @property
  def name(self):
    """See base class."""
    return 'per_channel_prng_uniform_quantization'

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return [self.ENCODED_VALUES_KEY]

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
    params = {self.MAX_INT_VALUE_PARAMS_KEY: 2**self._bits - 1}
    return params, params

  def encode(self, x, encode_params):
    """See base class."""
    dim = tf.shape(x)[-1]
    x = tf.reshape(x, [-1, dim])

    # Per-channel min and max.
    min_x = tf.reduce_min(x, axis=0)
    max_x = tf.reduce_max(x, axis=0)

    max_value = tf.cast(encode_params[self.MAX_INT_VALUE_PARAMS_KEY], x.dtype)
    # Shift the values to range [0, max_value].
    # In the case of min_x == max_x, this will return all zeros.
    x = tf.compat.v1.div_no_nan(x - min_x, max_x - min_x) * max_value

    # Randomized rounding.
    floored_x = tf.floor(x)
    random_seed = tf.random.uniform((2,), maxval=tf.int64.max, dtype=tf.int64)
    num_elements = tf.reduce_prod(tf.shape(x))
    rounding_floats = tf.reshape(
        self._random_floats(num_elements, random_seed, x.dtype), tf.shape(x))

    bernoulli = rounding_floats < (x - floored_x)
    quantized_x = floored_x + tf.cast(bernoulli, x.dtype)

    # Include the random seed in the encoded tensors so that it can be used to
    # generate the same random sequence in the decode method.
    encoded_tensors = {
        self.ENCODED_VALUES_KEY: quantized_x,
        self.SEED_PARAMS_KEY: random_seed,
        self.MIN_MAX_VALUES_KEY: tf.stack([min_x, max_x])
    }

    return encoded_tensors

  def decode(self,
             encoded_tensors,
             decode_params,
             num_summands=None,
             shape=None):
    """See base class."""
    del num_summands  # Unused.
    quantized_x = encoded_tensors[self.ENCODED_VALUES_KEY]
    random_seed = encoded_tensors[self.SEED_PARAMS_KEY]
    min_max = encoded_tensors[self.MIN_MAX_VALUES_KEY]
    min_x, max_x = min_max[0], min_max[1]
    max_value = tf.cast(decode_params[self.MAX_INT_VALUE_PARAMS_KEY],
                        quantized_x.dtype)

    num_elements = tf.reduce_prod(tf.shape(quantized_x))
    # The rounding_floats are identical to those used in the encode method.
    rounding_floats = tf.reshape(
        self._random_floats(num_elements, random_seed, min_x.dtype),
        tf.shape(quantized_x))

    # Regenerating the random values used in encode, enables us to determine a
    # narrower range of possible original values, before quantization was
    # applied. We shift the quantized values into the middle of this range,
    # corresponding to the intersection of
    # [quantized_x - 1 + rounding_floats, quantized_x + rounding_floats]
    # in the quantized range. This shifted value can be out of the range
    # [0, max_value] and therefore the decoded value can be out of the range
    # [min_x, max_x], which is impossible, but it ensures that the decoded x
    # is an unbiased estimator of the original values before quantization.
    q_shifted = quantized_x + rounding_floats - 0.5

    x = q_shifted / max_value * (max_x - min_x) + min_x

    x = tf.reshape(x, shape)

    return x

  def _random_floats(self, num_elements, seed, dtype):
    return tf_utils.random_floats(num_elements, seed, dtype)
