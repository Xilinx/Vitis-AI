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
"""Implementations of the encoding stage interfaces."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_model_optimization.python.core.internal.tensor_encoding.core import encoding_stage
from tensorflow_model_optimization.python.core.internal.tensor_encoding.utils import tf_utils


@encoding_stage.tf_style_encoding_stage
class IdentityEncodingStage(encoding_stage.EncodingStageInterface):
  """Encoding stage acting as the identity."""

  ENCODED_VALUES_KEY = 'identity_values'

  @property
  def name(self):
    """See base class."""
    return 'identity'

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
    return {}, {}

  def encode(self, x, encode_params):
    """See base class."""
    del encode_params  # Unused.
    return {self.ENCODED_VALUES_KEY: tf.identity(x)}

  def decode(self,
             encoded_tensors,
             decode_params,
             num_summands=None,
             shape=None):
    """See base class."""
    del decode_params, num_summands, shape  # Unused.
    return tf.identity(encoded_tensors[self.ENCODED_VALUES_KEY])


@encoding_stage.tf_style_encoding_stage
class FlattenEncodingStage(encoding_stage.EncodingStageInterface):
  """Encoding stage reshaping the input to be a rank 1 `Tensor`."""

  ENCODED_VALUES_KEY = 'flattened_values'

  @property
  def name(self):
    """See base class."""
    return 'flatten'

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
    return True

  def get_params(self):
    """See base class."""
    return {}, {}

  def encode(self, x, encode_params):
    """See base class."""
    del encode_params  # Unused.
    return {self.ENCODED_VALUES_KEY: tf.reshape(x, [-1])}

  def decode(self,
             encoded_tensors,
             decode_params,
             num_summands=None,
             shape=None):
    """See base class."""
    del decode_params, num_summands  # Unused.
    return tf.reshape(encoded_tensors[self.ENCODED_VALUES_KEY], shape)


@encoding_stage.tf_style_encoding_stage
class HadamardEncodingStage(encoding_stage.EncodingStageInterface):
  """Encoding stage multiplying input by appropriate randomized Hadamard matrix.

  This encoding stage implements the fast Walsh-Hadamard transform to
  efficiently compute matrix-vector product in O(n log n) time.
  https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform

  The useful property of randomized Hadamard transform is that it spreads the
  information in a vector more uniformly across its coefficients. This work well
  together with uniform quantization, as it reduces the dynamic range of the
  coefficients to be quantized, decreasing the error incurred by quantization.

  The encoding works as follows:
  The shape of the input `x` to the `encode` method must be either `(dim)` or
  `(b, dim)`, where `dim` is the dimenion of the vector to which the transform
  is to be applied, and must be statically known. `b` represents an optional
  batch dimension, and does not need to be statically known.

  If the shape of the input is `(dim)`, it is first expanded to `(1, dim)`. The
  input of shape `(b, dim)` has signs randomly flipped (as determined by random
  seed to be reused in decoding) and is padded with zeros to dimension
  `(b, dim_2)`, where `dim_2` is the smallest power of 2 larger than or equal
  to `dim`.

  The same transform is then applied to each of the `b` vectors of shape
  `dim_2`. If `H` represents the `(dim_2, dim_2)` Hadamard matrix and `D`
  represents the `(dim, dim_2)` diagonal matrix with randomly sampled `+1/-1`
  elements, the `encode` method computes `x[i, :]*D*H` for every `i`. In other
  words, if the leading dimension represents a batch, the transform is applied
  to every element in the batch. The encoded value then has shape `(b, dim_2)`,
  where `dim_2` is the smallest power of 2 larger than or equal to `dim`.
  """

  ENCODED_VALUES_KEY = 'hadamard_values'
  SEED_PARAMS_KEY = 'seed'

  @property
  def name(self):
    """See base class."""
    return 'hadamard'

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
    return True

  def get_params(self):
    """See base class."""
    params = {
        self.SEED_PARAMS_KEY:
            tf.random.uniform((2,), maxval=tf.int64.max, dtype=tf.int64),
    }
    return params, params

  def encode(self, x, encode_params):
    """See base class."""
    x = self._validate_and_expand_encode_input(x)
    signs = self._random_signs(x.shape.as_list()[1],
                               encode_params[self.SEED_PARAMS_KEY], x.dtype)
    x = x * signs
    x = self._pad(x)
    rotated_x = tf_utils.fast_walsh_hadamard_transform(x)
    return {self.ENCODED_VALUES_KEY: rotated_x}

  def decode(self,
             encoded_tensors,
             decode_params,
             num_summands=None,
             shape=None):
    """See base class."""
    del num_summands  # Unused.
    rotated_x = encoded_tensors[self.ENCODED_VALUES_KEY]
    unrotated_x = tf_utils.fast_walsh_hadamard_transform(rotated_x)

    # Take slice corresponding to the input shape.
    decoded_x = tf.slice(unrotated_x, [0, 0],
                         [tf.shape(unrotated_x)[0], shape[-1]])
    signs = self._random_signs(decoded_x.shape.as_list()[-1],
                               decode_params[self.SEED_PARAMS_KEY],
                               decoded_x.dtype)
    decoded_x = decoded_x * signs
    if shape.shape.num_elements() == 1:
      decoded_x = tf.squeeze(decoded_x, [0])
    return decoded_x

  def _validate_and_expand_encode_input(self, x):
    """Validates the input to encode and modifies it if necessary."""
    if x.shape.ndims not in [1, 2]:
      raise ValueError(
          'Number of dimensions must be 1 or 2. Shape of x: %s' % x.shape)
    if x.shape.ndims == 1:
      # The input to the fast_walsh_hadamard_transform must have 2 dimensions.
      x = tf.expand_dims(x, 0)
    if x.shape.as_list()[1] is None:
      raise ValueError(
          'The dimension of the object to be rotated must be fully known.')
    return x

  def _pad(self, x):
    """Pads with zeros to the next power of two."""
    dim = x.shape.as_list()[1]
    pad_dim = 2**int(np.ceil(np.log2(dim)))
    if pad_dim != dim:
      x = tf.pad(x, [[0, 0], [0, pad_dim - dim]])
    return x

  def _random_signs(self, num_elements, seed, dtype):
    return tf_utils.random_signs(num_elements, seed, dtype)


@encoding_stage.tf_style_encoding_stage
class UniformQuantizationEncodingStage(encoding_stage.EncodingStageInterface):
  """Encoding stage performing uniform quantization.

  This class performs quantization to uniformly spaced values, without realizing
  any savings by itself.

  In particular, given a floating point input `x` to the `encode` method, the
  output will have the same `dtype` as `x`, with values being "floating point
  integers" in the range `[0, 2**bits-1]`.

  If `min_max` is not provided, the extreme points of the quantized interval
  will correspond to the min and max values of the input `x`. If `min_max` is
  provided, the input `x` is first clipped to this range, and the extreme points
  of the quantized interval correspond to the provided `min_max` values.
  """

  ENCODED_VALUES_KEY = 'quantized_values'
  MIN_MAX_VALUES_KEY = 'min_max'
  MAX_INT_VALUE_PARAMS_KEY = 'max_value'
  # The allowed values for `bits` argument to initializer.
  # We cap the allowed quantization bits at 16, as the randomized rounding could
  # otherwise be numerically unstable for float32 values.
  _ALLOWED_BITS_ARG = list(range(1, 17))

  def __init__(self, bits=8, min_max=None, stochastic=True):
    """Initializer for the UniformQuantizationEncodingStage.

    Args:
      bits: The number of bits to quantize to. Must be an integer between 1 and
        16. Can be either a TensorFlow or a Python value.
      min_max: A range to be used for quantization. If `None`, the range of the
        vector to be encoded will be used. If provided, must be an array of 2
        elements, corresponding to the min and max value, respectively. Can be
        either a TensorFlow or a Python value.
      stochastic: A Python bool, whether to use stochastic or deterministic
        rounding. If `True`, the encoding is randomized and on expectation
        unbiased. If `False`, the encoding is deterministic.

    Raises:
      ValueError: The inputs do not satisfy the above constraints.
    """
    if (not tf.is_tensor(bits) and bits not in self._ALLOWED_BITS_ARG):
      raise ValueError('The bits argument must be an integer between 1 and 16.')
    self._bits = bits

    if min_max is not None:
      if tf.is_tensor(min_max):
        if min_max.shape.as_list() != [2]:
          raise ValueError(
              'The min_max argument must be Tensor with shape (2).')
      else:
        if not isinstance(min_max, list) or len(min_max) != 2:
          raise ValueError(
              'The min_max argument must be a list with two elements.')
        if min_max[0] >= min_max[1]:
          raise ValueError('The first element of the min_max argument must be '
                           'smaller than the second element.')
    self._min_max = min_max

    if not isinstance(stochastic, bool):
      raise TypeError('The stochastic argument must be a bool.')
    self._stochastic = stochastic

  @property
  def name(self):
    """See base class."""
    return 'uniform_quantization'

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return [self.ENCODED_VALUES_KEY]

  @property
  def commutes_with_sum(self):
    """See base class."""
    # The stage commutes with sum only if min_max values are shared.
    if self._min_max is not None:
      return True
    else:
      return False

  @property
  def decode_needs_input_shape(self):
    """See base class."""
    return False

  def get_params(self):
    """See base class."""
    params = {self.MAX_INT_VALUE_PARAMS_KEY: 2**self._bits - 1}
    if self._min_max is not None:
      # If fixed min and max is provided, expose them via params.
      params[self.MIN_MAX_VALUES_KEY] = self._min_max
    return params, params

  def encode(self, x, encode_params):
    """See base class."""
    if self.MIN_MAX_VALUES_KEY in encode_params:
      min_max = tf.cast(encode_params[self.MIN_MAX_VALUES_KEY], x.dtype)
      min_x, max_x = min_max[0], min_max[1]
      x = tf.clip_by_value(x, min_x, max_x)
    else:
      min_x = tf.reduce_min(x)
      max_x = tf.reduce_max(x)

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

    encoded_tensors = {self.ENCODED_VALUES_KEY: quantized_x}
    if self.MIN_MAX_VALUES_KEY not in encode_params:
      encoded_tensors[self.MIN_MAX_VALUES_KEY] = tf.stack([min_x, max_x])
    return encoded_tensors

  def decode(self,
             encoded_tensors,
             decode_params,
             num_summands=None,
             shape=None):
    """See base class."""
    del shape  # Unused.
    quantized_x = encoded_tensors[self.ENCODED_VALUES_KEY]
    if self.MIN_MAX_VALUES_KEY in decode_params:
      min_max = tf.cast(decode_params[self.MIN_MAX_VALUES_KEY],
                        quantized_x.dtype)
    else:
      min_max = encoded_tensors[self.MIN_MAX_VALUES_KEY]
    min_x, max_x = min_max[0], min_max[1]
    max_value = tf.cast(decode_params[self.MAX_INT_VALUE_PARAMS_KEY],
                        quantized_x.dtype)

    # If num_summands is None, it is to be interpreted as 1.
    if self.commutes_with_sum and num_summands is not None:
      shift = min_x * tf.cast(num_summands, min_x.dtype)
    else:
      shift = min_x

    x = quantized_x / max_value * (max_x - min_x) + shift
    return x


@encoding_stage.tf_style_encoding_stage
class BitpackingEncodingStage(encoding_stage.EncodingStageInterface):
  """Encoding stage for bitpacking values into an integer type.

  This class performs a lossless transformation, and realizes representation
  savings.

  The encode method expects integer values in range `[0, 2**input_bits-1]` in a
  floating point type (`tf.float32` or `tf.float64`). It packs the values to
  `tf.int32` type, and returns a rank 1 `Tensor` of packed values. The packed
  values are in the range `[0, 2**28-1]`, as the serialization in protocol
  buffer for this type is varint, and this thus ensures every element fits into
  4 bytes.
  """

  ENCODED_VALUES_KEY = 'bitpacked_values'
  DUMMY_TYPE_VALUES_KEY = 'dummy_type_value'
  _ALLOWED_INPUT_BITS_ARG = list(range(1, 17))

  def __init__(self, input_bits):
    """Initializer for the UniformQuantizationEncodingStage.

    Args:
      input_bits: The number of bits expected to represent the input to the
        `encode` method. Must be between 1 and 16. Cannot be a TensorFlow value.

    Raises:
      TypeError: If `input_bits` is a TensorFlow value.
      ValueError: If `input_bits` is not between 1 and 16.
    """
    if tf.is_tensor(input_bits):
      raise TypeError('The input_bits argument cannot be a TensorFlow value.')
    if input_bits not in self._ALLOWED_INPUT_BITS_ARG:
      raise ValueError(
          'The input_bits argument must be an integer between 1 and 16.')
    self._input_bits = input_bits

    # Because the proto serialization format for integers is varint, we pack to
    # 28 bits, ensuring each serialized value is represented by 4 bytes.
    self._target_bitrange = 28

  @property
  def name(self):
    """See base class."""
    return 'bitpacking'

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return []  # Bitpacked values should not be further modified.

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
    return {}, {}

  def encode(self, x, encode_params):
    """See base class."""
    del encode_params
    flat_x = tf.reshape(x, [-1])
    packed_x = tf_utils.pack_into_int(
        tf.cast(flat_x, tf.int32), self._input_bits, self._target_bitrange)

    # The most common type will be tf.float32, which we keep as default.
    # If another type is provided, return a Tensor with a single value of that
    # type to be able to recover the type from encoded_tensors in decode method.
    if x.dtype == tf.float32:
      return {self.ENCODED_VALUES_KEY: packed_x}
    elif x.dtype == tf.float64:
      return {self.ENCODED_VALUES_KEY: packed_x,
              self.DUMMY_TYPE_VALUES_KEY: tf.constant(0.0, dtype=tf.float64)}
    else:
      raise TypeError(
          'Unsupported packing type: %s. Supported types are tf.float32 and '
          'tf.float64 values' % x.dtype)

  def decode(self,
             encoded_tensors,
             decode_params,
             num_summands=None,
             shape=None):
    """See base class."""
    del decode_params, num_summands  # Unused.
    unpacked_x = tf_utils.unpack_from_int(
        encoded_tensors[self.ENCODED_VALUES_KEY], self._input_bits,
        self._target_bitrange, shape)

    dummy_type_value = encoded_tensors.get(self.DUMMY_TYPE_VALUES_KEY)
    if dummy_type_value is not None:
      return tf.cast(unpacked_x, dummy_type_value.dtype)
    else:
      return tf.cast(unpacked_x, tf.float32)
