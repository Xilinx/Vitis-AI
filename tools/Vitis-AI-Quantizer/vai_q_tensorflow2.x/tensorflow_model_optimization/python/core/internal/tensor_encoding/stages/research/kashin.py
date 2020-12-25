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
"""Experimental implementations of the encoding stage interfaces."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_model_optimization.python.core.internal.tensor_encoding.core import encoding_stage
from tensorflow_model_optimization.python.core.internal.tensor_encoding.utils import tf_utils


@encoding_stage.tf_style_encoding_stage
class KashinHadamardEncodingStage(encoding_stage.EncodingStageInterface):
  """Encoding stage computing Kashin's representation from Hadamard transform.

  This class is inspired by the work "Uncertainty Principle and Vector
  Quantization" (https://arxiv.org/pdf/math/0611343.pdf), and the algorithm
  presented therein.

  The encoding stage builds upon the `HadamardEncodingStage` and uses the
  overcomplete base to compute representation with a smaller dynamic range of
  coefficients. One of the benefits is that it incurrs a smaller error in
  subsequent uniform quantization, smaller even compared to using the
  `HadamardEncodingStage`.

  The idea is that certain overcomplete bases, or tight frames, admit what is
  called "Kashin's representation" for every vector. Kashin's representation is
  such that the dynamic range of coefficients is as small as (asymptotically)
  possible. In this case, the overcomplete base consists of subsampled Hadamard
  matrix.

  The algorithm iteratively computes coefficients of input vector in rotated
  space (or projected space), projects onto a L-infinity ball, computes the
  residual error incurred by the projection, and repeats the same procedure with
  the residual and a smaller ball.

  For the meaning of the parameters `eta` and `delta`, see Section III.A in the
  referenced paper. The main pseudocode presented there is as follows:
    Input:
      projection, parameters eta, delta (should be derived from the projection).
      vector x of which representation needs to be computed.
    Initialize:
      kashin_coefficients <- [0, 0, ..., 0]
      M <- norm(x) / sqrt(delta * output_dim)  # [initial clipping level]
    Repeat num_iter times:
      b <- frame representation of x
      b <- clip b at level M
      kashin_coefficients <- kashin_coefficients + b
      x' <- recover from frame representation b
      x <- x - x'
      M <- M * eta
    Output:
      kashin_coefficients

  The actual implementation of the encoding works as follows:
  The shape of the input `x` to the `encode` method must be either `(dim)` or
  `(b, dim)`, where `dim` is the dimenion of the vector to which the transform
  is to be applied, and must be statically known. `b` represents an optional
  batch dimension, and does not need to be statically known.

  If the shape of the input is `(dim)`, it is first expanded to `(1, dim)`. The
  input of shape `(b, dim)` is then padded with zeros to dimension `(b, dim_2)`,
  where `dim_2` is the smallest power of 2 larger than or equal to `dim`. In the
  case of `dim / dim_2 > pad_extra_level_threshold`, an extra factor of 2 is
  added to `dim_2`. Otherwise, the algorithm does not have enough leeway to
  realize benefits over only applying the randomized Hadamard transform.

  The Kashin's representation is then computed for each of the `b` vectors of
  shape `dim_2`. The encoded value thus has shape `(b, dim_2)`.
  """

  ENCODED_VALUES_KEY = 'kashin_hadamard_values'
  ETA_PARAMS_KEY = 'eta'
  DELTA_PARAMS_KEY = 'delta'
  SEED_PARAMS_KEY = 'seed'

  def __init__(self,
               num_iters=3,
               eta=0.9,
               delta=1.0,
               last_iter_clip=False,
               pad_extra_level_threshold=0.85):
    """Initializer for the `KashinHadamardEncodingStage`.

    Args:
      num_iters: An integer number of iterations to run to compute the
        representation. Cannot be a TensorFlow value.
      eta: A scalar parameter of the encoding algorithm. Determines the
        shrinkage of the clipping level in each iteration. Must be between 0 and
        1. Can be either a TensorFlow or a Python value.
      delta: A scalar parameter of the encoding algorithm. Determines the
        initial clipping level. Must be greater than 0. Can be either a
        TensorFlow or a Python value.
      last_iter_clip: A boolean, determining whether to apply clipping in last
        iteration of the encoding algorithm. If set to False, the encoded
        representation is always lossless. If set to True, the resulting
        representation can be lossy, although not necessarily. Cannot be a
        TensorFlow value.
      pad_extra_level_threshold: A scalar parameter determining the threshold,
        at which padding with zeros is to be expanded by an additional power of
        2. See class documentation for why this is needed. Cannot be a
        TensorFlow value.

    Raises:
      ValueError: The inputs do not satisfy the above constraints.
    """
    if tf.is_tensor(num_iters):
      raise ValueError('Parameter num_iters cannot be a TensorFlow value.')
    if not isinstance(num_iters, int) or num_iters <= 0:
      raise ValueError('Number of iterations must be a positive integer.'
                       'num_iters provided: %s' % num_iters)
    self._num_iters = num_iters

    if not tf.is_tensor(eta) and not 0.0 < eta < 1.0:
      raise ValueError('Parameter eta must be between 0 and 1. '
                       'Provided eta: %s' % eta)
    self._eta = eta

    if not tf.is_tensor(delta) and delta <= 0.0:
      raise ValueError('Parameter delta must be greater than 0. '
                       'Provided delta: %s' % delta)
    self._delta = delta

    if tf.is_tensor(last_iter_clip):
      raise ValueError('Parameter last_iter_clip cannot be a TensorFlow value.')
    if not isinstance(last_iter_clip, bool):
      raise ValueError('Parameter last_iter_clip must be a bool.')
    self._last_iter_clip = last_iter_clip

    if tf.is_tensor(pad_extra_level_threshold):
      raise ValueError(
          'Parameter pad_extra_level_threshold cannot be a TensorFlow value.')
    self._pad_extra_level_threshold = pad_extra_level_threshold

  @property
  def name(self):
    """See base class."""
    return 'kashin_hadamard'

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
    seed = tf.random.uniform((2,), maxval=tf.int64.max, dtype=tf.int64)
    encode_params = {
        self.ETA_PARAMS_KEY: self._eta,
        self.DELTA_PARAMS_KEY: self._delta,
        self.SEED_PARAMS_KEY: seed,
    }
    decode_params = {self.SEED_PARAMS_KEY: seed}
    return encode_params, decode_params

  def encode(self, x, encode_params):
    """See base class."""
    x = self._validate_and_expand_encode_input(x)
    dims = x.shape.as_list()
    # Get static or dynamic leading dimension if static not available.
    dim_0 = dims[0] if dims[0] else tf.shape(x)[0]
    dim_1 = dims[1]
    kashin_coefficients = tf.zeros([dim_0, self._get_pad_dim(dim_1)],
                                   dtype=x.dtype)
    clip_level = tf.norm(x, axis=1, keepdims=True) / tf.math.sqrt(
        tf.cast(encode_params[self.DELTA_PARAMS_KEY], x.dtype) * dim_1)
    last_iter_clip = self._last_iter_clip
    residual = x
    signs = self._random_signs(dim_1, encode_params[self.SEED_PARAMS_KEY],
                               x.dtype)

    # Compute the Kashin coefficients.
    for _ in range(self._num_iters - 1):
      residual, kashin_coefficients = self._kashin_iter(
          residual, kashin_coefficients, signs, clip_level)
      clip_level *= tf.cast(encode_params[self.ETA_PARAMS_KEY], x.dtype)
    # The last iteration can be with or without clipping.
    kashin_coefficients += self._kashin_forward(residual, signs, clip_level,
                                                last_iter_clip)
    if last_iter_clip:
      # If there is clipping in the last iteration, this can result in
      # biased representation of smaller magnitude. We compensate for this
      # by scaling such that the norm is preserved.
      kashin_coefficients *= tf.compat.v1.div_no_nan(
          tf.norm(x, axis=1, keepdims=True),
          tf.norm(kashin_coefficients, axis=1, keepdims=True))

    return {self.ENCODED_VALUES_KEY: kashin_coefficients}

  def decode(self,
             encoded_tensors,
             decode_params,
             num_summands=None,
             shape=None):
    """See base class."""
    del num_summands  # Unused.
    kashin_coefficients = encoded_tensors[self.ENCODED_VALUES_KEY]
    decoded_x = self._kashin_backward(kashin_coefficients, shape[-1])
    signs = self._random_signs(decoded_x.shape.as_list()[-1],
                               decode_params[self.SEED_PARAMS_KEY],
                               decoded_x.dtype)
    decoded_x = decoded_x * signs

    if shape.shape.num_elements() == 1:
      decoded_x = tf.squeeze(decoded_x, [0])
    return decoded_x

  def _kashin_forward(self, x, signs, clip_level, clip):
    """Forward step of the algorithm to obtain Kashin's representation."""
    x = x * signs
    x = self._pad(x)
    x = tf_utils.fast_walsh_hadamard_transform(x)
    if clip:
      x = tf.clip_by_value(x, -clip_level, clip_level)
    return x

  def _kashin_backward(self, x, shape, signs=None):
    """Backward step of the algorithm to obtain Kashin's representation."""
    x = tf_utils.fast_walsh_hadamard_transform(x)
    # Take the slice corresponding to the original object that was encoded.
    # Consistency in specific coordinates for padding and slicing is what makes
    # inverse transformation unique.
    x = tf.slice(x, [0, 0], [tf.shape(x)[0], shape])
    if signs is not None:
      x = x * signs
    return x

  def _kashin_iter(self, x, kashin_coefficients, signs, clip_level):
    """A single iteration of the algorithm to obtain Kashin's representation."""
    x_init = x
    x = self._kashin_forward(x, signs, clip_level, clip=True)
    kashin_coefficients += x
    # x_init.shape.as_list()[1] is the dimension of objects to be encoded.
    x = self._kashin_backward(x, x_init.shape.as_list()[1], signs)
    residual = x_init - x
    return residual, kashin_coefficients

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

  def _get_pad_dim(self, dim):
    """Computes the dimension the input needs to be padded into."""
    pad_dim = 2**int(np.ceil(np.log2(dim)))
    if dim / pad_dim > self._pad_extra_level_threshold:
      pad_dim *= 2
    return pad_dim

  def _pad(self, x):
    """Pads with zeros to the next power of two."""
    dim = x.shape.as_list()[1]
    pad_dim = self._get_pad_dim(dim)
    if pad_dim != dim:
      x = tf.pad(x, [[0, 0], [0, pad_dim - dim]])
    return x

  def _random_signs(self, num_elements, seed, dtype):
    return tf_utils.random_signs(num_elements, seed, dtype)
