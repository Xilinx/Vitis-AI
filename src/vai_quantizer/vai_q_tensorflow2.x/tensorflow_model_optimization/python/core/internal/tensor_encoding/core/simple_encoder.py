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
"""Base Encoder class for encoding in the "one-to-many" case."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_model_optimization.python.core.internal.tensor_encoding.core import core_encoder
from tensorflow_model_optimization.python.core.internal.tensor_encoding.utils import py_utils

_TENSORS = 'encoded_tensors'
_PARAMS = 'params'
_SHAPES = 'shapes'


class SimpleEncoder(object):
  """A simple class for encoding.

  This class provides functionality for encoding in the "one-to-many" case,
  where a `Tensor` is encoded in one location, and is to be decoded at
  potentially many other locations, leaving the communication between encoding
  and decoding up to the user.

  An instance of `SimpleEncoder` is capable of encoding only values of a shape
  and dtype, as specified at construction time. For example, an separate
  instance of this class should be used for encoding every `Variable` of a
  model, as opposed to a single instance being reused for each `Variable`.

  `SimpleEncoder` exposes the state of the underlying encoder, and the user is
  responsible for keeping track of the state in case the encoding should be
  adaptive as part of an iterative process. If state is not needed, it can be
  simply ignored when calling the `encode` method.
  """

  def __init__(self, encoder, tensorspec):
    """Creates a `SimpleEncoder` for encoding `tensorspec`-like values.

    This method instantiates `SimpleEncoder`, wrapping the functionality of
    `encoder` and exposing necessary logic for encoding values compatible with
    `tensorspec`. Note that the returned encoder will not accept inputs of other
    properties.

    Args:
      encoder: An `Encoder` object to be used for encoding.
      tensorspec: A `tf.TensorSpec`. The created `SimpleEncoder` will be
        constrained to only encode input values compatible with `tensorspec`.

    Returns:
      A `SimpleEncoder`.

    Raises:
      TypeError:
        If `encoder` is not an `Encoder` or `tensorspec` is not a
        `tf.TensorSpec`.
    """
    if not isinstance(encoder, core_encoder.Encoder):
      raise TypeError('The encoder must be an instance of `Encoder`.')
    if not isinstance(tensorspec, tf.TensorSpec):
      raise TypeError('The tensorspec must be a tf.TensorSpec.')
    if not tensorspec.shape.is_fully_defined():
      raise TypeError('The shape of provided tensorspec must be fully defined.')
    self._tensorspec = tensorspec

    # These dictionaries are filled inside of the initial_state_fn and encode_fn
    # methods, to be used in encode_fn and decode_fn methods, respectively.
    # Decorated by tf.function, their necessary side effects are realized during
    # call to get_concrete_function().
    state_py_structure = {}
    encoded_py_structure = {}

    @tf.function
    def initial_state_fn():
      state = encoder.initial_state()
      if not state_py_structure:
        state_py_structure['state'] = tf.nest.map_structure(
            lambda _: None, state)
      # Simplify the structure that needs to be manipulated by the user.
      return tuple(tf.nest.flatten(state))

    @tf.function(input_signature=[
        tensorspec,
        tf.nest.map_structure(
            tf.TensorSpec.from_tensor,
            initial_state_fn.get_concrete_function().structured_outputs)
    ])  # pylint: disable=missing-docstring
    def encode_fn(x, flat_state):
      state = tf.nest.pack_sequence_as(state_py_structure['state'], flat_state)
      encode_params, decode_params = encoder.get_params(state)
      encoded_x, state_update_tensors, input_shapes = encoder.encode(
          x, encode_params)
      updated_flat_state = tuple(
          tf.nest.flatten(encoder.update_state(state, state_update_tensors)))

      # The following code converts the nested structres necessary for the
      # underlying encoder, to a single flat dictionary, which is simpler to
      # manipulate by the users of SimpleEncoder.
      full_encoded_structure = {
          _TENSORS: encoded_x,
          _PARAMS: decode_params,
          _SHAPES: input_shapes
      }
      flat_encoded_structure = dict(
          py_utils.flatten_with_joined_string_paths(full_encoded_structure))
      flat_encoded_py_structure, flat_encoded_tf_structure = (
          py_utils.split_dict_py_tf(flat_encoded_structure))

      if not encoded_py_structure:
        encoded_py_structure['full'] = tf.nest.map_structure(
            lambda _: None, full_encoded_structure)
        encoded_py_structure['flat_py'] = flat_encoded_py_structure
      return flat_encoded_tf_structure, updated_flat_state

    @tf.function(input_signature=[
        tf.nest.map_structure(
            tf.TensorSpec.from_tensor,
            encode_fn.get_concrete_function().structured_outputs[0])
    ])  # pylint: disable=missing-docstring
    def decode_fn(encoded_structure):
      encoded_structure = py_utils.merge_dicts(encoded_structure,
                                               encoded_py_structure['flat_py'])
      encoded_structure = tf.nest.pack_sequence_as(
          encoded_py_structure['full'], tf.nest.flatten(encoded_structure))
      return encoder.decode(encoded_structure[_TENSORS],
                            encoded_structure[_PARAMS],
                            encoded_structure[_SHAPES])

    # Ensures the decode_fn is traced during initialization.
    decode_fn.get_concrete_function()

    self._initial_state_fn = initial_state_fn
    self._encode_fn = encode_fn
    self._decode_fn = decode_fn

  @property
  def input_tensorspec(self):
    """Returns `tf.TensorSpec` describing input expected by `SimpleEncoder`."""
    return self._tensorspec

  def initial_state(self, name=None):
    """Returns the initial state.

    Args:
      name: `string`, name of the operation.

    Returns:
      A tuple of `Tensor` values, representing the initial state.
    """
    with tf.compat.v1.name_scope(name, 'simple_encoder_initial_state'):
      return self._initial_state_fn()

  def encode(self, x, state=None, name=None):
    """Encodes the provided input.

    If `state` is not provided, the return value of the `initial_state` method
    will be used.

    Args:
      x: A `Tensor` to be encoded.
      state: The (optional) current state. A tuple, matching the structure
        returned by the `initial_state` method.
      name: `string`, name of the operation.

    Returns:
      A `(encoded_x, updated_state)` tuple, where `encoded_x` is a dictionary of
      `Tensor` values representing the encoded `x`, and `updated_state` is the
      state updated after encoding, of the same structure as `state`.

    Raises:
      ValueError:
        If `x` does not have the expected shape or dtype, or if `state` does not
        have the same structure as return value of the `initial_state` method.
    """
    if state is None:
      state = self.initial_state()
    with tf.compat.v1.name_scope(name, 'simple_encoder_encode',
                                 [x] + list(state)):
      return self._encode_fn(x, state)

  def decode(self, encoded_x, name=None):
    """Decodes the encoded value.

    Args:
      encoded_x: A dictionary of the same structure as returned by the `encode`
        method. Represents the encoded value to be decoded.
      name: `string`, name of the operation.

    Returns:
      A single `Tensor` of the same shape and dtype as the original input to the
      `encode` method.

    Raises:
      ValueError:
        If `encoded_x` is not of the same structure as returned by the `encode`
        method.
    """
    with tf.compat.v1.name_scope(name, 'simple_encoder_decode',
                                 encoded_x.values()):
      return self._decode_fn(encoded_x)
