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
"""Base Encoder class for encoding in the "many-to-one" case."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_model_optimization.python.core.internal.tensor_encoding.core import core_encoder
from tensorflow_model_optimization.python.core.internal.tensor_encoding.utils import py_utils

_PARAMS = 'params'
_SHAPES = 'shapes'
_TENSORS = 'tensors'


class GatherEncoder(object):
  """A class for a gather-like operations with encoding.

  This class provides functionality for encoding in the "many-to-one" case,
  where multiple locations hold a `Tensor` of the same shape and dtype, and one
  needs to compute their sum at a central location, while only encoded
  representations are communicated between the locations.

  An instance of `GatherEncoder` is capable of encoding only values of a shape
  and dtype, as specified at construction time. For example, a separate
  instance of this class should be used for encoding every `Variable` of a
  model, as opposed to a single instance being reused for each `Variable`.

  `GatherEncoder` exposes the state of the underlying encoder, and the user is
  responsible for keeping track of the state in case the encoding should be
  adaptive as part of an iterative process.

  In the following illustration of a typical pattern of usage of this class, we
  will refer to the "one" central location as `server` and the "many" locations
  as `workers`. By `user`, we denote the parts that are the responsibility of
  the user of this class -- for instance, the communication between the `server`
  and `workers` might need to happen outside of TensorFlow, depending on the
  target deployment.

  1.  `[server]` Use the `initial_state` method to create an initial `state`.
  2.  `[server]` Use the `get_params` method to derive `encode_params`,
      `decode_before_sum_params` and `decode_after_sum_params` for the remaining
      methods, based on the `state`.
  3.  `[user]` Make the `encode_params` available to the `workers`.
  4.  `[workers]` Use the `encode` method to get `encoded_x` and
      `state_update_tensors`.
  5.  `[user]` Make the `encoded_x` and `state_update_tensors` available to the
      `server`.
  6.  `[server]` Use the `decode_before_sum` method to partially decode all of
      the `encoded_x` values.
  7.  `[user]` Sum the partially decoded values.
  8.  `[server]` Use the `decode_after_sum` method to finish decoding the summed
      and partially decoded values.
  9.  `[user]` Aggregate the `state_update_tensors` according to aggregation
      modes declared by the `state_update_aggregation_modes` property.
  10. `[server]` Use the `update_state` method to update the `state` based on
      the existing `state` and aggregated `state_update_tensors`.

  NOTE Step 5 could overlap with the steps 6--9, depending on the indended
  deployment. For instnace, if communication is realized using a multi-tier
  aggregation architecture, intermediary nodes would need to access
  `decode_before_sum_params` and use the `decode_before_sum` method and sum the
  part decoded representations. These would be then summed again at the
  `server`, which would finish the decoding.

  NOTE The use of the `state` is optional. It is needed only when the encoding
  mechanism should adapt based on the values being encoded during an iterative
  execution.
  """

  def __init__(self, tensorspec, commuting_structure,
               state_update_aggregation_modes, initial_state_fn, get_params_fn,
               encode_fn, decode_before_sum_fn, decode_after_sum_fn,
               update_state_fn):
    """Creates a `GatherEncoder` for encoding `tensorspec`-like values.

    This class should not be instantiated directly. Instead, use the
    provided `@classmethod`.

    Args:
      tensorspec: A `tf.TensorSpec`. The created `GatherEncoder` will be
        constrained to only encode input values compatible with `tensorspec`.
      commuting_structure: The commuting structure of the `GatherEncoder`.
      state_update_aggregation_modes: The `StageAggregationMode` values to be
        used to aggregate `state_update_tensors`
      initial_state_fn: A `tf.function`.
      get_params_fn: A `tf.function`.
      encode_fn: A `tf.function`.
      decode_before_sum_fn: A `tf.function`.
      decode_after_sum_fn: A `tf.function`.
      update_state_fn: A `tf.function`.

    Returns:
      A `GatherEncoder`.
    """
    self._tensorspec = tensorspec
    self._commuting_structure = commuting_structure
    self._state_update_aggregation_modes = state_update_aggregation_modes

    self._initial_state_fn = initial_state_fn
    self._get_params_fn = get_params_fn
    self._encode_fn = encode_fn
    self._decode_before_sum_fn = decode_before_sum_fn
    self._decode_after_sum_fn = decode_after_sum_fn
    self._update_state_fn = update_state_fn

  @classmethod
  def from_encoder(cls, encoder, tensorspec):
    """Creates a `GatherEncoder` for encoding `tensorspec`-like values.

    This method instantiates `GatherEncoder`, wrapping the functionality of
    `encoder` and exposing necessary logic for encoding values compatible with
    `tensorspec`. Note that the returned encoder will not accept inputs of other
    properties.

    Args:
      encoder: An `Encoder` object to be used for encoding.
      tensorspec: A `tf.TensorSpec`. The created `GatherEncoder` will be
        constrained to only encode input values compatible with `tensorspec`.

    Returns:
      A `GatherEncoder`.

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

    tensorspec = tensorspec
    commuting_structure = encoder.commuting_structure
    state_update_aggregation_modes = tf.nest.flatten(
        encoder.state_update_aggregation_modes)

    # The following dictionaries are used to carry information known statically
    # during exectuion of the Python code (i.e., not the resulting TensorFlow
    # computation) between the `tf.function`s created below.
    #
    # The motivation behind this pattern is the following.
    #
    # The implementers of the `EncodingStageInterface` should not need to worry
    # about distinction between Python and TF values, when declaring certain
    # parameters. For instance, the number of quantization bits, can be both a
    # TenrosFlow value and a Python integer, and they should not need to be
    # handled differently by the implementer.
    #
    # However, for the user of the `GatherEncoder`, we only want to expose
    # values that are actually necessary to be handled outside of this tool.
    # That means, only the TF values. We quietly carry the Python values around
    # - in the internal_structure and internal_py_values dictionaries - and
    # place them at appropriate places at graph building time.
    #
    # As a consequence, it is impossible to statically determine the user-facing
    # signature of encode and decode methods, before we actually execute the
    # `get_params` method - the TF structure can depend on internal
    # configuration of the implementations of the `EncodingStageInterface`.
    #
    # A similar problem is we can't determine the signature of the decode
    # methods, before executing the encode method, because some implementations
    # of `EncodingStageInterface` need the original input_shape as an input to
    # their respective `decode` method. Hence, the user facing signature can
    # differ based on whether the shape is statically known or not. This
    # difference, again, can't be statically determined, without executing the
    # part of the relevant encoding tree above a given stage.
    #
    # The resulting complexity is of the good type, because either type of users
    # of the tensor_encoding tool do not even need to be aware of it. This
    # argument is well supported for instance in the book of John Ousterhout,
    # "A Philosophy of Software Design".
    internal_structure = {}
    internal_py_values = {}

    def _add_to_structure(key, value):
      if key not in internal_structure:
        internal_structure[key] = tf.nest.map_structure(lambda _: None, value)

    def _add_to_py_values(key, value):
      if key not in internal_py_values:
        internal_py_values[key] = value

    @tf.function
    def initial_state_fn():
      """See the `initial_state` method of this class."""
      state = encoder.initial_state()
      _add_to_structure('state', state)
      return tuple(tf.nest.flatten(state))

    state = initial_state_fn()
    flat_state_spec = tf.nest.map_structure(tf.TensorSpec.from_tensor, state)

    @tf.function
    def get_params_fn(flat_state):
      """See the `get_params` method of this class."""
      py_utils.assert_compatible(flat_state_spec, flat_state)
      state = tf.nest.pack_sequence_as(internal_structure['state'], flat_state)

      encode_params, decode_params = encoder.get_params(state)
      decode_before_sum_params, decode_after_sum_params = (
          core_encoder.split_params_by_commuting_structure(
              decode_params, commuting_structure))

      # Get the portion of input_shapes that will be relevant in the
      # decode_after_sum method and fold it into the params exposed to user.
      _, _, input_shapes = encoder.encode(
          tf.zeros(tensorspec.shape, tensorspec.dtype), encode_params)
      _, input_shapes_after_sum = (
          core_encoder.split_shapes_by_commuting_structure(
              input_shapes, commuting_structure))
      decode_after_sum_params = {
          _PARAMS: decode_after_sum_params,
          _SHAPES: input_shapes_after_sum
      }

      encode_params_py, encode_params_tf = py_utils.split_dict_py_tf(
          encode_params)
      decode_before_sum_params_py, decode_before_sum_params_tf = (
          py_utils.split_dict_py_tf(decode_before_sum_params))
      decode_after_sum_params_py, decode_after_sum_params_tf = (
          py_utils.split_dict_py_tf(decode_after_sum_params))

      _add_to_structure('encode_params', encode_params_tf)
      _add_to_structure('decode_before_sum_params', decode_before_sum_params_tf)
      _add_to_structure('decode_after_sum_params', decode_after_sum_params_tf)
      _add_to_py_values('encode_params', encode_params_py)
      _add_to_py_values('decode_before_sum_params', decode_before_sum_params_py)
      _add_to_py_values('decode_after_sum_params', decode_after_sum_params_py)

      return (tuple(tf.nest.flatten(encode_params_tf)),
              tuple(tf.nest.flatten(decode_before_sum_params_tf)),
              tuple(tf.nest.flatten(decode_after_sum_params_tf)))

    encode_params, decode_before_sum_params, decode_after_sum_params = (
        get_params_fn(state))
    encode_params_spec = tf.nest.map_structure(tf.TensorSpec.from_tensor,
                                               encode_params)
    decode_before_sum_params_spec = tf.nest.map_structure(
        tf.TensorSpec.from_tensor, decode_before_sum_params)
    decode_after_sum_params_spec = tf.nest.map_structure(
        tf.TensorSpec.from_tensor, decode_after_sum_params)

    @tf.function
    def encode_fn(x, params):
      """See the `encode` method of this class."""
      if not tensorspec.is_compatible_with(x):
        raise ValueError(
            'The provided x is not compatible with the expected tensorspec.')
      py_utils.assert_compatible(encode_params_spec, params)

      params = py_utils.merge_dicts(
          tf.nest.pack_sequence_as(internal_structure['encode_params'], params),
          internal_py_values['encode_params'])
      encoded_x, state_update_tensors, input_shapes = encoder.encode(x, params)
      input_shapes_before_sum, _ = (
          core_encoder.split_shapes_by_commuting_structure(
              input_shapes, commuting_structure))

      encoded_structure = {
          _TENSORS: encoded_x,
          _SHAPES: input_shapes_before_sum
      }
      encoded_structure_py, encoded_structure_tf = py_utils.split_dict_py_tf(
          encoded_structure)

      _add_to_structure('encoded_structure', encoded_structure_tf)
      _add_to_structure('state_update_tensors', state_update_tensors)
      _add_to_py_values('encoded_structure', encoded_structure_py)

      return (dict(
          py_utils.flatten_with_joined_string_paths(encoded_structure_tf)),
              tuple(tf.nest.flatten(state_update_tensors)))

    encoded_structure, state_update_tensors = encode_fn(
        tf.zeros(tensorspec.shape, tensorspec.dtype), encode_params)
    encoded_structure_spec = tf.nest.map_structure(tf.TensorSpec.from_tensor,
                                                   encoded_structure)

    @tf.function
    def decode_before_sum_fn(encoded_structure, params):
      """See the `decode_before_sum` method of this class."""
      py_utils.assert_compatible(encoded_structure_spec, encoded_structure)
      py_utils.assert_compatible(decode_before_sum_params_spec, params)

      encoded_structure = py_utils.merge_dicts(
          tf.nest.pack_sequence_as(internal_structure['encoded_structure'],
                                   tf.nest.flatten(encoded_structure)),
          internal_py_values['encoded_structure'])
      params = py_utils.merge_dicts(
          tf.nest.pack_sequence_as(
              internal_structure['decode_before_sum_params'], params),
          internal_py_values['decode_before_sum_params'])

      encoded_tensors = encoded_structure[_TENSORS]
      input_shapes = encoded_structure[_SHAPES]
      part_decoded_structure = encoder.decode_before_sum(
          encoded_tensors, params, input_shapes)

      _add_to_structure('part_decoded_structure', part_decoded_structure)
      if isinstance(part_decoded_structure, dict):
        return dict(
            py_utils.flatten_with_joined_string_paths(part_decoded_structure))
      else:
        return part_decoded_structure

    part_decoded_structure = decode_before_sum_fn(encoded_structure,
                                                  decode_before_sum_params)
    part_decoded_structure_spec = tf.nest.map_structure(
        tf.TensorSpec.from_tensor, part_decoded_structure)

    @tf.function
    def decode_after_sum_fn(part_decoded_structure, params, num_summands):
      """See the `decode_after_sum` method of this class."""
      py_utils.assert_compatible(part_decoded_structure_spec,
                                 part_decoded_structure)
      py_utils.assert_compatible(decode_after_sum_params_spec, params)

      part_decoded_structure = tf.nest.pack_sequence_as(
          internal_structure['part_decoded_structure'],
          tf.nest.flatten(part_decoded_structure))
      params = py_utils.merge_dicts(
          tf.nest.pack_sequence_as(
              internal_structure['decode_after_sum_params'], params),
          internal_py_values['decode_after_sum_params'])
      actual_params = params[_PARAMS]
      shapes = params[_SHAPES]
      decoded_x = encoder.decode_after_sum(part_decoded_structure,
                                           actual_params, num_summands, shapes)
      return decoded_x

    decoded_x = decode_after_sum_fn(part_decoded_structure,
                                    decode_after_sum_params, 1)
    assert tensorspec.is_compatible_with(decoded_x)

    @tf.function
    def update_state_fn(flat_state, state_update_tensors):
      """See the `update_state` method of this class."""
      py_utils.assert_compatible(flat_state_spec, flat_state)
      state = tf.nest.pack_sequence_as(internal_structure['state'], flat_state)
      state_update_tensors = tf.nest.pack_sequence_as(
          internal_structure['state_update_tensors'], state_update_tensors)
      updated_state = encoder.update_state(state, state_update_tensors)
      return tuple(tf.nest.flatten(updated_state))

    # Ensures the update_state_fn is traced during initialization.
    updated_state = update_state_fn(state, state_update_tensors)
    tf.nest.assert_same_structure(state, updated_state)

    return cls(tensorspec, commuting_structure, state_update_aggregation_modes,
               initial_state_fn, get_params_fn, encode_fn, decode_before_sum_fn,
               decode_after_sum_fn, update_state_fn)

  @property
  def input_tensorspec(self):
    """Returns `tf.TensorSpec` describing input expected by `GatherEncoder`."""
    return self._tensorspec

  @property
  def fully_commutes_with_sum(self):
    # If any element is not True, the whole thing does not fully commute.
    return sum(tf.nest.flatten(self._commuting_structure))

  @property
  def state_update_aggregation_modes(self):
    """Returns `state_update_aggregation_modes` of the underlying `Encoder`."""
    return self._state_update_aggregation_modes

  def initial_state(self, name=None):
    """Returns the initial state.

    Args:
      name: `string`, name of the operation.

    Returns:
      A tuple of `Tensor` values, representing the initial state.
    """
    with tf.compat.v1.name_scope(name, 'gather_encoder_initial_state'):
      return self._initial_state_fn()

  def get_params(self, state=None, name=None):
    """Returns parameters controlling the behavior of the `GatherEncoder`.

    If `state` is not provided, the return value of the `initial_state` method
    will be used.

    Args:
      state: The (optional) current state. A tuple, matching the structure
        returned by the `initial_state` method.
      name: `string`, name of the operation.

    Returns:
      A tuple `(encode_params, decode_before_sum_params,
      decode_after_sum_params)`, where all of these are tuples of `Tensor`
      values, expected as inputs to the `encode`, `decode_before_sum` and
      `decode_after_sum` methods, respectively.

    Raises:
      ValueError:
        If `state` is not `None` and does not have the same structure as the
        return value of the `initial_state` method.
    """
    if state is None:
      state = self.initial_state()
    with tf.compat.v1.name_scope(name, 'gather_encoder_get_params',
                                 list(state)):
      state = tf.nest.map_structure(tf.convert_to_tensor, state)
      return self._get_params_fn(state)

  def encode(self, x, encode_params, name=None):
    """Encodes the provided input.

    Args:
      x: A `Tensor` to be encoded.
      encode_params: Parameters controlling the encoding. A tuple, matching the
        corresponding structure returned by the `get_params` method.
      name: `string`, name of the operation.

    Returns:
      A `(encoded_x, state_update_tensors)` tuple, where `encoded_x` is a
      dictionary of `Tensor` values representing the encoded `x`, and
      `state_update_tensors` is a tuple of `Tensor` values, which are expected
      to be aggregated according to modes provided by the
      `state_update_aggregation_modes` property, and afterwards passed to the
      `update_state` method.

    Raises:
      ValueError:
        If `x` does not have the expected shape or dtype, or if `encode_params`
        does not have the same structure as corresponding return value of the
        `get_params` method.
    """
    values = [x] + list(encode_params)
    with tf.compat.v1.name_scope(name, 'gather_encoder_encode', values):
      x = tf.convert_to_tensor(x)
      encode_params = tf.nest.map_structure(tf.convert_to_tensor, encode_params)
      return self._encode_fn(x, encode_params)

  def decode_before_sum(self, encoded_x, decode_before_sum_params, name=None):
    """Decodes encoded value, up to the point which commutes with sum.

    Args:
      encoded_x: A dictionary of `Tensor` values to be decoded. Must be of the
        same structure as the `encoded_x` returned by the `encode` method.
      decode_before_sum_params: Parameters controlling the decoding. A tuple,
        matching the corresponding structure returned by the `get_params`
        method.
      name: `string`, name of the operation.

    Returns:
      A part-decoded structure, which is expected to be summed before being
      passed to the `decode_after_sum` method. If no part of the underlying
      `Encoder` commutes with sum, this is a `Tensor`. If the `Encoder`
      partially or fully commutes with sum, this is a dictionary of `Tensor`
      values.

    Raises:
      ValueError:
        If `encoded_x` does not have the same structure as corresponding return
        value of the `encode` method, or if `decode_before_sum_params` does not
        have the same structure as corresponding return value of the
        `get_params` method.
    """
    values = list(encoded_x.values()) + list(decode_before_sum_params)
    with tf.compat.v1.name_scope(name, 'gather_encoder_decode_before_sum',
                                 values):
      encoded_x = tf.nest.map_structure(tf.convert_to_tensor, encoded_x)
      decode_before_sum_params = tf.nest.map_structure(
          tf.convert_to_tensor, decode_before_sum_params)
      return self._decode_before_sum_fn(encoded_x, decode_before_sum_params)

  def decode_after_sum(self,
                       part_decoded_x,
                       decode_after_sum_params,
                       num_summands,
                       name=None):
    """Finishes decoding of encoded value, after summing part-decoded values.

    Args:
      part_decoded_x: A dictionary of `Tensor` values to be decoded. Must be of
        the same structure as the `encoded_x` returned by the `encode` method.
      decode_after_sum_params: Parameters controlling the decoding. A tuple,
        matching the corresponding structure returned by the `get_params`
        method.
      num_summands: A `Tensor` representing the number of `part_decoded_x`
        values summed before passed into this method.
      name: `string`, name of the operation.

    Returns:
      A single `Tensor` of the same shape and dtype as the original input to the
      `encode` method.

    Raises:
      ValueError:
        If `part_decoded_x` does not have the same structure as the return value
        of the `decode_before_sum` method, or if `decode_after_sum_params` does
        not have the same structure as corresponding return value of the
        `get_params` method.
    """
    values = list(part_decoded_x.values()) if isinstance(
        part_decoded_x, dict) else [part_decoded_x]
    values = (values + list(decode_after_sum_params) + [num_summands])
    with tf.compat.v1.name_scope(name, 'gather_encoder_decode_after_sum',
                                 values):
      part_decoded_x = tf.nest.map_structure(tf.convert_to_tensor,
                                             part_decoded_x)
      decode_after_sum_params = tf.nest.map_structure(tf.convert_to_tensor,
                                                      decode_after_sum_params)
      num_summands = tf.convert_to_tensor(num_summands)
      return self._decode_after_sum_fn(part_decoded_x, decode_after_sum_params,
                                       num_summands)

  def update_state(self, state, state_update_tensors, name=None):
    """Updates the state of the `GatherEncoder`.

    Args:
      state: The (optional) current state. A tuple, matching the structure
        returned by the `initial_state` method.
      state_update_tensors: A tuple of `Tensor` values returned by the `encode`
        method, aggregated according to modes provided by the
        `state_update_aggregation_modes` property. Note that the tuple has the
        same structure, but the `Tensor` values it contains do not necessarily
        have the same shapes.
      name: `string`, name of the operation.

    Returns:
      A tuple of `Tensor` values of the same structure as `state`, representing
      the updated state.

    Raises:
      ValueError:
        If `state` is not `None` and does not have the same structure as the
        return value of the `initial_state` method.
    """
    values = list(state) + list(state_update_tensors)
    with tf.compat.v1.name_scope(name, 'gather_encoder_update_state', values):
      state = tf.nest.map_structure(tf.convert_to_tensor, state)
      state_update_tensors = tf.nest.map_structure(tf.convert_to_tensor,
                                                   state_update_tensors)
      return self._update_state_fn(state, state_update_tensors)
