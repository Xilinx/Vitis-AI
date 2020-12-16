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
"""Class responsible for composing individual encoding stages."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from tensorflow_model_optimization.python.core.internal.tensor_encoding.core import encoding_stage
from tensorflow_model_optimization.python.core.internal.tensor_encoding.utils import py_utils


# OrderedEnum necessary for compatibility with tf.nest.
class EncoderKeys(py_utils.OrderedEnum):
  """Constants for keys in nested structures in the `Encoder` class."""
  CHILDREN = 1
  PARAMS = 2
  SHAPE = 3
  STATE = 4
  TENSORS = 5
  COMMUTE = 6


class Encoder(object):
  """Class for composing individual encoding stages.

  This class provides functionality for arbitrarily composing individual
  encoding stages (implementations of `EncodingStageInterface` or
  `AdaptiveEncodingStageInterface`) into a tree, where an output `Tensor` of an
  encoding stage can be further encoded by another encoding stage.

  The interface is similar to that of `AdaptiveEncodingStageInterface`, with the
  additional methods `decode_before_sum` and `decode_after_sum`. These methods
  split the decoding functionality to two parts, based on commutativity with
  summation.

  Similar to the `AdaptiveEncodingStageInterface`, the methods are designed to
  be functional transformations. That means, for instance, that the
  `initial_state` and `update_state` methods do not modify any underlying state.
  Rather, the user of the `Encoder` class is responsible for storing any values
  if necessary, and providing them back to appropriate methods of this class.
  """

  def __init__(self, stage, children):
    """Constructor of the `Encoder`.

    If `stage` is an `EncodingStageInterface`, the constructor will wrap it as
    an `AdaptiveEncodingStageInterface` with no state.

    Args:
      stage: An `EncodingStageInterface` or an `AdaptiveEncodingStageInterface`.
      children: A dictionary mapping a subset of
        `stage.compressible_tensors_keys` to instances of `Encoder` objects,
        which are to be used to further encode the corresponding encoded tensors
        of `stage`.
    """
    stage = encoding_stage.as_adaptive_encoding_stage(stage)
    self.stage = stage
    self.children = children
    self._commuting_structure = None
    super(Encoder, self).__init__()

  @property
  def fully_commutes_with_sum(self):
    """`True/False` based on whether the `Encoder` commutes with sum.

    This property will return `True` iff the entire composition of encoding
    stages controlled by this object commutes with sum. That is, the stage
    wrapped by this object commutes with sum, and each of its children also
    `fully_commutes_with_sum`.

    Returns:
      A boolean, `True` iff the `Encoder` commutes with sum.
    """
    result = True
    for encoder in six.itervalues(self.children):
      result &= encoder.fully_commutes_with_sum
    return result & self.stage.commutes_with_sum

  @property
  def commuting_structure(self):
    """Represents the structure of the `Encoder` which commutes with sum.

    Returns:
      A dictionary with two keys: `EncoderKeys.COMMUTE` and
      `EncoderKeys.CHILDREN`. The `EncoderKeys.COMMUTE` key maps to `True` or
      `False`, based on whether the encoding stage controlled by this class
      comutes with sum. The `EncoderKeys.CHILDREN` key maps to a dictionary with
      the same keys as `self.children`, each of which maps to an object like
      this one, recursively.
    """
    if not self._commuting_structure:
      self._commuting_structure = self._commuting_structure_impl(True)
    return self._commuting_structure

  def _commuting_structure_impl(self, previous):
    """Implementation for the `commuting_structure` property."""
    current = previous & self.stage.commutes_with_sum
    commuting_structure = {
        EncoderKeys.COMMUTE: current,
        EncoderKeys.CHILDREN: {}
    }
    for key, encoder in six.iteritems(self.children):
      commuting_structure[EncoderKeys.CHILDREN][key] = (
          encoder._commuting_structure_impl(current))  # pylint: disable=protected-access
    return commuting_structure

  @property
  def state_update_aggregation_modes(self):
    """State aggregation modes of the Encoder.

    Returns:
      An object of the same structure as `state_update_tensors` returned by the
      `encode` method, with values replaced by their corresponding
      `StateAggregationMode` keys.
    """
    children_aggregation_modes = {}
    for key, encoder in six.iteritems(self.children):
      children_aggregation_modes[key] = encoder.state_update_aggregation_modes
    return {
        EncoderKeys.CHILDREN: children_aggregation_modes,
        EncoderKeys.TENSORS: self.stage.state_update_aggregation_modes
    }

  def initial_state(self, name=None):
    """Creates an initial state for the Encoder.

    Args:
      name: `string`, name of the operation.

    Returns:
      A dictionary with two keys: `EncoderKeys.STATE` and
      `EncoderKeys.CHILDREN`. The `EncoderKeys.STATE` key maps to a dictionary
      containing the initial state of the encoding stage controlled by this
      class. The `EncoderKeys.CHILDREN` key maps to a dictionary with the same
      keys as `self.children`, each of which maps to an object like this one,
      recursively.
    """
    with tf.compat.v1.name_scope(name, 'encoder_initial_state'):
      return self._initial_state_impl()

  def _initial_state_impl(self):
    """Implementation for the `initial_state` method."""
    children_state = {}
    for key, encoder in six.iteritems(self.children):
      with tf.compat.v1.name_scope(None, '/'.join([self.stage.name, key])):
        children_state[key] = encoder._initial_state_impl()  # pylint: disable=protected-access
    return {
        EncoderKeys.STATE: self.stage.initial_state(),
        EncoderKeys.CHILDREN: children_state
    }

  def update_state(self, state, state_update_tensors, name=None):
    """Updates the state of the Encoder.

    Args:
      state: A dictionary of the same structure as returned by the
        `initial_state` method, representing the current state.
      state_update_tensors: A dictionary of the same structure as returned by
        the `encode` method, representing the tensors needed for updating the
        state. The values are possibly aggregated across multiple realizations
        of the encoding.
      name: `string`, name of the operation.

    Returns:
      A dictionary with two keys: `EncoderKeys.STATE` and
      `EncoderKeys.CHILDREN`. The `EncoderKeys.STATE` key maps to a dictionary
      containing the initial state of the encoding stage controlled by this
      class. The `EncoderKeys.CHILDREN` key maps to a dictionary with the same
      keys as `self.children`, each of which maps to an object like this one,
      recursively.
    """
    values = tf.nest.flatten(state) + tf.nest.flatten(state_update_tensors)
    with tf.compat.v1.name_scope(name, 'encoder_update_state', values):
      return self._update_state_impl(state, state_update_tensors)

  def _update_state_impl(self, state, state_update_tensors):
    """Implementation for the `update_state` method."""
    children_states = {}
    for key, encoder in six.iteritems(self.children):
      with tf.compat.v1.name_scope(None, '/'.join([self.stage.name, key])):
        children_states[key] = encoder._update_state_impl(  # pylint: disable=protected-access
            state[EncoderKeys.CHILDREN][key],
            state_update_tensors[EncoderKeys.CHILDREN][key])
    return {
        EncoderKeys.CHILDREN:
            children_states,
        EncoderKeys.STATE:
            self.stage.update_state(state[EncoderKeys.STATE],
                                    state_update_tensors[EncoderKeys.TENSORS])
    }

  def get_params(self, state, name=None):
    """Gets parameters controlling the behavior of the Encoder.

    Args:
      state: A dictionary of the same structure as returned by the
        `initial_state` and `update_state` methods.
      name: `string`, name of the operation.

    Returns:
      A tuple `(encode_params, decode_params)`, where these are the parameters
      expected by the `encode` and `decode` methods, respectively. Both of them
      are dictionaries with two keys: `EncoderKeys.PARAMS` and
      `EncoderKeys.CHILDREN`. The `EncoderKeys.PARAMS` key maps to a dictionary
      containing the parameters of the encoding stage controlled by this class.
      The `EncoderKeys.CHILDREN` key maps to a dictionary with the same keys as
      `self.children`, each of which maps to an object like this one,
      recursively.
    """
    with tf.compat.v1.name_scope(name, 'encoder_get_params',
                                 tf.nest.flatten(state)):
      return self._get_params_impl(state)

  def _get_params_impl(self, state):
    """Implementation for the `get_params` method."""
    encode_params = {}
    decode_params = {}
    encode_params[EncoderKeys.PARAMS], decode_params[EncoderKeys.PARAMS] = (
        self.stage.get_params(state[EncoderKeys.STATE]))
    children_encode_params = {}
    children_decode_params = {}
    for key, encoder in six.iteritems(self.children):
      with tf.compat.v1.name_scope(None, '/'.join([self.stage.name, key])):
        children_encode_params[key], children_decode_params[key] = (
            encoder._get_params_impl(state[EncoderKeys.CHILDREN][key]))  # pylint: disable=protected-access
    encode_params[EncoderKeys.CHILDREN] = children_encode_params
    decode_params[EncoderKeys.CHILDREN] = children_decode_params
    return encode_params, decode_params

  def encode(self, x, encode_params, name=None):
    """Encodes a given `Tensor`.

    Args:
      x: A `Tensor`, input to be encoded.
      encode_params: A dictionary, containing the parameters needed for the
        encoding. The structure needs to be the return structure of the
        `get_params` method.
      name: `string`, name of the operation.

    Returns:
      A tuple `(encoded_tensors, state_update_tensors, input_shapes)`, where
      these are:
      `encoded_tensors`: A dictionary of `Tensor` objects representing the
        encoded input `x`.
      `state_update_tensors`: A dictionary of `Tensor` objects representing
        information necessary for updating the state. The dictionary has two
        keys: `EncoderKeys.TENSORS` and `EncoderKeys.CHILDREN`. The
        `EncoderKeys.TENSORS` key maps to the state_update_tensors produced by
        `self.stage`. The `EncoderKeys.CHILDREN` key maps to a dictionary with
        the same keys as `self.children`, each of which maps to an object like
        this one, recursively.
      'input_shapes': A dictionary of the shapes of inputs to individual
        encoding stages. The dictionary has two keys: `EncoderKeys.SHAPE` and
        `EncoderKeys.CHILDREN`. The `EncoderKeys.SHAPE` key maps to the shape of
        the input to `self.stage`. The `EncoderKeys.CHILDREN` key maps to a
        dictionary with the same keys as `self.children`, each of which maps to
        an object like this one, recursively. The values at the leaves of this
        dictionary can be either `Tensor` objects, non-TensorFlow constants such
        as a `list` or numpy value, or `None`, if the shape is not needed.
    """
    with tf.compat.v1.name_scope(name, 'encoder_encode',
                                 [x] + tf.nest.flatten(encode_params)):
      return self._encode_impl(x, encode_params)

  def _encode_impl(self, x, encode_params):
    """Implementation for the `encode` method."""
    encoded_tensors = {}
    state_update_tensors = {}
    input_shapes = {}
    if self.stage.decode_needs_input_shape:
      input_shapes[EncoderKeys.SHAPE] = py_utils.static_or_dynamic_shape(x)
    else:
      input_shapes[EncoderKeys.SHAPE] = None
    encoded_tensors, state_update_tensors[EncoderKeys.TENSORS] = (
        self.stage.encode(x, encode_params[EncoderKeys.PARAMS]))
    children_state_update_tensors = {}
    children_shapes = {}
    for key, encoder in six.iteritems(self.children):
      with tf.compat.v1.name_scope(None, '/'.join([self.stage.name, key])):
        (encoded_tensors[key], children_state_update_tensors[key],
         children_shapes[key]) = encoder._encode_impl(  # pylint: disable=protected-access
             encoded_tensors[key], encode_params[EncoderKeys.CHILDREN][key])
    state_update_tensors[EncoderKeys.CHILDREN] = children_state_update_tensors
    input_shapes[EncoderKeys.CHILDREN] = children_shapes
    return encoded_tensors, state_update_tensors, input_shapes

  def decode(self, encoded_tensors, decode_params, shape, name=None):
    """Decodes the encoded representation.

    Args:
      encoded_tensors: A dictionary representing the encoded value. The
        structure needs to be the corresponding return structure of the `encode`
        method.
      decode_params: A dictionary containing the parameters needed for the
        decoding. The structure needs to be the corresponding return structure
        of the `get_params` method.
      shape: A dictionary with the input shapes provided to the `encode` methods
        of individual encoding stages. The structure needs to be the
        corresponding return structure of the `encode` method.
      name: `string`, name of the operation.

    Returns:
      A single decoded `Tensor`.
    """
    values = (
        tf.nest.flatten(shape) + tf.nest.flatten(decode_params) +
        tf.nest.flatten(encoded_tensors))
    with tf.compat.v1.name_scope(name, 'encoder_decode', values):
      # Calling _decode_before_sum_impl with force_decode=True will decode the
      # entire tree, regardless of potential commutativity with sum.
      return self._decode_before_sum_impl(
          encoded_tensors, decode_params, shape, force_decode=True)

  def decode_before_sum(self, encoded_tensors, decode_params, shape, name=None):
    """Decodes the part of encoded representation not commuting with sum.

    This method represents part of the whole decoding logic, which does not
    commute with sum. It can be an identity (if everything commutes with sum),
    the full decoding (if nothing commutes with sum), or a partial decoding.

    Args:
      encoded_tensors: A dictionary representing the encoded value. The
        structure needs to be the corresponding return structure of the `encode`
        method.
      decode_params: A dictionary containing the parameters needed for the
        decoding. The structure needs to be the corresponding return structure
        of the `get_params` method.
      shape: A dictionary with the input shapes provided to the `encode` methods
        of individual encoding stages. The structure needs to be the
        corresponding return structure of the `encode` method.
      name: `string`, name of the operation.

    Returns:
      If `self.stage.commutes_with_sum` is `False`, returns a single decoded
      `Tensor`. If it is `True`, returns partially decoded `encoded_tensors`,
      i.e., dictionary with the same structure up to the part which does commute
      with sum. This structure is also the expected input to the
      `decode_after_sum` method.
    """
    values = (
        tf.nest.flatten(shape) + tf.nest.flatten(decode_params) +
        tf.nest.flatten(encoded_tensors))
    with tf.compat.v1.name_scope(name, 'encoder_decode_before_sum', values):
      return self._decode_before_sum_impl(
          encoded_tensors, decode_params, shape, force_decode=False)

  def _decode_before_sum_impl(self, encoded_tensors, decode_params, shape,
                              force_decode):
    """Implementation for the `decode_before_sum` method.

    The argument list of this method is different, to allow propagation of
    information about commutativity higher up the chain of encodings. For
    instance, consider an encoding consisting of three encoding stages, of which
    the first and third individually commute with sum. As a chain, the third
    stage does not commute with sum, because the second does not, and thus
    everything else after it.

    Args:
      encoded_tensors: See the `decode_before_sum` method.
      decode_params: See the `decode_before_sum` method.
      shape: See the `decode_before_sum` method.
      force_decode: If True, `self.stage.decode` will be called regardless of
        whether it commutes with sum or not.

    Returns:
      A structure as described in the `decode_before_sum` method.
    """
    temp_encoded_tensors = {}
    force_decode |= not self.stage.commutes_with_sum
    for key, value in six.iteritems(encoded_tensors):
      if key in self.children:
        with tf.compat.v1.name_scope(None, '/'.join([self.stage.name, key])):
          temp_encoded_tensors[key] = (
              self.children[key]._decode_before_sum_impl(  # pylint: disable=protected-access
                  value,
                  decode_params[EncoderKeys.CHILDREN][key],
                  shape[EncoderKeys.CHILDREN][key],
                  force_decode=force_decode))
      else:
        temp_encoded_tensors[key] = value
    if force_decode:
      # We explicitly provide num_summands=None as this is explicitly decoding
      # that happens *before* the sum.
      return self.stage.decode(temp_encoded_tensors,
                               decode_params[EncoderKeys.PARAMS],
                               num_summands=None,
                               shape=shape[EncoderKeys.SHAPE])
    else:
      return temp_encoded_tensors

  def decode_after_sum(self,
                       encoded_tensors,
                       decode_params,
                       num_summands,
                       shape,
                       name=None):
    """Decodes the part of encoded representation commuting with sum.

    This method is complementary to the `decode_before_sum` method in the sense
    that `decode_after_sum(decode_before_sum(encoded_tensors), ...)` is always
    equivalent to the full decoding of `encoded_tensors`.

    Args:
      encoded_tensors: A dictionary representing the encoded value. The
        structure needs to be the return structure of the `decode_before_sum`
        method.
      decode_params: A dictionary containing the parameters needed for the
        decoding. The structure needs to be the corresponding return structure
        of the `get_params` method.
      num_summands: The number of summands to be passed to individual encoding
        stages.
      shape: A dictionary with the input shapes provided to the `encode` methods
        of individual encoding stages. The structure needs to be the
        corresponding return structure of the `encode` method.
      name: `string`, name of the operation.

    Returns:
      A single decoded `Tensor`.
    """
    values = (
        tf.nest.flatten(shape) + tf.nest.flatten(decode_params) +
        tf.nest.flatten(encoded_tensors) + [num_summands])
    num_summands = tf.convert_to_tensor(num_summands)
    with tf.compat.v1.name_scope(name, 'encoder_decode_after_sum', values):
      return self._decode_after_sum_impl(encoded_tensors, decode_params,
                                         num_summands, shape)

  def _decode_after_sum_impl(self, encoded_tensors, decode_params, num_summands,
                             shape):
    """Implementation for the `decode_after_sum` method."""
    if not self.stage.commutes_with_sum:
      # This should have been decoded earlier in the decode_before_sum method.
      assert tf.is_tensor(encoded_tensors)
      return encoded_tensors

    temp_encoded_tensors = {}
    for key, value in six.iteritems(encoded_tensors):
      if key in self.children:
        with tf.compat.v1.name_scope(None, '/'.join([self.stage.name, key])):
          temp_encoded_tensors[key] = self.children[key]._decode_after_sum_impl(  # pylint: disable=protected-access
              value, decode_params[EncoderKeys.CHILDREN][key], num_summands,
              shape[EncoderKeys.CHILDREN][key])
      else:
        temp_encoded_tensors[key] = value
    return self.stage.decode(temp_encoded_tensors,
                             decode_params[EncoderKeys.PARAMS],
                             num_summands=num_summands,
                             shape=shape[EncoderKeys.SHAPE])


class EncoderComposer(object):
  """Class for composing and creating `Encoder`.

  This class is a utility for separating the creation of the `Encoder` from its
  intended functionality. Each instance of `EncoderComposer` corresponds to a
  node in a tree of encoding stages to be used for encoding.

  The `make` method converts this object to an `Encoder`, which exposes the
  encoding functionality.
  """

  def __init__(self, stage):
    """Constructor for the `EncoderComposer`.

    Args:
      stage: An `EncodingStageInterface` or an `AdaptiveEncodingStageInterface`.
    """
    if not isinstance(stage, (encoding_stage.EncodingStageInterface,
                              encoding_stage.AdaptiveEncodingStageInterface)):
      raise TypeError('The input argument is %s but must be an instance of '
                      'EncodingStageInterface or of '
                      'AdaptiveEncodingStageInterface' % type(stage))
    self._stage = stage
    self._children = {}
    super(EncoderComposer, self).__init__()

  def add_child(self, stage, key):
    """Adds a child node.

    Args:
      stage: An `EncodingStageInterface` or an `AdaptiveEncodingStageInterface`.
      key: `string`, specifying the output key of the encoding stage controlled
        by this object, to be further encoded by `stage`.

    Returns:
      An `EncoderComposer`, the newly created child node.

    Raises:
      KeyError: If `key` is not a compressible tensor of the encoding stage
        controlled by this object, or if it already has a child.
    """
    if key not in self._stage.compressible_tensors_keys:
      raise KeyError('The specified key is either not compressible or not '
                     'returned by the current encoding stage.')
    if key in self._children:
      raise KeyError('The specified key is already used.')
    new_builder = EncoderComposer(stage)
    self._children[key] = new_builder
    return new_builder

  def add_parent(self, stage, key):
    """Adds a parent node.

    Args:
      stage: An `EncodingStageInterface` or an `AdaptiveEncodingStageInterface`.
      key: `string`, specifying the output key of `stage` to be further encoded
        by the encoding stage controlled by this object.

    Returns:
      An `EncoderComposer`, the newly created parent node.

    Raises:
      KeyError: If `key` is not a compressible tensor of `stage`.
    """
    if key not in stage.compressible_tensors_keys:
      raise KeyError('The specified key is either not compressible or not '
                     'returned by the encoding stage.')
    new_builder = EncoderComposer(stage)
    new_builder._children[key] = self  # pylint: disable=protected-access
    return new_builder

  def make(self):
    """Recursively creates the `Encoder`.

    This method also recursively creates all children, but not parents.

    Returns:
      An `Encoder` composing the encoding stages.
    """
    children = {}
    for k, v in six.iteritems(self._children):
      children[k] = v.make()
    return Encoder(self._stage, children)


def split_params_by_commuting_structure(params, commuting_structure):
  """Splits params by commuting_structure.

  Args:
    params: A structure to be split. This should be the value returned by the
      `get_params` method of the `Encoder`.
    commuting_structure: A structure describing the part of `Encoder` that
      commutes with sum. This should be value returned by the
      `commuting_structure` property of `Encoder`.

  Returns:
    A tuple `(before_sum_params, after_sum_params)`, where both values have the
    same structure as `params`, with fields not relevant before/after summation
    replaced by `None`.
  """
  return _split_value_by_commuting_structure(params, EncoderKeys.PARAMS,
                                             commuting_structure)


def split_shapes_by_commuting_structure(shapes, commuting_structure):
  """Splits shapes by commuting_structure.

  Args:
    shapes: A structure to be split. This should be the value returned as
      `input_shapes` by the `encode` method of the `Encoder`.
    commuting_structure: A structure describing the part of `Encoder` that
      commutes with sum. This should be value returned by the
      `commuting_structure` property of `Encoder`.

  Returns:
    A tuple `(before_sum_shapes, after_sum_shapes)`, where both values have the
    same structure as `shapes`, with fields not relevant before/after summation
    replaced by `None`.
  """
  return _split_value_by_commuting_structure(shapes, EncoderKeys.SHAPE,
                                             commuting_structure)


def _split_value_by_commuting_structure(value, encoder_key,
                                        commuting_structure):
  """Utility for splitting value along commuting_structure of Encoder."""
  before_sum_value = {}
  after_sum_value = {}

  if isinstance(value[encoder_key], dict):
    empty_value = {k: None for k in value[encoder_key]}
  else:
    empty_value = None
  full_value = value[encoder_key]
  if commuting_structure[EncoderKeys.COMMUTE]:
    before_sum_value[encoder_key] = empty_value
    after_sum_value[encoder_key] = full_value
  else:
    before_sum_value[encoder_key] = full_value
    after_sum_value[encoder_key] = empty_value

  children_before_sum_value = {}
  children_after_sum_value = {}
  for key, val in six.iteritems(value[EncoderKeys.CHILDREN]):
    children_before_sum_value[key], children_after_sum_value[key] = (
        _split_value_by_commuting_structure(
            val, encoder_key, commuting_structure[EncoderKeys.CHILDREN][key]))
  before_sum_value[EncoderKeys.CHILDREN] = children_before_sum_value
  after_sum_value[EncoderKeys.CHILDREN] = children_after_sum_value

  return before_sum_value, after_sum_value
