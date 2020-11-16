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
"""EncodingStageInterface, its adaptive extension, and their implementations.

The interfaces are designed to support encoding and decoding that may happen
in different locations, including possibly different TensorFlow `Session`
objects, without the implementer needing to understand how any communication is
realized. Example scenarios include
* Both encoding and decoding can happen in the same location, such as for
  experimental evaluation of efficiency, and no communication is necessary.
* Both encoding and decoding can happen in different locations, but run in the
  same `Session`, such as distributed datacenter training. The communication
  between locations is handled by TensorFlow.
* Encoding and decoding can happen on multiple locations, and communication
  between them needs to happen outside of `TensorFlow`, such as encoding the
  state of a model which is sent to a mobile device to be later decoded and used
  for inference.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import enum
import six
import tensorflow as tf

INITIAL_STATE_SCOPE_SUFFIX = '_initial_state'
UPDATE_STATE_SCOPE_SUFFIX = '_update_state'
GET_PARAMS_SCOPE_SUFFIX = '_get_params'
ENCODE_SCOPE_SUFFIX = '_encode'
DECODE_SCOPE_SUFFIX = '_decode'


class StateAggregationMode(enum.Enum):
  """Enum of available modes of aggregation for state.

  This enum serves as a declaration of how the `state_update_tensors` returned
  by the `encode` method of `StatefulEncodingStageInterface` should be
  aggregated, before being passed to the `update_state` method.

  This is primarily relevant for the setting where the encoding happens in
  multiple locations, and a function of the encoded objects needs to be computed
  at a central node. The implementation of these modes can differ depending on
  the context. For instance, aggregation of these values in a star topology will
  look differently from a multi-tier aggregation, which needs to know how some
  intermediary representations is to be merged.

  List of available values:
  * `SUM`: Summation.
  * `MIN`: Minimum.
  * `MAX`: Maximum.
  * `STACK`: Stacking along a new dimentsion. This can necessary for computing
    arbitrary function of a collection of those values, such as a percentile.
  """
  SUM = 1
  MIN = 2
  MAX = 3
  STACK = 4


@six.add_metaclass(abc.ABCMeta)
class EncodingStageInterface(object):
  """Interface for the core of encoding logic.

  This core interface should support encoding being executed in a variety of
  contexts. For instance,
  * Both encoding and decoding can happen in the same location, such as for
    experimental evaluation of efficiency.
  * Both encoding and decoding can happen in different locations, but run in the
    same `Session`, such as distributed datacenter training.
  * Encoding and decoding can happen in multiple locations, and communication
    between them needs to happen outside of `TensorFlow`, such as compressing
    a state of a model which is sent to a mobile device to be later used for
    inference.

  This interface is designed such that its implementer need not worry about the
  potential communication patterns, and the implementation will support all.

  Each implementation of this interface is supposed to be a relatively
  elementary transformation. In particular, it does not need to realize any
  representation savings by itself. Instead, a particular compositions of these
  elementary transformations will realize the desired savings. These
  compositions are realized by the `Encoder` class.

  Each implementation should also be wrapped by `tf_style_encoding_stage` to
  ensure adherence to the TensorFlow style guide. The adherence is enforced by
  the `BaseEncodingStageTest` test class. See `test_utils.py` for more details.

  For an adaptive version with a broader interface, see
  `AdaptiveEncodingStageInterface`.
  """

  @abc.abstractproperty
  def name(self):
    """Name of the encoding stage.

    This is a general name for the implementation of this interface, which is
    used mainly by the `Encoder` class to create appropriate TensorFlow name
    scopes when composing individual encoding stages.
    """

  @abc.abstractproperty
  def compressible_tensors_keys(self):
    """Keys of encoded tensors allowed to be further encoded.

    These keys correspond to tensors in object returned by the `encode` method,
    that are allowed to be further lossily compressed.

    This property does not directly impact the functionality, but is used by the
    `Encoder` class to validate composition.

    Returns:
      A list of `string` values.
    """

  @abc.abstractproperty
  def commutes_with_sum(self):
    """`True/False` based on whether the encoding commutes with sum.

    Iff `True`, it means that given multiple inputs `x` with the same `shape`
    and `dtype`, and the same `params` argument of the `encode` method, the
    implementation is such that every value in the returned `encoded_tensors`
    can be first summed, before being passed to the decoding functionality, and
    the output should be identical (up to numerical precision) to summing the
    fully decoded `Tensor` objects.

    Note that this also assumes that each of the `decode` methods would be used
    with the same values of `decode_params`.

    Returns:
      A boolean, `True` iff the encoding commutes with sum.
    """

  @abc.abstractproperty
  def decode_needs_input_shape(self):
    """Whether original shape of the encoded object is needed for decoding.

    Iff `True`, it means that the `shape` of the `x` argument to the `encode`
    method needs to be provided to the `decode` method. For instance, this is
    needed for bitpacking, where inputs of multiple shapes can result in
    identical bitpacked representations. Note however, the shape information
    should not be stored in the return structure of the `encode` method.

    This property will be used by `Encoder` to efficiently realize the
    composition of implementations of this interface, and to make the necessary
    shape information available.
    """

  @abc.abstractmethod
  def get_params(self):
    """Returns the parameters needed for encoding.

    This method returns parameters controlling the behavior of the `encode` and
    `decode` methods.

    Implementation of this method should clearly document what are the keys of
    parameters returned by this method, in order for a potential stateful
    subclass being able to adaptively modify only existing keys.

    Note that this method is not purely functional in terms of `TensorFlow`. The
    params can be derived from an internal state of the compressor. For
    instance, if a constructor optionally takes a `Variable` as an input
    argument, which is allowed to change during iterative execution, that
    `Variable`, or a function of it, would be exposed via this method. However,
    only values that can be TensorFlow values should be exposed via params. If a
    parameter always needs to be a Python constant, for instance used for Python
    control flow, it should not be exposed via params, and accessed via `self`
    instead.

    Returns:
      A tuple `(encode_params, decode_params)`, where
      `encode_params`: A dictionary to be passed as argument to the `encode`
        method.
      `decode_params`: A dictionary to be passed as argument to the `decode`
        method.
      Each value of the dictionaries can be either a `Tensor` or any python
      constant.
    """

  @abc.abstractmethod
  def encode(self, x, encode_params):
    """Encodes a given `Tensor`.

    This method can create TensorFlow variables, which can be updated every time
    the encoding is executed. An example is an encoder that internally remembers
    the error incurred by previous encoding, and adds it to `x` in the next
    iteration, before executing the encoding.

    However, this method may be called in an entirely separate graph from all
    other methods. That is, the implementer of this class can *only* assume such
    variables can be accessed from this method but not from others.

    Args:
      x: A `Tensor`, input to be encoded.
      encode_params: A dictionary, containing the parameters needed for the
        encoding. The structure needs to be the return structure of the
        `get_params` method.

    Returns:
      A dictionary of `Tensor` objects representing the encoded input `x`.
    """

  @abc.abstractmethod
  def decode(self,
             encoded_tensors,
             decode_params,
             num_summands=None,
             shape=None):
    """Decodes the encoded representation.

    This method is the inverse transformation of the `encode` method. The
    `encoded_tensors` argument is expected to be the output structure of
    `encode` method.

    The `num_summands` argument is needed because the `decode` of some encoding
    stages only commute with sum if the number of summands is known.

    Consider the example of uniform quantization on a specified interval.
    Encoding applies a pre-defined linear transformation to the input, and maps
    the resulting values to a discrete set of values. Because of the linear
    transformation, the decoding functionality does not immediately commute with
    sum. However, if we knew how many summands are in the sum, we can determine
    what is the appropriate inverse linear transformation, enabling the
    commutativity.

    A simple way to make this functionality available is to add a
    `tf.constant(1, tf.int32)` to the encoded tensors returned by the `encode`
    method.

    The problem is that this approach will be often inefficient. Typically, we
    are intereseted in encoding a collection of values, such as all weights of a
    model, and multiple encoding stages might require this information. The
    result will be a lot of redundant information being communicated. Moreover,
    a user interested in this will always have the relevant information already
    available.

    Such information can thus be provided to the `decode` method of an encoding
    stage via the `num_summands` argument, which will be handled by higher-level
    interfaces.

    Args:
      encoded_tensors: A dictionary containing `Tensor` objects, representing
        the encoded value.
      decode_params: A dictionary, containing the parameters needed for the
        decoding. The structure needs to be the return structure of the
        `get_params` method.
      num_summands: An integer representing the number of summands, if
        `encoded_tensors` is a sum of the encoded representations. The default
        value `None` refers to the case when no summation occurred, and can thus
        be interpreted as `1`.
      shape: Required if the `decode_needs_input_shape` property is `True`. A
        shape of the original input to `encode`, if needed for decoding. Can be
        either a `Tensor`, or a python object.

    Returns:
      A single decoded `Tensor`.
    """


@six.add_metaclass(abc.ABCMeta)
class AdaptiveEncodingStageInterface(object):
  """Adaptive version of the `EncodingStageInterface`.

  This class has the same functionality as the `EncodingStageInterface`, but in
  addition maintains a state, which is adaptive based on the values being
  compressed and can parameterize the way encoding functionality works. Note
  that this is useful only in case where the encoding is executed in multiple
  iterations.

  A typical implementation of this interface would be a wrapper of an
  implementation of `EncodingStageInterface, which uses the existing stateless
  transformations and adds state that controls some of the parameters returned
  by the `get_params` method.

  The important distinction is that in addition to `encoded_tensors`, the
  `encode` method of this class returns an additional dictionary of
  `state_update_tensors`. The `commutes_with_sum` property talks about summation
  of only the `encoded_tensors`. The `state_update_tensors` can be aggregated
  in more flexible ways, specified by the `state_update_aggregation_modes`
  property, before being passed to the `update_state` method.

  Each implementation should also be wrapped by `tf_style_encoding_stage` to
  ensure adherence to the TensorFlow style guide. The adherence is enforced by
  the `BaseEncodingStageTest` test class. See `test_utils.py` for more details.
  """

  @abc.abstractproperty
  def name(self):
    """Name of the encoding stage.

    This is a general name for the implementation of this interface, which is
    used mainly by the `Encoder` class to create appropriate TensorFlow name
    scopes when composing individual encoding stages.
    """

  @abc.abstractproperty
  def compressible_tensors_keys(self):
    """Keys of encoded tensors allowed to be further encoded.

    These keys correspond to tensors in object returned by the `encode` method,
    that are allowed to be further lossily compressed.

    This property does not directly impact the functionality, but is used by the
    `Encoder` class to validate composition.

    Returns:
      A list of `string` values.
    """

  @abc.abstractproperty
  def commutes_with_sum(self):
    """`True/False` based on whether the encoding commutes with sum.

    Iff `True`, it means that given multiple inputs `x` with the same `shape`
    and `dtype`, and the same `params` argument of the `encode` method, the
    implementation is such that every value in the returned `encoded_tensors`
    can be first summed, before being passed to the decoding functionality, and
    the output should be identical (up to numerical precision) to summing the
    fully decoded `Tensor` objects.

    Note that this also assumes that each of the `decode` methods would be used
    with the same values of `decode_params`.

    Returns:
      A boolean, `True` iff the encoding commutes with sum.
    """

  @abc.abstractproperty
  def decode_needs_input_shape(self):
    """Whether original shape of the encoded object is needed for decoding.

    Iff `True`, it means that the `shape` of the `x` argument to the `encode`
    method needs to be provided to the `decode` method. For instance, this is
    needed for bitpacking, where inputs of multiple shapes can result in
    identical bitpacked representations.

    This property will be used by `Encoder` to efficiently realize the
    composition of implementations of this interface.
    """

  @abc.abstractproperty
  def state_update_aggregation_modes(self):
    """Aggregation mode of state update tensors.

    Returns a dictionary mapping keys appearing in `state_update_tensors`
    returned by the `encode` method to a `StateAggregationMode` object, which
    declares how should the `Tensor` objects be aggreggated.
    """

  @abc.abstractmethod
  def initial_state(self):
    """Creates an initial state.

    Returns:
      A dictionary of `Tensor` objects, representing the initial state.
    """

  @abc.abstractmethod
  def update_state(self, state, state_update_tensors):
    """Updates the state.

    This method updates the `state` based on the current value of `state`, and
    (potentially aggregated) `state_update_tesors`, returned by the `encode`
    method. This will typically happen at the end of a notion of iteration.

    Args:
      state: A dictionary of `Tensor` objects, representing the current state.
        The dictionary has the same structure as return dictionary of the
        `initial_state` method.
      state_update_tensors: A dictionary of `Tensor` objects, representing the
        `state_update_tensors` returned by the `encode` method and appropriately
        aggregated.

    Returns:
      A dictionary of `Tensor` objects, representing the updated `state`.
    """

  @abc.abstractmethod
  def get_params(self, state):
    """Returns the parameters needed for encoding.

    This method returns parameters controlling the behavior of the `encode` and
    `decode` methods.

    Note that this method is not purely functional in terms of `TensorFlow`. The
    params can be derived from an internal state of the compressor. For
    instance, if a constructor optionally takes a `Variable` as an input
    argument, which is allowed to change during iterative execution, that
    `Variable`, or a function of it, would be exposed via this method.However,
    only values that can be TensorFlow values should be exposed via params. If a
    parameter always needs to be a Python constant, for instance used for Python
    control flow, it should not be exposed via params, and accessed via `self`
    instead.

    Args:
      state: A dictionary of `Tensor` objects. This should be the object
        controlled by the `initial_state` and `update_state` methods.

    Returns:
      A tuple `(encode_params, decode_params)`, where
      `encode_params`: A dictionary to be passed as argument to the `encode`
        method.
      `decode_params`: A dictionary to be passed as argument to the `decode`
        method.
      Each value of the dictionaries can be either a `Tensor` or any python
      constant.
    """

  @abc.abstractmethod
  def encode(self, x, encode_params):
    """Encodes a given `Tensor`.

    This method can create TensorFlow variables, which can be updated every time
    the encoding is executed. An example is an encoder that internally remembers
    the error incurred by previous encoding, and adds it to `x` in the next
    iteration, before executing the encoding.

    However, this method may be called in an entirely separate graph from all
    other methods. That is, the implementer of this class can *only* assume such
    variables can be accessed from this method but not from others.

    Args:
      x: A `Tensor`, input to be encoded.
      encode_params: A dictionary, containing the parameters needed for the
        encoding. The structure needs to be the return structure of `get_params`
        method.

    Returns:
      A tuple `(encoded_tensors, state_update_tensors)`, where these are:
      `encoded_tensors`: A dictionary of `Tensor` objects representing the
        encoded input `x`.
      `state_update_tensors`: A dictionary of `Tensor` objects representing
        information necessary for updating the state.
    """

  @abc.abstractmethod
  def decode(self,
             encoded_tensors,
             decode_params,
             num_summands=None,
             shape=None):
    """Decodes the encoded representation.

    This method is the inverse transformation of the `encode` method. The
    `encoded_tensors` argument is expected to be the output structure of
    `encode` method.

    The `num_summands` argument is needed because the `decode` some encoding
    stages only commute with sum if the number of summands is known. Consider
    the example of uniform quantization on a specified interval.

    Encoding applies a pre-defined linear transformation to the input, and
    maps the resulting values to a discrete set of values. Because of the linear
    transformation, the decoding functionality does not immediately commute with
    sum. However, if we knew how many summands are in the sum, we can determine
    what is the appropriate inverse linear transformation, enabling the
    commutativity.

    A simple way to make this functionality available is to add a
    `tf.constant(1, tf.int32)` to the encoded tensors returned by the `encode`
    method.

    The problem is that this approach will be often inefficient. Typically, we
    are intereseted in encoding a collection of values, such as all weights of a
    model, and multiple encoding stages might require this information. The
    result will be a lot of redundant information being communicated. Moreover,
    a user interested in this will always have the relevant information already
    available.

    Such information can thus be provided to the `decode` method of an encoding
    stage via the `num_summands` argument, which will be handled by higher-level
    interfaces.

    Args:
      encoded_tensors: A dictionary containing `Tensor` objects, representing
        the encoded value.
      decode_params: A dictionary, containing the parameters needed for the
        decoding. The structure needs to be the return structure of `get_params`
        method.
      num_summands: An integer representing number of summands, if
        `encoded_tensors` is a sum of the encoded representations. The default
        value `None` is to be interpreted as `1`.
      shape: Required if the `decode_needs_input_shape` property is `True`. A
        shape of the original input to `encode`, if needed for decoding. Can be
        either a `Tensor`, or a python object.

    Returns:
      A single decoded `Tensor`.
    """


def tf_style_encoding_stage(cls):
  """Decorator for implementations of `EncodingStageInterface`.

  This decorator ensures adherence to the TensorFlow style guide, and should be
  used to decorate every implementation of `EncodingStageInterface`. In
  particular, it captures the methods of the interface in appropriate name
  scopes or variable scopes, and calls `tf.convert_to_tensor` on the provided
  inputs.

  For `get_params`, `encode` and `decode` methods, it adds an optional `name`
  argument with default value `None`, as per the style guide.

  Args:
    cls: The class to be decorated. Must be an `EncodingStageInterface`.

  Returns:
    Decorated class.

  Raises:
    TypeError: If `cls` is not an `EncodingStageInterface`.
  """

  if not issubclass(cls, EncodingStageInterface):
    raise TypeError('Unable to decorate %s. Provided class must be a subclass '
                    'of EncodingStageInterface.' % cls)

  class TFStyleEncodingStage(cls):
    """The decorated encoding stage."""

    def __init__(self, *args, **kwargs):
      self._wrapped_class = cls(*args, **kwargs)

    def __getattr__(self, attr):
      return self._wrapped_class.__getattribute__(attr)

    @_tf_style_get_params
    def get_params(self, name=None):
      return super(TFStyleEncodingStage, self).get_params()

    @_tf_style_encode
    def encode(self, x, encode_params, name=None):
      return super(TFStyleEncodingStage, self).encode(x, encode_params)

    @_tf_style_decode
    def decode(self,
               encoded_tensors,
               decode_params,
               num_summands=None,
               shape=None,
               name=None):
      return super(TFStyleEncodingStage,
                   self).decode(encoded_tensors, decode_params, num_summands,
                                shape)

  return TFStyleEncodingStage


def tf_style_adaptive_encoding_stage(cls):
  """Decorator for implementations of `AdaptiveEncodingStageInterface`.

  This decorator ensures adherence to the TensorFlow style guide, and should be
  used to decorate every implementation of `AdaptiveEncodingStageInterface`. In
  particular, it captures the methods of the interface in appropriate name
  scopes or variable scopes, and calls `tf.convert_to_tensor` on the provided
  inputs.

  For `initial_state`, `update_state`, `get_params`, `encode` and `decode`
  methods, it adds an optional `name` argument with default value `None`, as per
  the style guide.

  Args:
    cls: The class to be decorated. Must be an `AdaptiveEncodingStageInterface`.

  Returns:
    Decorated class.

  Raises:
    TypeError: If `cls` is not an `AdaptiveEncodingStageInterface`.
  """

  if not issubclass(cls, AdaptiveEncodingStageInterface):
    raise TypeError('Unable to decorate %s. Provided class must be a subclass '
                    'of AdaptiveEncodingStageInterface.' % cls)

  class TFStyleAdaptiveEncodingStage(cls):
    """The decorated adaptive encoding stage."""

    def __init__(self, *args, **kwargs):
      self._wrapped_class = cls(*args, **kwargs)

    def __getattr__(self, attr):
      return self._wrapped_class.__getattribute__(attr)

    @_tf_style_initial_state
    def initial_state(self, name=None):
      return super(TFStyleAdaptiveEncodingStage, self).initial_state()

    @_tf_style_update_state
    def update_state(self, state, state_update_tensors, name=None):
      return super(TFStyleAdaptiveEncodingStage, self).update_state(
          state, state_update_tensors)

    @_tf_style_adaptive_get_params
    def get_params(self, state, name=None):
      return super(TFStyleAdaptiveEncodingStage, self).get_params(state)

    @_tf_style_encode
    def encode(self, x, encode_params, name=None):
      return super(TFStyleAdaptiveEncodingStage, self).encode(x, encode_params)

    @_tf_style_decode
    def decode(self,
               encoded_tensors,
               decode_params,
               num_summands=None,
               shape=None,
               name=None):
      return super(TFStyleAdaptiveEncodingStage,
                   self).decode(encoded_tensors, decode_params, num_summands,
                                shape)

  return TFStyleAdaptiveEncodingStage


def _tf_style_initial_state(initial_state_fn):
  """Method decorator for `tf_style_adaptive_encoding_stage`."""

  def actual_initial_state_fn(self, name=None):
    """Modified `initial_state` method."""
    with tf.compat.v1.name_scope(name, self.name + INITIAL_STATE_SCOPE_SUFFIX):
      return initial_state_fn(self, name=name)

  return actual_initial_state_fn


def _tf_style_update_state(update_state_fn):
  """Method decorator for `tf_style_adaptive_encoding_stage`."""

  def actual_initial_state_fn(self, state, state_update_tensors, name=None):
    """Modified `update_state` method."""
    values = list(state.values()) + list(state_update_tensors.values())
    with tf.compat.v1.name_scope(name, self.name + UPDATE_STATE_SCOPE_SUFFIX,
                                 values):
      state = tf.nest.map_structure(tf.convert_to_tensor, state)
      state_update_tensors = tf.nest.map_structure(tf.convert_to_tensor,
                                                   state_update_tensors)
      return update_state_fn(self, state, state_update_tensors, name=name)

  return actual_initial_state_fn


def _tf_style_get_params(get_params_fn):
  """Method decorator for `tf_style_encoding_stage`."""

  def actual_get_params_fn(self, name=None):
    """Modified `get_params` method."""
    with tf.compat.v1.name_scope(name, self.name + GET_PARAMS_SCOPE_SUFFIX):
      return get_params_fn(self, name=name)

  return actual_get_params_fn


def _tf_style_adaptive_get_params(get_params_fn):
  """Method decorator for `tf_style_adaptive_encoding_stage`."""

  def actual_get_params_fn(self, state, name=None):
    """Modified `get_params` method."""
    with tf.compat.v1.name_scope(name, self.name + GET_PARAMS_SCOPE_SUFFIX,
                                 state.values()):
      state = tf.nest.map_structure(tf.convert_to_tensor, state)
      return get_params_fn(self, state, name=name)

  return actual_get_params_fn


def _tf_style_encode(encode_fn):
  """Method decorator for `tf_style(_adaptive)_encoding_stage`."""

  def actual_encode_fn(self, x, encode_params, name=None):
    """Modified `encode` method."""
    values = list(encode_params.values()) + [x]
    with tf.compat.v1.variable_scope(name, self.name + ENCODE_SCOPE_SUFFIX,
                                     values):
      x = tf.convert_to_tensor(x)
      encode_params = tf.nest.map_structure(tf.convert_to_tensor, encode_params)
      return encode_fn(self, x, encode_params, name=name)

  return actual_encode_fn


def _tf_style_decode(decode_fn):
  """Method decorator for `tf_style(_adaptive)_encoding_stage`."""

  def actual_decode_fn(self,
                       encoded_tensors,
                       decode_params,
                       num_summands=None,
                       shape=None,
                       name=None):
    """Modified `decode` method."""
    values = list(encoded_tensors.values()) + list(decode_params.values())
    with tf.compat.v1.variable_scope(name, self.name + DECODE_SCOPE_SUFFIX,
                                     values):
      encoded_tensors = tf.nest.map_structure(tf.convert_to_tensor,
                                              encoded_tensors)
      decode_params = tf.nest.map_structure(tf.convert_to_tensor, decode_params)
      if shape is not None:
        shape = tf.convert_to_tensor(shape)
      if num_summands is not None:
        num_summands = tf.convert_to_tensor(num_summands)
      return decode_fn(
          self,
          encoded_tensors,
          decode_params,
          num_summands=num_summands,
          shape=shape,
          name=name)

  return actual_decode_fn


def as_adaptive_encoding_stage(stage):
  """Returns an instance of `AdaptiveEncodingStageInterface`.

  If an `EncodingStageInterface` object is provided, the returned instance of
  `AdaptiveEncodingStageInterface` is passing around an empty state, not
  modifying the functionality of the `stage` in any way.

  Args:
    stage: An `EncodingStageInterface` or `AdaptiveEncodingStageInterface`
      object.

  Returns:
    An instance of `AdaptiveEncodingStageInterface` with the same functionality
    as `stage`.

  Raises:
    TypeError: If `stage` is not `EncodingStageInterface` or
    `AdaptiveEncodingStageInterface`.
  """

  if isinstance(stage, EncodingStageInterface):
    return NoneStateAdaptiveEncodingStage(stage)
  elif isinstance(stage, AdaptiveEncodingStageInterface):
    return stage
  else:
    raise TypeError


class NoneStateAdaptiveEncodingStage(AdaptiveEncodingStageInterface):
  """Wraps an `EncodingStageInterface` as `AdaptiveEncodingStageInterface`."""

  def __init__(self, wrapped_stage):
    if not isinstance(wrapped_stage, EncodingStageInterface):
      raise TypeError(
          'The provided stage must be an instance of EncodingStageInterface.')
    self._wrapped_stage = wrapped_stage

  def __getattr__(self, attr):
    return self._wrapped_stage.__getattr__(attr)

  @property
  def name(self):
    return self._wrapped_stage.name

  @property
  def compressible_tensors_keys(self):
    return self._wrapped_stage.compressible_tensors_keys

  @property
  def commutes_with_sum(self):
    return self._wrapped_stage.commutes_with_sum

  @property
  def decode_needs_input_shape(self):
    return self._wrapped_stage.decode_needs_input_shape

  @property
  def state_update_aggregation_modes(self):
    return {}

  def initial_state(self, name=None):
    del name  # Unused.
    return {}

  def update_state(self, state, state_update_tensors, name=None):
    del state  # Unused.
    del state_update_tensors  # Unused.
    del name  # Unused.
    return {}

  def get_params(self, state, name=None):
    del state  # Unused.
    return self._wrapped_stage.get_params(name)

  def encode(self, x, encode_params, name=None):
    return self._wrapped_stage.encode(x, encode_params, name), {}

  def decode(self,
             encoded_tensors,
             decode_params,
             num_summands=None,
             shape=None,
             name=None):
    return self._wrapped_stage.decode(encoded_tensors, decode_params,
                                      num_summands, shape, name)
