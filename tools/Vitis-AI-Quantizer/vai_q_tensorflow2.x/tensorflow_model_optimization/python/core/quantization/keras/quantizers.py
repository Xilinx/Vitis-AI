# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Quantizer classes which implement quantization using TF Ops on a tensor.

Module: tfmot.quantization.keras
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quant_ops

keras = tf.keras


@six.add_metaclass(abc.ABCMeta)
class Quantizer(object):
  """ABC interface which encapsulates the logic of how to quantize tensors.

  This is an experimental API not subject to backward compatibility.

  A `Quantizer` is used by the library code to apply the mathematical
  transformations which actually quantize a tensor, hence allowing the user
  precise control over the algorithm with which tensors are quantized. When used
  in conjunction with `QuantizeConfig` it controls how a layer is quantized.

  Create a custom quantizer:

  ```python
  class FixedRangeQuantizer(Quantizer):
    # Example quantizer which clips tensors in a fixed range.

    def build(self, tensor_shape, name, layer):
      range_var = layer.add_weight(
        name + '_range',
        initializer=keras.initializers.Constant(6.0),
        trainable=False)

      return {
        'range_var': range_var,
      }

    def __call__(self, inputs, training, weights, **kwargs):
      return tf.keras.backend.clip(
          inputs, 0.0, weights['range_var'])

    def get_config(self):
      # Not needed. No __init__ parameters to serialize.
      return {}
  ```

  For a full example, see
  https://www.tensorflow.org/model_optimization/guide/quantization/training_comprehensive_guide.md
  """

  @abc.abstractmethod
  def build(self, tensor_shape, name, layer):
    """Construct the weights required by the quantizer.

    A quantizer may need to construct variables to hold the state for its
    algorithm. This function is invoked during the `build` stage of the layer
    that the quantizer is used for. Any variables constructed are under the
    scope of the `layer` and serialized as part of the layer.

    Args:
      tensor_shape: Shape of tensor which needs to be quantized.
      name: Name of tensor.
      layer: Keras layer which is quantizing the tensors. The layer is needed
        to construct the weights, and is also the owner of the weights.

    Returns: Dictionary of constructed weights. This dictionary will be
      passed to the quantizer's __call__ function as a `weights` dictionary.
    """

  @abc.abstractmethod
  def __call__(self, inputs, training, weights, **kwargs):
    """Apply quantization to the input tensor.

    This is the main function of the `Quantizer` which implements the core logic
    to quantize the tensor. It is invoked during the `call` stage of the layer,
    and allows modifying the tensors used in graph construction.

    Args:
      inputs: Input tensor to be quantized.
      training: Whether the graph is currently training.
      weights: Dictionary of weights the quantizer can use to quantize the
        tensor. This contains the weights created in the `build` function.
      **kwargs: Additional variables which may be passed to the quantizer.

    Returns: quantized tensor.
    """

  @abc.abstractmethod
  def get_config(self):
    """Returns the config used to serialize the `Quantizer`."""
    raise NotImplementedError('Quantizer should implement get_config().')

  @classmethod
  def from_config(cls, config):
    """Instantiates a `Quantizer` from its config.

    Args:
        config: Output of `get_config()`.

    Returns:
        A `Quantizer` instance.
    """
    return cls(**config)


class _QuantizeHelper(object):
  """Mixin with helper functions for quantizers."""

  def _add_range_weights(self, layer, name):
    """Add min and max vars to layer."""
    min_weight = layer.add_weight(
        name + '_min',
        initializer=keras.initializers.Constant(-6.0),
        trainable=False)
    max_weight = layer.add_weight(
        name + '_max',
        initializer=keras.initializers.Constant(6.0),
        trainable=False)

    return {'min_var': min_weight, 'max_var': max_weight}


class LastValueQuantizer(_QuantizeHelper, Quantizer):
  """Quantize tensor based on range the last batch of values."""

  # TODO(pulkitb): Decide and change num_bits to num_fixedpoint_values.

  def __init__(self, num_bits, per_axis, symmetric, narrow_range):
    """Construct a LastValueQuantizer.

    This is an experimental API not subject to backward compatibility.

    Args:
      num_bits: Number of bits for quantization
      per_axis: Whether to apply per_axis quantization. The last dimension is
        used as the axis.
      symmetric: If true, use symmetric quantization limits instead of training
        the minimum and maximum of each quantization range separately.
      narrow_range: In case of 8 bits, narrow_range nudges the quantized range
        to be [-127, 127] instead of [-128, 127]. This ensures symmetric
        range has 0 as the centre.
    """
    self.num_bits = num_bits
    self.per_axis = per_axis
    self.symmetric = symmetric
    self.narrow_range = narrow_range

  def build(self, tensor_shape, name, layer):
    return self._add_range_weights(layer, name)

  def __call__(self, inputs, training, weights, **kwargs):
    """Quantize tensor.

    Args:
      inputs: Input tensor to be quantized.
      training: Whether the graph is currently training.
      weights: Dictionary of weights the quantizer can use to quantize the
        tensor. This contains the weights created in the `build` function.
      **kwargs: Additional variables which may be passed to the quantizer.

    Returns:
      Quantized tensor.
    """
    return quant_ops.LastValueQuantize(
        inputs,
        weights['min_var'],
        weights['max_var'],
        is_training=training,
        num_bits=self.num_bits,
        per_channel=self.per_axis,
        symmetric=self.symmetric,
        narrow_range=self.narrow_range
    )

  def get_config(self):
    return {
        'num_bits': self.num_bits,
        'per_axis': self.per_axis,
        'symmetric': self.symmetric,
        'narrow_range': self.narrow_range
    }

  def __eq__(self, other):
    if not isinstance(other, LastValueQuantizer):
      return False

    return (self.num_bits == other.num_bits and
            self.per_axis == other.per_axis and
            self.symmetric == other.symmetric and
            self.narrow_range == other.narrow_range)

  def __ne__(self, other):
    return not self.__eq__(other)


class MovingAverageQuantizer(_QuantizeHelper, Quantizer):
  """Quantize tensor based on a moving average of values across batches."""

  def __init__(self, num_bits, per_axis, symmetric, narrow_range):
    """Construct a MovingAverageQuantizer.

    This is an experimental API not subject to backward compatibility.

    Args:
      num_bits: Number of bits for quantization
      per_axis: Whether to apply per_axis quantization. The last dimension is
        used as the axis.
      symmetric: If true, use symmetric quantization limits instead of training
        the minimum and maximum of each quantization range separately.
      narrow_range: In case of 8 bits, narrow_range nudges the quantized range
        to be [-127, 127] instead of [-128, 127]. This ensures symmetric
        range has 0 as the centre.
    """
    self.num_bits = num_bits
    self.per_axis = per_axis
    self.symmetric = symmetric
    self.narrow_range = narrow_range

  def build(self, tensor_shape, name, layer):
    return self._add_range_weights(layer, name)

  def __call__(self, inputs, training, weights, **kwargs):
    """Quantize tensor.

    Args:
      inputs: Input tensor to be quantized.
      training: Whether the graph is currently training.
      weights: Dictionary of weights the quantizer can use to quantize the
        tensor. This contains the weights created in the `build` function.
      **kwargs: Additional variables which may be passed to the quantizer.

    Returns:
      Quantized tensor.
    """
    return quant_ops.MovingAvgQuantize(
        inputs,
        weights['min_var'],
        weights['max_var'],
        ema_decay=0.999,
        is_training=training,
        num_bits=self.num_bits,
        per_channel=self.per_axis,
        symmetric=self.symmetric,
        narrow_range=self.narrow_range,
    )

  def get_config(self):
    return {
        'num_bits': self.num_bits,
        'per_axis': self.per_axis,
        'symmetric': self.symmetric,
        'narrow_range': self.narrow_range
    }

  def __eq__(self, other):
    if not isinstance(other, MovingAverageQuantizer):
      return False

    return (self.num_bits == other.num_bits and
            self.per_axis == other.per_axis and
            self.symmetric == other.symmetric and
            self.narrow_range == other.narrow_range)

  def __ne__(self, other):
    return not self.__eq__(other)


class AllValuesQuantizer(_QuantizeHelper, Quantizer):
  """Quantize tensor based on min/max of tensor values across all batches."""

  def __init__(self, num_bits, per_axis, symmetric, narrow_range):
    """Construct an AllValuesQuantizer.

    This is an experimental API not subject to backward compatibility.

    Args:
      num_bits: Number of bits for quantization
      per_axis: Whether to apply per_axis quantization. The last dimension is
        used as the axis.
      symmetric: If true, use symmetric quantization limits instead of training
        the minimum and maximum of each quantization range separately.
      narrow_range: In case of 8 bits, narrow_range nudges the quantized range
        to be [-127, 127] instead of [-128, 127]. This ensures symmetric
        range has 0 as the centre.
    """
    self.num_bits = num_bits
    self.per_axis = per_axis
    self.symmetric = symmetric
    self.narrow_range = narrow_range

  def build(self, tensor_shape, name, layer):
    min_weight = layer.add_weight(
        name + '_min',
        initializer=keras.initializers.Constant(0.0),
        trainable=False)
    max_weight = layer.add_weight(
        name + '_max',
        initializer=keras.initializers.Constant(0.0),
        trainable=False)
    return {'min_var': min_weight, 'max_var': max_weight}

  def __call__(self, inputs, training, weights, **kwargs):
    """Quantize tensor.

    Args:
      inputs: Input tensor to be quantized.
      training: Whether the graph is currently training.
      weights: Dictionary of weights the quantizer can use to quantize the
        tensor. This contains the weights created in the `build` function.
      **kwargs: Additional variables which may be passed to the quantizer.

    Returns:
      Quantized tensor.
    """
    return quant_ops.AllValuesQuantize(
        inputs,
        weights['min_var'],
        weights['max_var'],
        is_training=training,
        num_bits=self.num_bits,
        symmetric=self.symmetric,
        narrow_range=self.narrow_range,
    )

  def get_config(self):
    return {
        'num_bits': self.num_bits,
        'per_axis': self.per_axis,
        'symmetric': self.symmetric,
        'narrow_range': self.narrow_range
    }

  def __eq__(self, other):
    if not isinstance(other, AllValuesQuantizer):
      return False

    return (self.num_bits == other.num_bits and
            self.per_axis == other.per_axis and
            self.symmetric == other.symmetric and
            self.narrow_range == other.narrow_range)

  def __ne__(self, other):
    return not self.__eq__(other)


def _types_dict():
  return {
      'AllValuesQuantizer': AllValuesQuantizer,
      'LastValueQuantizer': LastValueQuantizer,
      'MovingAverageQuantizer': MovingAverageQuantizer
  }
