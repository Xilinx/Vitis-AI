# Copyright 2019 Xilinx Inc.
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
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

import tensorflow as tf

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
