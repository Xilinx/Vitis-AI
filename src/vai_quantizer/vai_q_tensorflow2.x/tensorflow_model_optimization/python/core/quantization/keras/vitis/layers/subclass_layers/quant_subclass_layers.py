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
"""Quantizable subclass layers for quantization of subclassed models."""

import tensorflow as tf
from tensorflow import keras

from keras.layers.core import TFOpLambda
from tensorflow.python.util.tf_export import get_symbol_from_name


class QuantIdentity(tf.keras.layers.Layer):
  """Simple Identity layer for quantization or replacement.
  The available settings of 'application' are ['input', 'output', 'none'], and
  'input' for quantizing input tensors, 'output' for quantizing intermediate
  tensors, 'none' for replacing specific layers, such as dropout layer.
  """

  def __init__(self, application='input', **kwargs):
    super(QuantIdentity, self).__init__(**kwargs)

    self.application = application

  def call(self, inputs, **kwargs):
    return inputs


class QuantTFOpLambda(tf.keras.layers.Layer):
  """Wraps TF API symbols in a `Layer` object and
  followed by a identity layer for quantization.
  """

  def __init__(self, function, **kwargs):
    super(QuantTFOpLambda, self).__init__(**kwargs)

    self.lambda_layer = TFOpLambda(function)
    self.quantize_layer = QuantIdentity(
        application='output', name='output_' + self.lambda_layer.name)

  def call(self, *args, **kwargs):
    '''Call the tf.op and tf.identity'''
    x = self.lambda_layer(*args, **kwargs)

    return self.quantize_layer(x)

  def get_config(self):
    config = {"function": self.lambda_layer.symbol}

    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    config = config.copy()

    symbol_name = config["function"]
    function = get_symbol_from_name(symbol_name)
    if not function:
      raise ValueError(f"TF symbol `{symbol_name}` could not be found.")

    config["function"] = function
    return cls(**config)


class CustomizedTFOpLambda(TFOpLambda):
  """Wraps TF API symbols in a `Layer` object.
  For multiple inputs we provide a list rather than independent input,
  this makes compute_output_shape method working.
  """

  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def __init__(self, function, **kwargs):
    original_kwargs = kwargs

    base_kwargs = {}

    for key, value in kwargs.items():
      if key == 'trainable' or key == 'name' or key == 'dtype' or key == 'dynamic':
        base_kwargs[key] = value

    super().__init__(function, **base_kwargs)

    original_call = self.call

    # Decorate the function to produce this layer's call method
    def _call_wrapper(*args, **kwargs):
      '''Turn the list of layer inputs into individual inputs for tf.op'''
      new_args = []

      for arg in args:
        if isinstance(arg, (list, tuple)):
          for sub_arg in arg:
            new_args.append(sub_arg)
        else:
          new_args.append(arg)

      new_kwargs = original_kwargs

      return original_call(*new_args, **new_kwargs)

    self.call = tf.__internal__.decorator.make_decorator(
        original_call, _call_wrapper)
