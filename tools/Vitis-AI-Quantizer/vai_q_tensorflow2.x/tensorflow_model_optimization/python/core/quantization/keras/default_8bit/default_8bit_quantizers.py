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
"""Quantizers specific to default 8-bit behavior."""

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quantizers


class Default8BitConvWeightsQuantizer(quantizers.LastValueQuantizer):
  """Quantizer for handling weights in Conv2D/DepthwiseConv2D layers."""

  def __init__(self):
    """Construct LastValueQuantizer with params specific for TFLite Convs."""

    super(Default8BitConvWeightsQuantizer, self).__init__(
        num_bits=8, per_axis=True, symmetric=True, narrow_range=True)

  def build(self, tensor_shape, name, layer):
    min_weight = layer.add_weight(
        name + '_min',
        shape=(tensor_shape[-1],),
        initializer=tf.keras.initializers.Constant(-6.0),
        trainable=False)
    max_weight = layer.add_weight(
        name + '_max',
        shape=(tensor_shape[-1],),
        initializer=tf.keras.initializers.Constant(6.0),
        trainable=False)

    return {'min_var': min_weight, 'max_var': max_weight}
