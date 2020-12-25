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
"""Vitis 8-bit quantize related transforms."""

import collections
import inspect

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras.vitis import vitis_quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis import vitis_8bit_quantize_configs
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_quantize_layer
from tensorflow_model_optimization.python.core.quantization.keras.vitis import vitis_quantizers
from tensorflow_model_optimization.python.core.quantization.keras.vitis.graph_transformations import transforms

LayerNode = transforms.LayerNode
LayerPattern = transforms.LayerPattern

keras = tf.keras


class InputLayerQuantize(transforms.Transform):
  """Quantizes InputLayer, by adding QuantizeLayer after it.

  InputLayer => InputLayer -> QuantizeLayer
  """

  def __init__(self, input_quantizer, mode):
    super(InputLayerQuantize).__init__()
    self.input_quantizer = vitis_8bit_quantize_configs._make_quantizer(
        input_quantizer['quantizer_type'], input_quantizer['quantizer_params'])
    self.mode = mode

  def pattern(self):
    return LayerPattern('InputLayer')

  def replacement(self, match_layer):
    match_layer_name = match_layer.layer['config']['name']
    quant_layer = vitis_quantize_layer.QuantizeLayer(
        self.input_quantizer,
        self.mode,
        name='{}_{}'.format('quant', match_layer_name))
    layer_config = keras.layers.serialize(quant_layer)
    layer_config['name'] = quant_layer.name

    quant_layer_node = LayerNode(layer_config, input_layers=[match_layer])

    return quant_layer_node

  def custom_objects(self):
    objs = vitis_quantizers._types_dict()
    objs.update({
        'QuantizeLayer': vitis_quantize_layer.QuantizeLayer,
    })
    return objs


class ConvActivationQuantize(transforms.Transform):
  """Ensure FQ does not get placed between Add and Activation."""

  def pattern(self):
    return LayerPattern(
        'ReLU|Activation',
        inputs=[
            LayerPattern(
                'Conv2D|DepthwiseConv2D|Conv2DTranspose|Dense',
                config={'activation': 'linear'})
        ])

  def replacement(self, match_layer):
    act_layer_node = match_layer
    conv_layer_node = act_layer_node.input_layers[0]

    conv_layer_node.layer['config']['activation'] = \
      keras.activations.serialize(vitis_quantize_aware_activation.NoQuantizeActivation())

    return match_layer

  def custom_objects(self):
    return {
        'NoQuantizeActivation':
            vitis_quantize_aware_activation.NoQuantizeActivation,
    }


class AddActivationQuantize(transforms.Transform):
  """Ensure FQ does not get placed between Add and Activation."""

  def pattern(self):
    return LayerPattern('ReLU|Activation', inputs=[LayerPattern('Add')])

  def replacement(self, match_layer):
    relu_layer_node = match_layer
    add_layer_node = relu_layer_node.input_layers[0]

    add_layer_node.metadata['quantize_config'] = \
      vitis_8bit_quantize_configs.NoQuantizeConfig()

    return match_layer

  def custom_objects(self):
    return {
        'NoQuantizeConfig': vitis_8bit_quantize_configs.NoQuantizeConfig,
    }
