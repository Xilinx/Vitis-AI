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
"""Vitis float scale quantize related transforms."""

import collections
import inspect
import copy

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_wrapper
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_configs
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantizers
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_quantize
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_pooling
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_conv_bn
from tensorflow_model_optimization.python.core.quantization.keras.vitis.graph_transformations import transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.optimizations import vitis_optimize_transforms

activations = tf.keras.activations
serialize_keras_object = tf.keras.utils.serialize_keras_object
deserialize_keras_object = tf.keras.utils.deserialize_keras_object
LayerNode = transforms.LayerNode
LayerPattern = transforms.LayerPattern
logger = common_utils.VAILogger

keras = tf.keras

# To be completed after release
ug_link = 'User Guide'


class InputLayerQuantize(transforms.Transform):
  """Quantizes InputLayer, by adding VitisQuantize Layer after it.

  InputLayer => InputLayer -> VitisQuantize Layer
  """

  def __init__(self, quantize_registry, mode):
    super(InputLayerQuantize, self).__init__()
    input_quantize_config = quantize_registry.get_input_quantize_config()
    input_quantizer = input_quantize_config['input_quantizer']
    self.input_quantizer = vitis_quantize_configs._make_quantizer(
        input_quantizer['quantizer_type'], input_quantizer['quantizer_params'])
    self.input_layers = input_quantize_config['input_layers']
    self.mode = mode

  def pattern(self):
    if self.input_layers:
      return LayerPattern('.*', config={'name': '|'.join(self.input_layers)})
    else:
      return LayerPattern('InputLayer')

  def replacement(self, match_layer):
    match_layer_name = match_layer.layer['config']['name']
    quant_layer = vitis_quantize.VitisQuantize(
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
        'VitisQuantize': vitis_quantize.VitisQuantize,
    })
    return objs


class LayersQuantize(transforms.Transform):
  """Quantize layers in the quantize support list, by wrapping them with QuantizeWrappers.
  Special layers like InputLayer and Convolution + BatchNormalization will be handled by
  other transformations.

  Layer => QuantizeWrapper(Layer inside)
  """

  def __init__(self, input_model, quantize_registry, mode):
    super(LayersQuantize, self).__init__()
    self.quantize_registry = quantize_registry
    self.mode = mode
    self.input_model = input_model

  def pattern(self):
    return LayerPattern('.*')

  def replacement(self, match_layer):
    layer_node = match_layer.layer
    metadata = match_layer.metadata

    skip_layers = [
        'InputLayer', 'Vitis>VitisQuantize', 'Vitis>QuantizeWrapper',
        'Vitis>VitisConvBN', 'Vitis>VitisConvBNQuantize',
        'Vitis>VitisDepthwiseConvBN', 'Vitis>VitisDepthwiseConvBNQuantize'
    ]
    if layer_node['class_name'] in skip_layers:
      return match_layer

    quantize_config = metadata.get('quantize_config')

    if not quantize_config:
      if layer_node['class_name'] in ['ReLU', 'Activation']:

        def _get_act_type(layer_node):
          if layer_node['class_name'] == 'ReLU':
            return 'relu'
          elif layer_node['class_name'] == 'Activation':
            return layer_node['config']['activation']
          return 'activation'

        info_msg = 'Standalone activation {} layer {} is not supported.'
        logger.debug(
            info_msg.format(_get_act_type(layer_node), layer_node['name']))
        return match_layer
    if layer_node['class_name'] == 'TensorFlowOpLayer':
      layer = self.input_model.get_layer(layer_node['name'])
    else:
      layer = self.input_model.get_layer(layer_node['config']['name'])
    if not quantize_config and self.quantize_registry.supports(layer):
      quantize_config = self.quantize_registry.get_quantize_config(layer)

    if not quantize_config:
      info_msg = ('Layer {}({}) is not quantizable.')
      logger.debug(info_msg.format(layer.name, layer.__class__))
      return match_layer

    quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        layer, quantize_config, self.mode)

    quant_layer_node = LayerNode.from_layer(
        quant_layer, weights=match_layer.weights, metadata=metadata)
    return quant_layer_node

  def custom_objects(self):
    objs = vitis_quantizers._types_dict()
    objs.update(vitis_quantize_configs._types_dict())
    objs.update({
        'QuantizeAwareActivation':
            vitis_quantize_aware_activation.QuantizeAwareActivation,
        'NoQuantizeActivation':
            vitis_quantize_aware_activation.NoQuantizeActivation,
        'QuantizeWrapper':
            vitis_quantize_wrapper.QuantizeWrapper,
        'VitisQuantize':
            vitis_quantize.VitisQuantize,
    })
    return objs


class LayersInputQuantize(transforms.Transform):
  """Quantize the inputs of the quantized layers. As the graph may be separated by
  some unquantizable layers, the quantized layers following those layers should have
  quantized inputs.

  Unquantizable Layer + QuantizeWrapper => Unquantizable Layer + QuantizeLayer + QuantizeWrapper
  """

  def __init__(self, input_model, quantize_registry, mode):
    super(LayersInputQuantize, self).__init__()
    self.input_model = input_model
    self.model_config = self.input_model.get_config()
    input_quantize_config = quantize_registry.get_input_quantize_config()
    input_quantizer = input_quantize_config['input_quantizer']
    self.input_quantizer = vitis_quantize_configs._make_quantizer(
        input_quantizer['quantizer_type'], input_quantizer['quantizer_params'])
    self.mode = mode
    self.quant_layers = [
        'Vitis>QuantizeWrapper', 'Vitis>VitisConvBN',
        'Vitis>VitisConvBNQuantize', 'Vitis>VitisDepthwiseConvBN',
        'Vitis>VitisDepthwiseConvBNQuantize'
    ]

  def pattern(self):
    return LayerPattern('|'.join(self.quant_layers), {}, [])

  def replacement(self, match_layer):
    layer_node = match_layer.layer
    layer_name = layer_node['config']['layer']['config']['name']
    layer_type = layer_node['config']['layer']['class_name']

    inbound_nodes = layer_node['inbound_nodes']
    input_layer_names = []
    for inbound_node in inbound_nodes:
      for connection_info in inbound_node:
        input_layer_names.append(connection_info[0])

    #(TODO) Handle multiple inputs layers
    if len(input_layer_names) > 1:
      logger.debug(
          'Skip quantize inputs of layer {}({}) with multiple inputs: {}.'
          .format(layer_name, layer_type, len(input_layer_names)))
      return match_layer

    for layer in self.model_config['layers']:
      if layer['name'] == input_layer_names[0]:
        input_layer = layer
    input_layer_type = input_layer['class_name']

    if input_layer_type == 'Vitis>VitisQuantize':
      return match_layer

    if input_layer_type in self.quant_layers:
      if input_layer_type == 'Vitis>QuantizeWrapper':
        quantize_config = input_layer['config']['quantize_config']
        if quantize_config['config']:
          activation_quantizers = quantize_config['config'][
              'activation_quantizers']
          output_quantizers = quantize_config['config']['output_quantizers']
          if output_quantizers or activation_quantizers:
            return match_layer
      else:
        return match_layer

    input_layer_name = input_layer_names[0]
    quant_layer = vitis_quantize.VitisQuantize(
        self.input_quantizer,
        self.mode,
        name='{}_{}'.format('quant', input_layer_name))
    layer_config = keras.layers.serialize(quant_layer)
    layer_config['name'] = quant_layer.name
    quant_layer_node = LayerNode(layer_config, input_layers=[])

    match_layer.input_layers = [quant_layer_node]

    logger.debug('Quantize input of layer: {}({}).'.format(
        layer_name, layer_type))

    return match_layer

  def custom_objects(self):
    objs = vitis_quantizers._types_dict()
    objs.update(vitis_quantize_configs._types_dict())
    objs.update({
        'QuantizeAwareActivation':
            vitis_quantize_aware_activation.QuantizeAwareActivation,
        'NoQuantizeActivation':
            vitis_quantize_aware_activation.NoQuantizeActivation,
        'QuantizeWrapper':
            vitis_quantize_wrapper.QuantizeWrapper,
        'VitisQuantize':
            vitis_quantize.VitisQuantize,
    })
    return objs


class ConvBNQuantize(transforms.Transform):
  """Quantize Conv + BatchNormalization layers, by wrapping them with VitisConvBNQuantize.

  VitisConvBN => VitisConvBNQuantize
  """

  def __init__(self, quantize_registry, mode, freeze_bn_delay):
    super(ConvBNQuantize, self).__init__()
    self.quantize_registry = quantize_registry
    self.mode = mode
    self.freeze_bn_delay = freeze_bn_delay

  def pattern(self):
    return LayerPattern('Vitis>VitisConvBN|Vitis>VitisDepthwiseConvBN', {}, [])

  def replacement(self, match_layer):
    conv_bn_layer = keras.layers.deserialize(
        match_layer.layer, custom_objects=self.custom_objects())

    quantize_config = None
    if self.quantize_registry.supports(conv_bn_layer):
      quantize_config = self.quantize_registry.get_quantize_config(
          conv_bn_layer)

    if not quantize_config:
      return match_layer

    conv_layer_type = match_layer.layer['config']['conv_layer']['class_name']
    if conv_layer_type == 'Conv2D':
      quant_conv_bn_layer = vitis_conv_bn.VitisConvBNQuantize(
          conv_layer=conv_bn_layer.conv_layer,
          bn_layer=conv_bn_layer.bn_layer,
          activation=activations.serialize(conv_bn_layer.activation),
          freeze_bn_delay=self.freeze_bn_delay,
          quantize_config=quantize_config,
          mode=self.mode)
    elif conv_layer_type == 'DepthwiseConv2D':
      quant_conv_bn_layer = vitis_conv_bn.VitisDepthwiseConvBNQuantize(
          conv_layer=conv_bn_layer.conv_layer,
          bn_layer=conv_bn_layer.bn_layer,
          activation=activations.serialize(conv_bn_layer.activation),
          freeze_bn_delay=self.freeze_bn_delay,
          quantize_config=quantize_config,
          mode=self.mode)
    else:
      return match_layer

    quant_conv_bn_weights = match_layer.weights
    quant_conv_bn_layer_node = LayerNode.from_layer(
        quant_conv_bn_layer, weights=quant_conv_bn_weights)
    return quant_conv_bn_layer_node

  def custom_objects(self):
    return {}


class ConvBNQuantizeFold(transforms.Transform):
  """Fold VitisConvBNQuantize layers, by converting them to quantized Conv layer.

  VitisConvBNQuantize => QuantizeWrapper(Conv)
  """

  def pattern(self):
    return LayerPattern(
        'Vitis>VitisConvBNQuantize|Vitis>VitisDepthwiseConvBNQuantize', {}, [])

  def replacement(self, match_layer):
    conv_bn_layer_node = match_layer
    conv_layer = match_layer.layer['config']['conv_layer']
    bn_layer = match_layer.layer['config']['bn_layer']

    conv_layer_type = conv_layer['class_name']
    kernel_attr = 'depthwise_kernel:0' if conv_layer_type == 'DepthwiseConv2D' else 'kernel:0'
    conv_kernel = conv_bn_layer_node.weights[kernel_attr]

    use_bias = conv_layer['config']['use_bias']
    conv_bias = conv_bn_layer_node.weights['bias:0'] if use_bias else None

    if bn_layer['config']['scale'] is True:
      bn_gamma = conv_bn_layer_node.weights['gamma:0']
    else:
      bn_gamma = None
    bn_beta = conv_bn_layer_node.weights['beta:0']
    bn_mm = conv_bn_layer_node.weights['moving_mean:0']
    bn_mv = conv_bn_layer_node.weights['moving_variance:0']
    bn_epsilon = bn_layer['config']['epsilon']

    # Build folded conv layer
    folded_conv_layer = conv_layer
    folded_conv_layer['name'] = conv_layer['config']['name']
    folded_conv_layer['config']['use_bias'] = True

    activation = match_layer.layer['config']['activation']
    if activation['class_name'] == 'Vitis>QuantizeAwareActivation':
      activation = activation['config']['activation']
    folded_conv_layer['config']['activation'] = activation

    folded_conv_layer = keras.layers.deserialize(
        folded_conv_layer, custom_objects=self.custom_objects())

    quantize_config = deserialize_keras_object(
        match_layer.layer['config']['quantize_config'])
    mode = match_layer.layer['config']['mode']
    quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        folded_conv_layer, quantize_config, mode)

    # Build quant layer weights, inherit quantize config variables from match_layer
    quant_layer_weights = copy.deepcopy(match_layer.weights)
    if bn_layer['config']['scale'] is True:
      del quant_layer_weights['gamma:0']
    del quant_layer_weights['beta:0']
    del quant_layer_weights['moving_mean:0']
    del quant_layer_weights['moving_variance:0']
    quant_layer_weights[kernel_attr], quant_layer_weights[
        'bias:0'] = vitis_optimize_transforms._get_folded_conv_weights(
            conv_layer_type, conv_kernel, conv_bias, bn_gamma, bn_beta, bn_mm,
            bn_mv, bn_epsilon)

    quant_layer_node = LayerNode.from_layer(
        quant_layer, weights=quant_layer_weights)
    return quant_layer_node

  def custom_objects(self):
    return {}


class NoQuantInConvAct(transforms.Transform):
  """Ensure FQ does not get placed between Conv-like and Activation."""

  def __init__(self, input_model, quantize_registry):
    super(NoQuantInConvAct, self).__init__()
    self.input_model = input_model
    self.quantize_registry = quantize_registry

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
    act_metadata = act_layer_node.metadata
    conv_layer_node = act_layer_node.input_layers[0]
    conv_metadata = conv_layer_node.metadata

    act_layer = self.input_model.get_layer(
        act_layer_node.layer['config']['name'])
    conv_layer = self.input_model.get_layer(
        conv_layer_node.layer['config']['name'])

    # No need to annotate if conv or act not quantizable
    conv_quantize_config = conv_metadata.get('quantize_config')
    if not conv_quantize_config and self.quantize_registry.supports(conv_layer):
      conv_quantize_config = self.quantize_registry.get_quantize_config(
          conv_layer)
    if not conv_quantize_config:
      return match_layer

    act_quantize_config = act_metadata.get('quantize_config')
    if not act_quantize_config and self.quantize_registry.supports(act_layer):
      act_quantize_config = self.quantize_registry.get_quantize_config(
          act_layer)
    if not act_quantize_config:
      return match_layer

    conv_layer_node.layer['config']['activation'] = \
      keras.activations.serialize(vitis_quantize_aware_activation.NoQuantizeActivation())
    act_metadata['quantize_config'] = act_quantize_config

    return match_layer

  def custom_objects(self):
    return {
        'NoQuantizeActivation':
            vitis_quantize_aware_activation.NoQuantizeActivation,
    }


class NoQuantInConvBNAct(transforms.Transform):
  """Ensure FQ does not get placed between ConvBN and Activation."""

  def pattern(self):
    return LayerPattern(
        'ReLU|Activation',
        inputs=[LayerPattern('Vitis>VitisConvBN|Vitis>VitisDepthwiseConvBN')])

  def replacement(self, match_layer):
    act_layer_node = match_layer
    conv_layer_node = act_layer_node.input_layers[0]

    if act_layer_node.layer[
        'class_name'] == 'Activation' and act_layer_node.layer['config'][
            'activation'] not in ['relu', 'linear']:
      return match_layer

    conv_layer_node.layer['config']['activation'] = \
      keras.activations.serialize(vitis_quantize_aware_activation.NoQuantizeActivation())

    return match_layer

  def custom_objects(self):
    return {
        'NoQuantizeActivation':
            vitis_quantize_aware_activation.NoQuantizeActivation,
    }


class NoQuantInAddAct(transforms.Transform):
  """Ensure FQ does not get placed between Add and Activation."""

  def __init__(self, input_model, quantize_registry):
    super(NoQuantInAddAct, self).__init__()
    self.input_model = input_model
    self.quantize_registry = quantize_registry

  def pattern(self):
    return LayerPattern('ReLU|Activation', inputs=[LayerPattern('Add')])

  def replacement(self, match_layer):
    act_layer_node = match_layer
    act_metadata = act_layer_node.metadata
    add_layer_node = act_layer_node.input_layers[0]
    add_metadata = add_layer_node.metadata

    act_layer = self.input_model.get_layer(
        act_layer_node.layer['config']['name'])
    add_layer = self.input_model.get_layer(
        add_layer_node.layer['config']['name'])

    # No need to annotate if add/mul or act not quantizable
    add_quantize_config = add_metadata.get('quantize_config')
    if not add_quantize_config and self.quantize_registry.supports(add_layer):
      add_quantize_config = self.quantize_registry.get_quantize_config(
          add_layer)
    if not add_quantize_config:
      return match_layer

    act_quantize_config = act_metadata.get('quantize_config')
    if not act_quantize_config and self.quantize_registry.supports(act_layer):
      act_quantize_config = self.quantize_registry.get_quantize_config(
          act_layer)
    if not act_quantize_config:
      return match_layer

    add_layer_node.metadata['quantize_config'] = \
      vitis_quantize_configs.NoQuantizeConfig()
    act_metadata['quantize_config'] = act_quantize_config

    return act_layer_node

  def custom_objects(self):
    return {
        'NoQuantizeConfig': vitis_quantize_configs.NoQuantizeConfig,
    }
