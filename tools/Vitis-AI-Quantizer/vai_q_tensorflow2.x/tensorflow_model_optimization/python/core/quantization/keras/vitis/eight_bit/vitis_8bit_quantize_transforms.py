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
import copy

import tensorflow as tf
from tensorflow.python.keras import activations

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

  def __init__(self, quantize_registry, mode):
    super(LayersQuantize, self).__init__()
    self.quantize_registry = quantize_registry
    self.mode = mode

  def pattern(self):
    return LayerPattern('.*')

  def replacement(self, match_layer):
    layer_node = match_layer.layer

    skip_layers = [
        'InputLayer', 'Vitis>VitisQuantize', 'Vitis>QuantizeWrapper',
        'Vitis>VitisConvBN', 'Vitis>VitisConvBNQuantize',
        'Vitis>VitisDepthwiseConvBN', 'Vitis>VitisDepthwiseConvBNQuantize'
    ]
    if layer_node['class_name'] in skip_layers:
      return match_layer

    # DPU now only supports below activations:
    # 1. Linear
    # 2. ReLU/ReLU6
    # 3. LeakyReLU(alpha==0.1): alpha will be converted to 26/256
    # 4. Sigmoid/HardSigmoid: sigmoid will be converted to hard sigmoid by default
    # 5. Swish/HardSwish: swish will be converted to hard swish by default
    # 6. Softmax: softmax will be mapped to run on Softmax IP
    #
    # Other acitvations will not be quantized, for example:
    # 1. keras.layers.Activation layer with other types of activation function will not be quantized.
    # 2. keras.layers.LeakyReLU layer with alpha!=0.1 will not be quantized.
    if layer_node['class_name'] == 'Softmax':
      return match_layer

    if layer_node['class_name'] == 'Activation':
      activation = layer_node['config']['activation']
      if activation == 'softmax':
        return match_layer

      if activation not in ['linear', 'relu']:
        info_msg = (
            'Activation layer {}(activation={}) is not supported by DPU, '
            'it will not be quantized and may be mapped to run on CPU or other IPs. '
            'Please see {} for list of supported operations and APIs of vai_q_tensorflow2.'
        )
        logger.info(info_msg.format(layer_node['name'], activation, ug_link))
        return match_layer

    if layer_node['class_name'] == 'LeakyReLU':
      alpha = layer_node['config']['alpha']
      if not abs(alpha - 26. / 256.) < 1e-7:
        info_msg = (
            'LeakyReLU layer {}(alpha={}) is not supported by DPU, '
            'currently DPU only supports LeakyReLU layer with `alpha=0.1`.'
            'it will not be quantized and may be mapped to run on CPU or other IPs. '
            'Please see {} for list of supported operations and APIs of vai_q_tensorflow2.'
        )
        logger.info(info_msg.format(layer_node['name'], alpha, ug_link))
        return match_layer

    #TODO(Xiao): Deal with user-defined custom layers
    layer = keras.layers.deserialize(
        layer_node, custom_objects=self.custom_objects())
    quantize_config = None

    quantize_config = match_layer.metadata.get('quantize_config')
    if not quantize_config and self.quantize_registry.supports(layer):
      quantize_config = self.quantize_registry.get_quantize_config(layer)

    if not quantize_config:
      info_msg = (
          'Layer {}({}) is not supported by DPU, it will not be quantized and may be mapped to run on CPU or other IPs. '
          'Please see {} for list of supported operations and APIs of vai_q_tensorflow2.'
      )
      logger.info(info_msg.format(layer.name, layer.__class__, ug_link))
      return match_layer

    quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        layer, quantize_config, self.mode)

    quant_layer_node = LayerNode.from_layer(
        quant_layer, weights=match_layer.weights)
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

  def __init__(self, quantize_registry, mode):
    super(LayersInputQuantize, self).__init__()
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
    return LayerPattern('|'.join(self.quant_layers), {}, [LayerPattern('.*')])

  def replacement(self, match_layer):
    input_layer_node = match_layer.input_layers[0]

    input_layer_type = input_layer_node.layer['class_name']
    if input_layer_type in ['InputLayer', 'Vitis>VitisQuantize'
                           ] or input_layer_type in self.quant_layers:
      return match_layer

    input_layer_name = input_layer_node.layer['config']['name']
    quant_layer = vitis_quantize.VitisQuantize(
        self.input_quantizer,
        self.mode,
        name='{}_{}'.format('quant', input_layer_name))
    layer_config = keras.layers.serialize(quant_layer)
    layer_config['name'] = quant_layer.name
    quant_layer_node = LayerNode(layer_config, input_layers=[input_layer_node])

    match_layer.input_layers = [quant_layer_node]

    logger.debug('Quantize input of layer: {}({}).'.format(
        match_layer.layer['config']['layer']['config']['name'],
        match_layer.layer['config']['layer']['class_name']))

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


class ConvActivationAnnotate(transforms.Transform):
  """Ensure FQ does not get placed between Conv-like and Activation."""

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


class ConvBNActivationAnnotate(transforms.Transform):
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


class AddActivationAnnotate(transforms.Transform):
  """Ensure FQ does not get placed between Add and Activation."""

  def pattern(self):
    return LayerPattern('ReLU|Activation', inputs=[LayerPattern('Add')])

  def replacement(self, match_layer):
    act_layer_node = match_layer
    add_layer_node = act_layer_node.input_layers[0]

    if act_layer_node.layer[
        'class_name'] == 'Activation' and act_layer_node.layer['config'][
            'activation'] not in ['relu', 'linear']:
      return match_layer

    add_layer_node.metadata['quantize_config'] = \
      vitis_quantize_configs.NoQuantizeConfig()

    return match_layer

  def custom_objects(self):
    return {
        'NoQuantizeConfig': vitis_quantize_configs.NoQuantizeConfig,
    }


class ReplaceActivationSwish(transforms.Transform):
  """Replace keras.layers.Activation(swish) with VitisSigmoid and mul.

  ActivationLayer(swish) --> VitisSigmoid + Multiply
  """

  def pattern(self):
    return LayerPattern(
        'Activation',
        config={'activation': 'swish'},
        inputs=[LayerPattern('.*')])

  def replacement(self, match_layer):
    input_layer_node = match_layer.input_layers[0]
    act_layer_node = match_layer
    act_layer_name = act_layer_node.layer['name']

    vitis_sigmoid_layer = vitis_activation.VitisSigmoid(name=act_layer_name +
                                                        '_sigmoid')
    vitis_sigmoid_layer_node = LayerNode.from_layer(
        vitis_sigmoid_layer, input_layers=[input_layer_node])

    mul_layer = keras.layers.Multiply(name=act_layer_name + '_mul')
    mul_layer_node = LayerNode.from_layer(
        mul_layer, input_layers=[input_layer_node, vitis_sigmoid_layer_node])

    logger.debug('ReplaceActivationSwish: {}({}).'.format(
        act_layer_node.layer['config']['name'],
        act_layer_node.layer['class_name']))

    return mul_layer_node


class ReplaceActivationSigmoid(transforms.Transform):
  """Replace Activation(sigmoid) with VitisSigmoid.

  ActivationLayer(sigmoid) --> VitisSigmoid
  """

  def pattern(self):
    return LayerPattern(
        'Activation', config={'activation': 'sigmoid'}, inputs=[])

  def replacement(self, match_layer):
    act_layer_node = match_layer
    act_layer_name = act_layer_node.layer['name']

    vitis_sigmoid_layer = vitis_activation.VitisSigmoid(name=act_layer_name +
                                                        '_sigmoid')
    vitis_sigmoid_layer_node = LayerNode.from_layer(vitis_sigmoid_layer)

    logger.debug('ReplaceActivationSigmoid: {}({}).'.format(
        act_layer_node.layer['config']['name'],
        act_layer_node.layer['class_name']))

    return vitis_sigmoid_layer_node


class ReplaceConvSwish(transforms.Transform):
  """Replace keras.layers.Conv(activation='swish') with VitisSigmoid and mul.

  ConvLayer(swish) --> Conv + VitisSigmoid + Multiply
  """

  def pattern(self):
    return LayerPattern(
        'Conv2D|DepthwiseConv2D|Conv2DTranspose|Dense',
        config={'activation': 'swish'})

  def replacement(self, match_layer):
    conv_layer_node = match_layer
    conv_layer_node.layer['config']['activation'] = 'linear'
    act_layer_name = conv_layer_node.layer['name'] + '_act'

    vitis_sigmoid_layer = vitis_activation.VitisSigmoid(name=act_layer_name +
                                                        '_sigmoid')
    vitis_sigmoid_layer_node = LayerNode.from_layer(
        vitis_sigmoid_layer, input_layers=[conv_layer_node])

    mul_layer = keras.layers.Multiply(name=act_layer_name + '_mul')
    mul_layer_node = LayerNode.from_layer(
        mul_layer, input_layers=[conv_layer_node, vitis_sigmoid_layer_node])

    logger.debug('ReplaceConvSwish: {}({}).'.format(
        conv_layer_node.layer['config']['name'],
        conv_layer_node.layer['class_name']))

    return mul_layer_node


class ReplaceConvSigmoid(transforms.Transform):
  """Replace keras.layers.Conv(activation='sigmoid') with VitisSigmoid.

  ConvLayer(sigmoid) --> Conv + VitisSigmoid
  """

  def pattern(self):
    return LayerPattern(
        'Conv2D|DepthwiseConv2D|Conv2DTranspose|Dense',
        config={'activation': 'sigmoid'})

  def replacement(self, match_layer):
    conv_layer_node = match_layer
    conv_layer_node.layer['config']['activation'] = 'linear'
    act_layer_name = conv_layer_node.layer['name'] + '_act'

    vitis_sigmoid_layer = vitis_activation.VitisSigmoid(name=act_layer_name +
                                                        '_sigmoid')
    vitis_sigmoid_layer_node = LayerNode.from_layer(
        vitis_sigmoid_layer, input_layers=[conv_layer_node])

    logger.debug('ReplaceConvSigmoid: {}({}).'.format(
        conv_layer_node.layer['config']['name'],
        conv_layer_node.layer['class_name']))

    return vitis_sigmoid_layer_node


class ReplaceHardSigmoid(transforms.Transform):
  """Replace hard_sigmoid with VitisSigmoid.

  TensorFlowOpLayer(AddV2) + Relu(6) + TensorflowOpLayer(Mul) --> VitisSigmoid
  """

  def pattern(self):
    return LayerPattern(
        'TensorFlowOpLayer|TFOpLambda',
        config={},
        inputs=[
            LayerPattern(
                'ReLU',
                config={'max_value': 6.0},
                inputs=[LayerPattern('TensorFlowOpLayer|TFOpLambda')])
        ])

  def replacement(self, match_layer):
    mul_layer_node = match_layer
    relu6_layer_node = mul_layer_node.input_layers[0]
    add_layer_node = relu6_layer_node.input_layers[0]

    def _match_inner_opfunc(layer_node, target_op, target_func):
      class_name = layer_node.layer['class_name']
      if class_name == 'TFOpLambda':
        return layer_node.layer['config']['function'] == target_func
      else:
        return layer_node.layer['config']['node_def']['op'] == target_op

    if not _match_inner_opfunc(
        mul_layer_node, 'Mul', 'math.multiply') or not _match_inner_opfunc(
            add_layer_node, 'AddV2', '__operators__.add'):
      logger.debug(
          'Skipped ReplaceHardSigmoid because inner op does not match: {}({}) {}({}).'
          .format(
              mul_layer_node.layer['class_name'],
              mul_layer_node.layer['config'],
              add_layer_node.layer['class_name'],
              add_layer_node.layer['config'],
          ))
      return match_layer

    vitis_sigmoid_layer = vitis_activation.VitisSigmoid()
    vitis_sigmoid_layer_node = LayerNode.from_layer(vitis_sigmoid_layer)

    logger.debug('ReplaceHardSigmoid: {}({}).'.format(
        mul_layer_node.layer['config']['name'],
        mul_layer_node.layer['class_name']))

    return vitis_sigmoid_layer_node


class ReplaceGlobalAveragePooling2D(transforms.Transform):
  """Replace keras.layers.GlobalAveragePooling2D with Vitis version.

  GlobalAveragePooling2D --> VitisGlobalAveragePooling2D
  """

  def pattern(self):
    return LayerPattern('GlobalAveragePooling2D')

  def replacement(self, match_layer):
    pooling_layer_node = match_layer

    config = pooling_layer_node.layer['config']
    config.pop('name')
    vitis_pooling_layer = vitis_pooling.VitisGlobalAveragePooling2D.from_config(
        pooling_layer_node.layer['config'])
    vitis_pooling_layer_node = LayerNode.from_layer(vitis_pooling_layer)
    return vitis_pooling_layer_node


class ReplaceAveragePooling2D(transforms.Transform):
  """Replace keras.layers.AveragePooling2D with Vitis version.

  AveragePooling2D --> VitisAveragePooling2D
  """

  def pattern(self):
    return LayerPattern('AveragePooling2D')

  def replacement(self, match_layer):
    pooling_layer_node = match_layer

    config = pooling_layer_node.layer['config']
    config.pop('name')
    vitis_pooling_layer = vitis_pooling.VitisAveragePooling2D.from_config(
        pooling_layer_node.layer['config'])
    vitis_pooling_layer_node = LayerNode.from_layer(vitis_pooling_layer)
    return vitis_pooling_layer_node


class ReplaceLeakyReLU(transforms.Transform):
  """Replace keras.layers.LeakyReLU with Vitis version.

  LeakyReLU(alpha=0.1) --> LeakyReLU(alpha=26/256)
  """

  def pattern(self):
    return LayerPattern('LeakyReLU')

  def replacement(self, match_layer):
    relu_layer_node = match_layer

    alpha = relu_layer_node.layer['config']['alpha']
    if abs(alpha - 0.1) < 1e-7:
      relu_layer_node.layer['config']['alpha'] = 26. / 256.
    return relu_layer_node
