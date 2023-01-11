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
"""Vitis power-of-2 scale quantize related transforms."""

import collections
import inspect
import copy

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_wrapper
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_custom_wrapper
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


def _is_leaky_relu_quantizable(layer_node, alpha_target=0.1, threshold=1e-7):
  """Due to DPU constraints, only leaky_relu with alpha=0.1 is quantizable."""
  alpha = layer_node['config']['alpha']
  if abs(alpha - alpha_target) < threshold:
    return True
  else:
    return False


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
    metadata = match_layer.metadata

    match_layer_name = match_layer.layer['config']['name']
    quant_layer = vitis_quantize.VitisQuantize(
        self.input_quantizer,
        self.mode,
        name='{}_{}'.format('quant', match_layer_name))
    layer_config = keras.layers.serialize(quant_layer)
    layer_config['name'] = quant_layer.name

    ins_res = metadata.get('inspect_result', None)
    if ins_res:
      ins_res.device = 'INPUT'

    quant_metadata = copy.deepcopy(metadata)
    quant_ins_res = quant_metadata.get('inspect_result', None)
    if quant_ins_res:
      quant_ins_res.device = 'INPUT'
      quant_ins_res.origin_layers = []

    quant_layer_node = LayerNode(
        layer_config, input_layers=[match_layer], metadata=quant_metadata)
    return quant_layer_node

  def custom_objects(self):
    objs = vitis_quantizers._types_dict()
    objs.update({
        'VitisQuantize': vitis_quantize.VitisQuantize,
    })
    return objs


class CustomLayerWrapper(transforms.Transform):
  """Wrap the custom layer specifid by arguments. So that the subsequent
  operation will know which layer is custom layer

  Layer => CustomLayerWrapper(Layer inside)
  """

  def __init__(self, quantize_registry):
    super(CustomLayerWrapper, self).__init__()
    self.quantize_registry = quantize_registry
    self.custom_layer_type = quantize_registry.get_configs(
    )["custom_layer_type"]

  def pattern(self):
    return LayerPattern('.*')

  def replacement(self, match_layer):
    layer_node = match_layer.layer

    layer_type = layer_node['class_name']
    if layer_type not in self.custom_layer_type:
      if layer_type != "Vitis>QuantizeWrapper":
        return match_layer
      else:
        kernel_layer_type = layer_node['config']['layer']['class_name']
        if kernel_layer_type not in self.custom_layer_type:
          return match_layer

    layer = keras.layers.deserialize(
        layer_node, custom_objects=self.custom_objects())

    wrapped_layer = vitis_custom_wrapper.CustomOpWrapper(layer)

    wrapped_layer_node = LayerNode.from_layer(
        wrapped_layer, weights=match_layer.weights)
    return wrapped_layer_node

  def custom_objects(self):
    return {}


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
    ins_res = metadata.get('inspect_result', None)

    skip_layers = [
        'InputLayer', 'Vitis>VitisQuantize', 'Vitis>QuantizeWrapper',
        'Vitis>VitisConvBN', 'Vitis>VitisConvBNQuantize',
        'Vitis>VitisDepthwiseConvBN', 'Vitis>VitisDepthwiseConvBNQuantize'
    ]
    if layer_node['class_name'] in skip_layers:
      return match_layer

    quantize_config = metadata.get('quantize_config')

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
    if not quantize_config:
      if layer_node['class_name'] in ['ReLU', 'LeakyReLU', 'Activation']:

        def _get_act_type(layer_node):
          if layer_node['class_name'] == 'ReLU':
            return 'relu'
          elif layer_node['class_name'] == 'LeakyReLU':
            return 'leaky_relu'
          elif layer_node['class_name'] == 'Activation':
            return layer_node['config']['activation']
          elif layer_node['class_name'] == 'Vitis>VitisSigmoid':
            return 'hard_sigmoid'
          return 'activation'

        info_msg = 'Standalone activation `{}` layer {} is not supported.'
        logger.info(
            info_msg.format(
                _get_act_type(layer_node), layer_node['config']['name']))

        if ins_res:
          ins_res.add_notes(
              'Standalone activation `{}` is not supported.'.format(
                  _get_act_type(layer_node)))
        return match_layer
    if layer_node['class_name'] == 'TensorFlowOpLayer':
      layer = self.input_model.get_layer(layer_node['name'])
    else:
      layer = self.input_model.get_layer(layer_node['config']['name'])

    if not quantize_config and self.quantize_registry.supports(layer):
      quantize_config = self.quantize_registry.get_quantize_config(layer)

    if not quantize_config:
      info_msg = (
          'Layer {}({}) is not supported by DPU, it will not be quantized and may be mapped to run on CPU or other IPs. '
          'Please see {} for list of supported operations and APIs of vai_q_tensorflow2.'
      )
      logger.info(info_msg.format(layer.name, layer.__class__, ug_link))
      layer_name = match_layer.layer['config']['name']
      if ins_res:
        ins_res.add_notes('`{}` is not supported by target.'.format(
            layer.__class__.__name__))
      return match_layer

    # Check layer limits
    layer_limits = self.quantize_registry.get_layer_limits(layer)
    if layer_limits:
      is_in_limit, msgs = layer_limits.in_limits(layer)
      if not is_in_limit:
        for msg in msgs:
          logger.warning('Layer {}({})\'s {}'.format(layer.name,
                                                     layer.__class__.__name__,
                                                     msg))
        if ins_res:
          ins_res.add_notes(msgs)
        return match_layer

    if ins_res:
      ins_res.device = 'DPU'

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
    metadata = match_layer.metadata

    input_layer_type = input_layer_node.layer['class_name']
    if input_layer_type in ['InputLayer', 'Vitis>VitisQuantize'
                           ] or input_layer_type in self.quant_layers:
      return match_layer

    input_layer_name = input_layer_node.layer['config']['name']

    match_layer_name = match_layer.layer['config']['name']
    quant_layer = vitis_quantize.VitisQuantize(
        self.input_quantizer,
        self.mode,
        name='{}_{}'.format('quant', input_layer_name))
    layer_config = keras.layers.serialize(quant_layer)
    layer_config['name'] = quant_layer.name

    quant_metadata = copy.deepcopy(metadata)
    ins_res = quant_metadata.get('inspect_result', None)
    if ins_res:
      ins_res.device = 'INPUT'
      ins_res.origin_layers = []

    quant_layer_node = LayerNode(
        layer_config, input_layers=[input_layer_node], metadata=quant_metadata)

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
    self.freeze_bn_delay = None if freeze_bn_delay < 0 else freeze_bn_delay

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


class AnnotateConvAct(transforms.Transform):
  """Ensure FQ does not get placed between Conv-like and Activation."""

  def __init__(self, input_model, quantize_registry):
    super(AnnotateConvAct, self).__init__()
    self.input_model = input_model
    self.quantize_registry = quantize_registry

  def pattern(self):
    return LayerPattern(
        'ReLU|Activation|LeakyReLU|Vitis>VitisSigmoid',
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

    # Check conv layer limits
    layer_limits = self.quantize_registry.get_layer_limits(conv_layer)
    if layer_limits:
      is_in_limit, msgs = layer_limits.in_limits(conv_layer)
      if not is_in_limit:
        return match_layer

    act_quantize_config = act_metadata.get('quantize_config')
    if not act_quantize_config and self.quantize_registry.supports(act_layer):
      act_quantize_config = self.quantize_registry.get_quantize_config(
          act_layer)
    if not act_quantize_config:
      return match_layer

    # Check conv + act limits
    if layer_limits:
      is_in_limit, msgs = layer_limits.in_act_limits(conv_layer, act_layer)
      if not is_in_limit:
        for msg in msgs:
          logger.info('Layer {}({}): {}'.format(conv_layer.name,
                                                conv_layer.__class__.__name__,
                                                msg))
        conv_ins_res = conv_metadata.get('inspect_result', None)
        act_ins_res = act_metadata.get('inspect_result', None)
        if conv_ins_res and act_ins_res:
          conv_layer_name = conv_layer.name
          if conv_layer.name == act_ins_res.origin_layers[0]:
            conv_ins_res.add_notes(msgs)
          else:
            act_ins_res.add_notes(msgs)
        return match_layer

    if act_layer_node.layer['class_name'] not in ['Vitis>VitisSigmoid']:
      conv_layer_node.layer['config']['activation'] = \
        keras.activations.serialize(vitis_quantize_aware_activation.NoQuantizeActivation())
    #  conv_metadata['quantize_config'] = conv_quantize_config
    act_metadata['quantize_config'] = act_quantize_config

    return match_layer

  def custom_objects(self):
    return {
        'NoQuantizeActivation':
            vitis_quantize_aware_activation.NoQuantizeActivation,
    }


class AnnotateConvBNAct(transforms.Transform):
  """Ensure FQ does not get placed between ConvBN and Activation."""

  def pattern(self):
    return LayerPattern(
        'ReLU|Activation|LeakyReLU',
        inputs=[LayerPattern('Vitis>VitisConvBN|Vitis>VitisDepthwiseConvBN')])

  def replacement(self, match_layer):
    act_layer_node = match_layer
    conv_layer_node = act_layer_node.input_layers[0]

    if act_layer_node.layer[
        'class_name'] == 'Activation' and act_layer_node.layer['config'][
            'activation'] not in ['relu', 'linear']:
      return match_layer
    elif act_layer_node.layer[
        'class_name'] == 'LeakyReLU' and not _is_leaky_relu_quantizable(
            act_layer_node.layer, 26. / 256.):
      return match_layer

    conv_layer_node.layer['config']['activation'] = \
      keras.activations.serialize(vitis_quantize_aware_activation.NoQuantizeActivation())

    return match_layer

  def custom_objects(self):
    return {
        'NoQuantizeActivation':
            vitis_quantize_aware_activation.NoQuantizeActivation,
    }


class AnnotateAddAct(transforms.Transform):
  """Ensure FQ does not get placed between Add and Activation."""

  def __init__(self, input_model, quantize_registry):
    super(AnnotateAddAct, self).__init__()
    self.input_model = input_model
    self.quantize_registry = quantize_registry

  def pattern(self):
    return LayerPattern(
        'ReLU|Activation', inputs=[LayerPattern('Add|Multiply')])

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

    # Check add/mul layer limits
    layer_limits = self.quantize_registry.get_layer_limits(add_layer)
    if layer_limits:
      is_in_limit, msgs = layer_limits.in_limits(add_layer)
      if not is_in_limit:
        return match_layer

    act_quantize_config = act_metadata.get('quantize_config')
    if not act_quantize_config and self.quantize_registry.supports(act_layer):
      act_quantize_config = self.quantize_registry.get_quantize_config(
          act_layer)
    if not act_quantize_config:
      return match_layer

    # Check act limits
    if layer_limits:
      is_in_limit, msgs = layer_limits.in_act_limits(add_layer, act_layer)
      if not is_in_limit:
        for msg in msgs:
          logger.info('Layer {}({}): {}'.format(add_layer.name,
                                                add_layer.__class__.__name__,
                                                msg))
        add_ins_res = add_metadata.get('inspect_result', None)
        act_ins_res = act_metadata.get('inspect_result', None)
        if add_ins_res and act_ins_res:
          add_layer_name = add_layer.name
          if add_layer.name == act_ins_res.origin_layers[0]:
            add_ins_res.add_notes(msgs)
          else:
            act_ins_res.add_notes(msgs)
        return match_layer

    add_layer_node.metadata['quantize_config'] = \
      vitis_quantize_configs.NoQuantizeConfig()
    #  add_metadata['quantize_config'] = add_quantize_config
    act_metadata['quantize_config'] = act_quantize_config

    return act_layer_node

  def custom_objects(self):
    return {
        'NoQuantizeConfig': vitis_quantize_configs.NoQuantizeConfig,
    }


class AnnotatePoolAct(transforms.Transform):
  """Check dpu limits and annotate Pool and Activation."""

  def __init__(self, input_model, quantize_registry):
    super(AnnotatePoolAct, self).__init__()
    self.input_model = input_model
    self.quantize_registry = quantize_registry

  def pattern(self):
    return LayerPattern(
        'ReLU|Activation',
        inputs=[
            LayerPattern(
                'MaxPooling2D|Vitis>AveragePooling2D|Vitis>VitisGlobalAveragePooling2D'
            )
        ])

  def replacement(self, match_layer):
    act_layer_node = match_layer
    act_metadata = act_layer_node.metadata
    pool_layer_node = act_layer_node.input_layers[0]
    pool_metadata = pool_layer_node.metadata

    act_layer = self.input_model.get_layer(
        act_layer_node.layer['config']['name'])
    pool_layer = self.input_model.get_layer(
        pool_layer_node.layer['config']['name'])

    # No need to annotate if pool or act not quantizable
    pool_quantize_config = pool_metadata.get('quantize_config')
    if not pool_quantize_config and self.quantize_registry.supports(pool_layer):
      pool_quantize_config = self.quantize_registry.get_quantize_config(
          pool_layer)
    if not pool_quantize_config:
      return match_layer

    # Check pool layer limits
    layer_limits = self.quantize_registry.get_layer_limits(pool_layer)
    if layer_limits:
      is_in_limit, msgs = layer_limits.in_limits(pool_layer)
      if not is_in_limit:
        return match_layer

    act_quantize_config = act_metadata.get('quantize_config')
    if not act_quantize_config and self.quantize_registry.supports(act_layer):
      act_quantize_config = self.quantize_registry.get_quantize_config(
          act_layer)
    if not act_quantize_config:
      return match_layer

    # Check act limits
    if layer_limits:
      is_in_limit, msgs = layer_limits.in_act_limits(pool_layer, act_layer)
      if not is_in_limit:
        for msg in msgs:
          logger.info('Layer {}({}): {}'.format(pool_layer.name,
                                                pool_layer.__class__.__name__,
                                                msg))
        pool_ins_res = pool_metadata.get('inspect_result', None)
        act_ins_res = act_metadata.get('inspect_result', None)
        if pool_ins_res and act_ins_res:
          pool_layer_name = pool_layer.name
          if pool_layer.name == act_ins_res.origin_layers[0]:
            pool_ins_res.add_notes(msgs)
          else:
            act_ins_res.add_notes(msgs)
        return match_layer

    pool_layer_node.metadata['quantize_config'] = \
      vitis_quantize_configs.NoQuantizeConfig()
    pool_metadata['quantize_config'] = pool_quantize_config
    act_metadata['quantize_config'] = act_quantize_config

    return act_layer_node

  def custom_objects(self):
    return {
        'NoQuantizeConfig': vitis_quantize_configs.NoQuantizeConfig,
    }


class ConvertActivationSwish(transforms.Transform):
  """Convert keras.layers.Activation(swish) to VitisSigmoid and mul.

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
    act_metadata = act_layer_node.metadata
    act_layer_name = act_layer_node.layer['name']

    vitis_sigmoid_layer = vitis_activation.VitisSigmoid(name=act_layer_name)

    act_ins_res = act_metadata.get('inspect_result', None)
    if act_ins_res:
      act_ins_res.add_notes(
          'Convert activation `swish` to VitisSigmoid + Multiply.')

    vitis_sigmoid_layer_node = LayerNode.from_layer(
        vitis_sigmoid_layer,
        input_layers=[input_layer_node],
        metadata=act_metadata)

    mul_layer = keras.layers.Multiply(name=act_layer_name + '_mul')

    mul_metadata = copy.deepcopy(act_metadata)
    mul_ins_res = mul_metadata.get('inspect_result', None)
    if mul_ins_res:
      mul_ins_res.clear_notes()

    mul_layer_node = LayerNode.from_layer(
        mul_layer,
        input_layers=[input_layer_node, vitis_sigmoid_layer_node],
        metadata=mul_metadata)

    logger.debug('ConvertActivationSwish: {}({}).'.format(
        act_layer_node.layer['config']['name'],
        act_layer_node.layer['class_name']))

    return mul_layer_node


class ConvertActivationSigmoid(transforms.Transform):
  """Convert Activation(sigmoid) to VitisSigmoid.

  ActivationLayer(sigmoid) --> VitisSigmoid
  """

  def pattern(self):
    return LayerPattern(
        'Activation', config={'activation': 'sigmoid'}, inputs=[])

  def replacement(self, match_layer):
    act_layer_node = match_layer
    act_metadata = act_layer_node.metadata
    act_layer_name = act_layer_node.layer['name']

    vitis_sigmoid_layer = vitis_activation.VitisSigmoid(name=act_layer_name)

    act_ins_res = act_metadata.get('inspect_result', None)
    if act_ins_res:
      act_ins_res.add_notes('Convert activation `sigmoid` to VitisSigmoid.')

    logger.debug('ConvertActivationSigmoid: {}({}).'.format(
        act_layer_node.layer['config']['name'],
        act_layer_node.layer['class_name']))

    vitis_sigmoid_layer_node = LayerNode.from_layer(
        vitis_sigmoid_layer, metadata=act_metadata)
    return vitis_sigmoid_layer_node


class ConvertConvSwish(transforms.Transform):
  """Convert keras.layers.Conv(activation='swish') to VitisSigmoid and mul.

  ConvLayer(swish) --> Conv + VitisSigmoid + Multiply
  """

  def pattern(self):
    return LayerPattern(
        'Conv2D|DepthwiseConv2D|Conv2DTranspose|Dense',
        config={'activation': 'swish'})

  def replacement(self, match_layer):
    conv_layer_node = match_layer
    conv_metadata = conv_layer_node.metadata
    conv_layer_node.layer['config']['activation'] = 'linear'
    conv_layer_name = conv_layer_node.layer['name']

    vitis_sigmoid_layer = vitis_activation.VitisSigmoid(name=conv_layer_name +
                                                        '_sigmoid')

    sigmoid_metadata = copy.deepcopy(conv_metadata)
    sigmoid_ins_res = sigmoid_metadata.get('inspect_result', None)
    if sigmoid_ins_res:
      sigmoid_ins_res.add_notes(
          'Convert activation `swish` to VitisSigmoid + Multiply.')

    vitis_sigmoid_layer_node = LayerNode.from_layer(
        vitis_sigmoid_layer,
        input_layers=[conv_layer_node],
        metadata=sigmoid_metadata)

    mul_layer = keras.layers.Multiply(name=conv_layer_name + '_mul')

    mul_metadata = copy.deepcopy(conv_metadata)
    mul_ins_res = mul_metadata.get('inspect_result', None)
    if mul_ins_res:
      mul_ins_res.clear_notes()

    mul_layer_node = LayerNode.from_layer(
        mul_layer,
        input_layers=[conv_layer_node, vitis_sigmoid_layer_node],
        metadata=mul_metadata)

    logger.debug('ConvertConvSwish: {}({}).'.format(
        conv_layer_node.layer['config']['name'],
        conv_layer_node.layer['class_name']))

    return mul_layer_node


class ConvertConvSigmoid(transforms.Transform):
  """Convert keras.layers.Conv(activation='sigmoid') to VitisSigmoid.

  ConvLayer(sigmoid) --> Conv + VitisSigmoid
  """

  def pattern(self):
    return LayerPattern(
        'Conv2D|DepthwiseConv2D|Conv2DTranspose|Dense',
        config={'activation': 'sigmoid'})

  def replacement(self, match_layer):
    conv_layer_node = match_layer
    conv_metadata = conv_layer_node.metadata
    conv_layer_node.layer['config']['activation'] = 'linear'
    conv_layer_name = conv_layer_node.layer['name']

    vitis_sigmoid_layer = vitis_activation.VitisSigmoid(name=conv_layer_name +
                                                        '_sigmoid')

    sigmoid_metadata = copy.deepcopy(conv_metadata)
    sigmoid_ins_res = sigmoid_metadata.get('inspect_result', None)
    if sigmoid_ins_res:
      sigmoid_ins_res.add_notes('Convert activation `sigmoid` to VitisSigmoid.')

    vitis_sigmoid_layer_node = LayerNode.from_layer(
        vitis_sigmoid_layer,
        input_layers=[conv_layer_node],
        metadata=sigmoid_metadata)

    logger.debug('ConvertConvSigmoid: {}({}).'.format(
        conv_layer_node.layer['config']['name'],
        conv_layer_node.layer['class_name']))

    return vitis_sigmoid_layer_node


class ConvertHardSigmoidToDpuVersion(transforms.Transform):
  """Convert hard_sigmoid to VitisSigmoid.

  Pattern is from: tf.keras.layer.ReLU(6.)(x + 3.) * (1. / 6.))

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
    mul_metadata = match_layer.metadata
    relu6_layer_node = mul_layer_node.input_layers[0]
    relu6_metadata = relu6_layer_node.metadata
    relu6_layer_name = relu6_layer_node.layer['config']['name']
    add_layer_node = relu6_layer_node.input_layers[0]
    add_metadata = add_layer_node.metadata
    add_layer_name = add_layer_node.layer['config']['name']

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
          'Skipped ConvertHardSigmoidToDpuVersion because inner op does not match: {}({}) {}({}).'
          .format(
              mul_layer_node.layer['class_name'],
              mul_layer_node.layer['config'],
              add_layer_node.layer['class_name'],
              add_layer_node.layer['config'],
          ))
      return match_layer

    vitis_sigmoid_layer = vitis_activation.VitisSigmoid()

    sigmoid_metadata = copy.deepcopy(mul_metadata)
    sigmoid_ins_res = sigmoid_metadata.get('inspect_result', None)
    relu6_ins_res = relu6_metadata.get('inspect_result', None)
    add_ins_res = add_metadata.get('inspect_result', None)
    if sigmoid_ins_res and relu6_ins_res and add_ins_res:
      sigmoid_ins_res.add_notes('Convert `relu6(x+3)*(1/6)` to VitisSigmoid.')
      sigmoid_ins_res.merge_origin_layers(relu6_ins_res)
      sigmoid_ins_res.add_notes('Convert `relu6(x+3)*(1/6)` to VitisSigmoid.',
                                relu6_layer_name)
      sigmoid_ins_res.merge_origin_layers(add_ins_res)
      sigmoid_ins_res.add_notes('Convert `relu6(x+3)*(1/6)` to VitisSigmoid.',
                                add_layer_name)

    vitis_sigmoid_layer_node = LayerNode.from_layer(
        vitis_sigmoid_layer, metadata=sigmoid_metadata)

    logger.debug('ConvertHardSigmoidToDpuVersion: {}({}).'.format(
        mul_layer_node.layer['config']['name'],
        mul_layer_node.layer['class_name']))

    return vitis_sigmoid_layer_node


class ConvertGlobalAveragePooling2DToDpuVersion(transforms.Transform):
  """Convert keras.layers.GlobalAveragePooling2D to DPU version.

  GlobalAveragePooling2D --> VitisGlobalAveragePooling2D
  """

  def pattern(self):
    return LayerPattern('GlobalAveragePooling2D')

  def replacement(self, match_layer):
    pooling_layer_node = match_layer
    metadata = match_layer.metadata

    config = pooling_layer_node.layer['config']
    vitis_pooling_layer = vitis_pooling.VitisGlobalAveragePooling2D.from_config(
        pooling_layer_node.layer['config'])

    ins_res = metadata.get('inspect_result', None)
    if ins_res:
      ins_res.add_notes('Converted to VitisGlobalAveragePooling2D.')

    vitis_pooling_layer_node = LayerNode.from_layer(
        vitis_pooling_layer, metadata=metadata)
    return vitis_pooling_layer_node


class ConvertAveragePooling2DToDpuVersion(transforms.Transform):
  """Convert keras.layers.AveragePooling2D with DPU version.

  AveragePooling2D --> VitisAveragePooling2D
  """

  def pattern(self):
    return LayerPattern('AveragePooling2D')

  def replacement(self, match_layer):
    pooling_layer_node = match_layer
    metadata = match_layer.metadata

    config = pooling_layer_node.layer['config']
    vitis_pooling_layer = vitis_pooling.VitisAveragePooling2D.from_config(
        pooling_layer_node.layer['config'])

    ins_res = metadata.get('inspect_result', None)
    if ins_res:
      ins_res.add_notes('Converted to VitisAveragePooling2D.')

    vitis_pooling_layer_node = LayerNode.from_layer(
        vitis_pooling_layer, metadata=metadata)
    return vitis_pooling_layer_node


class ConvertLeakyReLUToDpuVersion(transforms.Transform):
  """Convert keras.layers.LeakyReLU to DPU version.

  LeakyReLU(alpha=0.1) --> LeakyReLU(alpha=26/256)
  """

  def pattern(self):
    return LayerPattern('LeakyReLU')

  def replacement(self, match_layer):
    relu_layer_node = match_layer
    metadata = match_layer.metadata

    alpha = relu_layer_node.layer['config']['alpha']
    if _is_leaky_relu_quantizable(relu_layer_node.layer):
      relu_layer_node.layer['config']['alpha'] = 26. / 256.

    ins_res = metadata.get('inspect_result', None)
    if ins_res:
      ins_res.add_notes(
          'Converted alpha 0.1 to 26./256. to match DPU implementation.')

    return relu_layer_node
