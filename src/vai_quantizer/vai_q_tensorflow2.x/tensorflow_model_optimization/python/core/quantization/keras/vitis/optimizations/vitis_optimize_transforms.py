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
"""Vitis general model optimization transforms."""

import collections
import copy

import tensorflow as tf
import numpy as np

from tensorflow_model_optimization.python.core.quantization.keras.vitis.graph_transformations import transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_conv_bn

LayerNode = transforms.LayerNode
LayerPattern = transforms.LayerPattern

keras = tf.keras
logger = common_utils.VAILogger


def _get_conv_bn_layers(bn_layer_node):
  bn_layer = bn_layer_node.layer
  conv_layer = bn_layer_node.input_layers[0].layer
  return conv_layer, bn_layer


def _get_weights(bn_layer_node):
  """Returns weight values for fused layer, including copying original values in unfused version."""

  return collections.OrderedDict(
      list(bn_layer_node.input_layers[0].weights.items()) +
      list(bn_layer_node.weights.items()))


def _get_params(conv_layer, bn_layer, relu_layer=None):
  """Retrieve conv_bn params within wrapped layers."""
  if 'use_bias' in conv_layer['config']:
    if conv_layer['config']['use_bias']:
      raise ValueError(
          'use_bias should not be set to True in a Conv layer when followed '
          'by BatchNormalization. The bias in the Conv would be redundant '
          'with the one in the BatchNormalization.')

    del conv_layer['config']['use_bias']

  if 'name' in bn_layer['config']:
    del bn_layer['config']['name']

  # TODO(pulkitb): remove key conflicts
  params = dict(
      list(conv_layer['config'].items()) + list(bn_layer['config'].items()))

  if relu_layer is not None:
    params['post_activation'] = keras.layers.deserialize(relu_layer)

  return params


def _get_folded_conv_weights(conv_layer_type, conv_kernel, conv_bias, bn_gamma,
                             bn_beta, bn_mm, bn_mv, bn_epsilon):
  if bn_gamma is not None:
    multiplier = bn_gamma / np.sqrt(bn_mv + bn_epsilon)
  else:
    multiplier = 1 / np.sqrt(bn_mv + bn_epsilon)

  if conv_layer_type is None:
    folded_conv_kernel = multiplier
  elif conv_layer_type in ['Dense']:
    folded_conv_kernel = conv_kernel * multiplier
  elif conv_layer_type in ['Conv2D']:
    mul_channel = conv_kernel.shape[-1]
    folded_conv_kernel = conv_kernel * multiplier.reshape(
        [1, 1, 1, mul_channel])
  elif conv_layer_type in ['DepthwiseConv2D', 'Conv2DTranspose']:
    mul_channel = conv_kernel.shape[-2]
    folded_conv_kernel = conv_kernel * multiplier.reshape(
        [1, 1, mul_channel, 1])

  if conv_bias is not None:
    folded_conv_bias = bn_beta + (conv_bias - bn_mm) * multiplier
  else:
    folded_conv_bias = bn_beta + (-bn_mm) * multiplier
  return folded_conv_kernel, folded_conv_bias


class ConvBNFold(transforms.Transform):
  """Fold batchnorm into previous convolution-like layer."""

  def pattern(self):
    return LayerPattern(
        'BatchNormalization', {},
        [LayerPattern('Conv2D|DepthwiseConv2D|Conv2DTranspose|Dense', {}, [])])

  def replacement(self, match_layer):
    bn_layer_node = match_layer
    bn_metadata = bn_layer_node.metadata
    conv_layer_node = match_layer.input_layers[0]
    conv_metadata = conv_layer_node.metadata

    # Only support folding when conv layer has linear activation
    if 'activation' in conv_layer_node.layer['config']:
      if conv_layer_node.layer['config']['activation'] != 'linear':
        logger.warning(
            'Only support Conv-BN folding for conv layers with linear activation, '
            'but layer `{}` has non-linear activation `{}`. Skip folding for it.'
            .format(conv_layer_node.layer['config']['name'],
                    conv_layer_node.layer['config']['activation']))
        return match_layer

    # TODO: see if can fetch the tensors without explict names
    conv_layer_type = conv_layer_node.layer['class_name']
    kernel_attr = 'depthwise_kernel:0' if conv_layer_type == 'DepthwiseConv2D' else 'kernel:0'
    conv_kernel = conv_layer_node.weights[kernel_attr]

    use_bias = conv_layer_node.layer['config']['use_bias']
    conv_bias = conv_layer_node.weights['bias:0'] if use_bias else None

    if bn_layer_node.layer['config']['scale'] is True:
      bn_gamma = bn_layer_node.weights['gamma:0']
    else:
      bn_gamma = None
    bn_beta = bn_layer_node.weights['beta:0']
    bn_mm = bn_layer_node.weights['moving_mean:0']
    bn_mv = bn_layer_node.weights['moving_variance:0']
    bn_epsilon = bn_layer_node.layer['config']['epsilon']

    folded_conv_layer = conv_layer_node.layer
    folded_conv_layer['config']['use_bias'] = True

    folded_conv_weights = collections.OrderedDict()
    folded_conv_weights[kernel_attr], folded_conv_weights[
        'bias:0'] = _get_folded_conv_weights(conv_layer_type, conv_kernel,
                                             conv_bias, bn_gamma, bn_beta,
                                             bn_mm, bn_mv, bn_epsilon)

    bn_ins_res = bn_metadata.get('inspect_result', None)
    conv_ins_res = conv_metadata.get('inspect_result', None)
    if conv_ins_res and bn_ins_res:
      bn_layer_name = bn_layer_node.layer['config']['name']
      conv_ins_res.merge_origin_layers(bn_ins_res)
      msg = 'Folded into previous layer {}.'.format(
          conv_layer_node.layer['config']['name'])
      conv_ins_res.add_notes(msg, layer_name=bn_layer_name)

    return LayerNode(
        folded_conv_layer, folded_conv_weights, metadata=conv_metadata)

  def custom_objects(self):
    return {}


class FakeConvBNFold(transforms.Transform):
  """Fake fold Conv + BatchNormalization layers, by wrapping them with VitisConvBN.

  Conv + BatchNormalization => VitisConvBN
  """

  def pattern(self):
    return LayerPattern('BatchNormalization', {},
                        [LayerPattern('Conv2D|DepthwiseConv2D', {}, [])])

  def replacement(self, match_layer):
    bn_layer_node = match_layer
    conv_layer_node = match_layer.input_layers[0]

    # Only support folding when conv layer has linear activation
    if 'activation' in conv_layer_node.layer['config']:
      if conv_layer_node.layer['config']['activation'] != 'linear':
        logger.warning(
            'Only support Conv-BN folding for conv layers with linear activation, '
            'but layer `{}` has non-linear activation `{}`. Skip folding for it.'
            .format(conv_layer_node.layer['config']['name'],
                    conv_layer_node.layer['config']['activation']))
        return match_layer

    bn_layer = keras.layers.deserialize(
        bn_layer_node.layer, custom_objects=self.custom_objects())
    conv_layer = keras.layers.deserialize(
        conv_layer_node.layer, custom_objects=self.custom_objects())

    conv_layer_type = conv_layer_node.layer['class_name']
    if conv_layer_type == 'Conv2D':
      conv_bn_layer = vitis_conv_bn.VitisConvBN(conv_layer, bn_layer)
    elif conv_layer_type == 'DepthwiseConv2D':
      conv_bn_layer = vitis_conv_bn.VitisDepthwiseConvBN(conv_layer, bn_layer)
    else:
      return match_layer

    conv_bn_weights = collections.OrderedDict(
        list(conv_layer_node.weights.items()) +
        list(bn_layer_node.weights.items()))

    conv_bn_layer_node = LayerNode.from_layer(
        conv_bn_layer, weights=conv_bn_weights)
    return conv_bn_layer_node

  def custom_objects(self):
    return {}


class RealConvBNFold(transforms.Transform):
  """Really fold VitisConvBN layers, by converting them to Conv layer.

  VitisConvBN => Conv
  """

  def pattern(self):
    return LayerPattern('Vitis>VitisConvBN', {}, [])

  def replacement(self, match_layer):
    conv_bn_layer_node = match_layer
    conv_layer = match_layer.layer['config']['conv_layer']

    conv_layer_type = conv_layer['class_name']
    kernel_attr = 'depthwise_kernel:0' if conv_layer_type == 'DepthwiseConv2D' else 'kernel:0'
    conv_kernel = conv_bn_layer_node.weights[kernel_attr]

    use_bias = conv_layer['config']['use_bias']
    conv_bias = conv_bn_layer_node.weights['bias:0'] if use_bias else None

    bn_layer = match_layer.layer['config']['bn_layer']
    if bn_layer['config']['scale'] is True:
      bn_gamma = conv_bn_layer_node.weights['gamma:0']
    else:
      bn_gamma = None
    bn_beta = conv_bn_layer_node.weights['beta:0']
    bn_mm = conv_bn_layer_node.weights['moving_mean:0']
    bn_mv = conv_bn_layer_node.weights['moving_variance:0']
    bn_epsilon = bn_layer['config']['epsilon']

    folded_conv_layer = conv_layer
    folded_conv_layer['name'] = conv_layer['config']['name']
    folded_conv_layer['config']['use_bias'] = True

    folded_conv_weights = collections.OrderedDict()
    folded_conv_weights[kernel_attr], folded_conv_weights[
        'bias:0'] = _get_folded_conv_weights(conv_layer_type, conv_kernel,
                                             conv_bias, bn_gamma, bn_beta,
                                             bn_mm, bn_mv, bn_epsilon)

    return LayerNode(folded_conv_layer, folded_conv_weights)

  def custom_objects(self):
    return {}


class ConvertBNToDWConv(transforms.Transform):
  """Fold batchnorm's mean and variance into gamma and beta, and convert to Depthwise Conv2D."""

  def pattern(self):
    return LayerPattern('BatchNormalization', {}, [])

  def replacement(self, match_layer):
    bn_layer_node = match_layer
    bn_metadata = bn_layer_node.metadata

    if bn_layer_node.layer['config']['scale'] is True:
      bn_gamma = bn_layer_node.weights['gamma:0']
    else:
      bn_gamma = None
    bn_beta = bn_layer_node.weights['beta:0']
    bn_mm = bn_layer_node.weights['moving_mean:0']
    bn_mv = bn_layer_node.weights['moving_variance:0']
    bn_epsilon = bn_layer_node.layer['config']['epsilon']

    folded_bn_layer = bn_layer_node.layer
    folded_bn_layer['config']['scale'] = True

    folded_bn_weights = collections.OrderedDict()
    folded_bn_weights['gamma:0'], folded_bn_weights[
        'beta:0'] = _get_folded_conv_weights(None, None, None, bn_gamma,
                                             bn_beta, bn_mm, bn_mv, bn_epsilon)
    num_channel = bn_mm.shape[0]
    folded_bn_weights['moving_mean:0'] = np.zeros(num_channel)
    folded_bn_weights['moving_variance:0'] = (1 - bn_epsilon) * np.ones(
        bn_mv.shape)

    # Convert folded_bn to depthwise conv layer with 1x1 kernel
    dw_conv_layer = keras.layers.DepthwiseConv2D(
        kernel_size=(1, 1),
        use_bias=True,
        name=bn_layer_node.layer['config']['name'])
    depthwise_kernel = folded_bn_weights['gamma:0'].reshape(
        1, 1, num_channel, 1)
    bias = folded_bn_weights['beta:0']
    dw_conv_layer_weights = {
        'depthwise_kernel:0': depthwise_kernel,
        'bias:0': bias
    }

    bn_ins_res = bn_metadata.get('inspect_result', None)
    if bn_ins_res:
      bn_ins_res.add_notes('Converted to DepthwiseConv2D')

    dw_conv_layer_node = LayerNode.from_layer(
        dw_conv_layer, weights=dw_conv_layer_weights, metadata=bn_metadata)
    return dw_conv_layer_node

  def custom_objects(self):
    return {}


class SeparateConvAct(transforms.Transform):
  """Separate activation in Conv-like layers.

  Conv-like(activation=xxx) -> Conv-like(activation=linear) + Activation(activation=xxx)
  """

  def pattern(self):
    return LayerPattern('Conv2D|DepthwiseConv2D|Conv2DTranspose|Dense', {}, [])

  def replacement(self, match_layer):
    conv_layer_node = match_layer
    conv_metadata = conv_layer_node.metadata
    conv_layer_name = conv_layer_node.layer['config']['name']
    act_type = conv_layer_node.layer['config']['activation']
    if isinstance(act_type, dict):
      act_type = act_type['class_name']

    if act_type == 'linear':
      return match_layer

    logger.debug('Separate layer {} activation: {}'.format(
        conv_layer_name, act_type))
    conv_layer_node.layer['config']['activation'] = 'linear'
    act_layer = keras.layers.Activation(
        activation=act_type, name=conv_layer_name + '_' + act_type)

    act_metadata = copy.deepcopy(conv_metadata)
    act_ins_res = act_metadata.get('inspect_result', None)
    if act_ins_res:
      msg = 'Seperate layer activation `{}`.'.format(act_type)
      act_ins_res.add_notes(msg, conv_layer_name)

    act_layer_node = LayerNode.from_layer(
        act_layer, input_layers=[conv_layer_node], metadata=act_metadata)
    return act_layer_node

  def custom_objects(self):
    return {}


class SeparableConv(transforms.Transform):
  """Separate  SeparableConv2D layers  to DepthWiseConv and PointWiseConv(normal conv).
  """

  def pattern(self):
    return LayerPattern('SeparableConv2D', {}, [])

  def replacement(self, match_layer):
    separable_conv_layer_node = match_layer
    separable_conv_metadata = separable_conv_layer_node.metadata
    separable_conv_layer_name = separable_conv_layer_node.layer['config'][
        'name']
    depthwise_kernel_size = separable_conv_layer_node.layer['config'][
        'kernel_size']
    depthwise_padding = separable_conv_layer_node.layer['config']['padding']
    depthwise_strides = separable_conv_layer_node.layer['config']['strides']
    depth_multiplier = separable_conv_layer_node.layer['config'][
        'depth_multiplier']
    depth_multiplier = separable_conv_layer_node.layer['config'][
        'depth_multiplier']
    dilation_rate = separable_conv_layer_node.layer['config']['dilation_rate']
    depthwise_weights = separable_conv_layer_node.weights['depthwise_kernel:0']
    separable_conv_data_format = separable_conv_layer_node.layer['config'][
        'data_format']

    pointwise_filters = separable_conv_layer_node.layer['config']['filters']

    logger.debug('Separable conv {} '.format(separable_conv_layer_name))

    separable_conv_depthwise_layer = keras.layers.DepthwiseConv2D(
        name=separable_conv_layer_name + '_' + 'depthwise_conv',
        kernel_size=depthwise_kernel_size,
        use_bias=False,
        padding=depthwise_padding,
        strides=depthwise_strides,
        data_format=separable_conv_data_format,
        depth_multiplier=depth_multiplier,
        dilation_rate=dilation_rate)
    separable_conv_depthwise_layer.layer = separable_conv_layer_node.layer
    separable_conv_depthwise_layer.layer['config'][
        'padding'] = separable_conv_layer_node.layer['config']['padding']
    depthwise_weights_all = {'depthwise_kernel:0': depthwise_weights}
    separable_conv_depthwise_layer_node = LayerNode.from_layer(
        separable_conv_depthwise_layer,
        weights=depthwise_weights_all,
        metadata=separable_conv_metadata)

    use_bias = separable_conv_layer_node.layer['config']['use_bias']
    separable_conv_pointwise_layer = keras.layers.Conv2D(
        name=separable_conv_layer_name + '_' + 'pointwise_conv',
        kernel_size=(1, 1),
        filters=pointwise_filters,
        use_bias=use_bias,
        data_format=separable_conv_data_format)

    separable_conv_pointwise_metadata = copy.deepcopy(separable_conv_metadata)
    pointwise_weights = separable_conv_layer_node.weights['pointwise_kernel:0']
    pointwise_bias = separable_conv_layer_node.weights[
        'bias:0'] if use_bias else None
    pointwise_weights_all = {'kernel:0': pointwise_weights}
    if use_bias:
      pointwise_weights_all['bias:0'] = pointwise_bias
    separable_conv_pointwise_layer_node = LayerNode.from_layer(
        separable_conv_pointwise_layer,
        input_layers=[separable_conv_depthwise_layer_node],
        weights=pointwise_weights_all,
        metadata=separable_conv_pointwise_metadata)
    return separable_conv_pointwise_layer_node

  def custom_objects(self):
    return {}


class FoldConvAct(transforms.Transform):
  """Fold activation into previous convolution layer."""

  def pattern(self):
    return LayerPattern(
        'ReLU|Activation', {},
        [LayerPattern('Conv2D|DepthwiseConv2D|Conv2DTranspose|Dense', {}, [])])

  def replacement(self, match_layer):
    act_layer_node = match_layer
    conv_layer_node = match_layer.input_layers[0]

    # Only support folding when conv layer has linear activation
    if 'activation' in conv_layer_node.layer['config']:
      if conv_layer_node.layer['config']['activation'] != 'linear':
        logger.warning(
            'Only support Conv-Activation folding for conv layers with linear activation, '
            'but layer `{}` has non-linear activation `{}`. Skip folding for it.'
            .format(conv_layer_node.layer['config']['name'],
                    conv_layer_node.layer['config']['activation']))
        return match_layer

    folded_conv_layer = conv_layer_node.layer
    # TODO(Xiao) Now only 'relu' is tested. Add code and test for other activations.
    if act_layer_node.layer['class_name'] == 'Activation':
      act_type = act_layer_node.layer['config']['activation']
    else:
      act_type = 'relu'
    folded_conv_layer['config']['activation'] = act_type
    folded_conv_weights = conv_layer_node.weights

    # This config tracks which layers get quantized, and whether they have a
    # custom QuantizeConfig.
    layer_metadata = {'quantize_config': None}

    return LayerNode(
        folded_conv_layer, folded_conv_weights, metadata=layer_metadata)

  def custom_objects(self):
    return {}


class RemoveDropout(transforms.Transform):
  """Remove Dropout layers."""

  def pattern(self):
    return LayerPattern('Dropout', {}, [LayerPattern('.*', {}, [])])

  def replacement(self, match_layer):
    input_layer = match_layer.input_layers[0]

    dropout_ins_res = match_layer.metadata.get('inspect_result', None)
    input_ins_res = input_layer.metadata.get('inspect_result', None)
    if dropout_ins_res and input_ins_res:
      input_ins_res.merge_origin_layers(dropout_ins_res)
      msg = 'Dropout removed in inference.'
      dropout_layer_name = match_layer.layer['config']['name']
      input_ins_res.add_notes(msg, layer_name=dropout_layer_name)

    return input_layer

  def custom_objects(self):
    return {}


class RemoveLayer(transforms.Transform):
  """Remove layer from the model."""

  def __init__(self, class_name, name='.*'):
    super(RemoveLayer, self).__init__()
    self._pattern = LayerPattern(class_name, {'name': name},
                                 [LayerPattern('.*', {}, [])])

  def pattern(self):
    return self._pattern

  def replacement(self, match_layer):
    return match_layer.input_layers[0]

  def custom_objects(self):
    return {}


class ConvertReLU6ToReLU(transforms.Transform):
  """Convert ReLU6 layers to ReLU layers, mainly used in CLE."""

  def pattern(self):
    return LayerPattern('ReLU|Activation')

  def replacement(self, match_layer):
    relu_layer = match_layer
    if relu_layer.layer.get('class_name') == 'ReLU' and relu_layer.layer[
        'config'].get('max_value') == 6.0:
      relu_layer.layer['config'].pop('max_value')
    elif relu_layer.layer.get(
        'class_name') == 'Activation' and relu_layer.layer['config'].get(
            'activation') == 'relu6':
      relu_layer.layer['config']['activation'] = 'relu'
    return match_layer

  def custom_objects(self):
    return {}


class ConvertTFOpToKeras(transforms.Transform):
  """Convert some of the TensorFlowOpLayer to equivalent keras.layers."""

  def pattern(self):
    return LayerPattern('TensorFlowOpLayer', {}, [])

  def replacement(self, match_layer):
    tf_op_layer = match_layer.layer
    op_name = tf_op_layer['config']['name']
    op_def = tf_op_layer['config']['node_def']
    op_type = op_def['op']

    if op_type == 'ConcatV2':
      keras_layer = keras.layers.Concatenate()
      keras_layer_node = LayerNode.from_layer(
          keras_layer, metadata=match_layer.metadata)
      return keras_layer_node
    else:
      return match_layer

  def custom_objects(self):
    return {}
