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
"""Vitis model optimization transforms."""

import collections
import inspect
import copy

import tensorflow as tf
import numpy as np

from tensorflow_model_optimization.python.core.quantization.keras.vitis import vitis_8bit_quantize_configs
from tensorflow_model_optimization.python.core.quantization.keras.vitis import vitis_8bit_quantize_registry
from tensorflow_model_optimization.python.core.quantization.keras.vitis.graph_transformations import transforms

LayerNode = transforms.LayerNode
LayerPattern = transforms.LayerPattern

keras = tf.keras


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


def _get_layer_node(fused_layer, weights):
  layer_config = keras.layers.serialize(fused_layer)
  layer_config['name'] = layer_config['config']['name']
  # This config tracks which layers get quantized, and whether they have a
  # custom QuantizeConfig.
  layer_metadata = {'quantize_config': None}

  return LayerNode(layer_config, weights, metadata=layer_metadata)


def _get_folded_conv_weights(conv_layer_type, conv_kernel, conv_bias, bn_gamma,
                             bn_beta, bn_mm, bn_mv, bn_epsilon):
  if bn_gamma is not None:
    multiplier = bn_gamma / np.sqrt(bn_mv + bn_epsilon)
  else:
    multiplier = 1 / np.sqrt(bn_mv + bn_epsilon)

  if not conv_layer_type:
    folded_conv_kernel = multiplier
  elif conv_layer_type in ['Conv2D', 'Dense']:
    mul_channel = conv_kernel.shape[-1]
    folded_conv_kernel = (conv_kernel.reshape(-1, mul_channel) *
                          multiplier).reshape(conv_kernel.shape)
  elif conv_layer_type in ['DepthwiseConv2D', 'Conv2DTranspose']:
    mul_channel = conv_kernel.shape[-2]
    conv_kernel_trans = conv_kernel.transpose(0, 1, 3, 2)
    folded_conv_kernel_trans = (conv_kernel_trans.reshape(-1, mul_channel) *
                                multiplier).reshape(conv_kernel_trans.shape)
    folded_conv_kernel = folded_conv_kernel_trans.transpose(0, 1, 3, 2)

  if conv_bias is not None:
    folded_conv_bias = bn_beta + (conv_bias - bn_mm) * multiplier
  else:
    folded_conv_bias = bn_beta + (-bn_mm) * multiplier
  return folded_conv_kernel, folded_conv_bias


class Conv2DBatchNormFold(transforms.Transform):
  """Fold batchnorm into previous convolution layer."""

  def pattern(self):
    return LayerPattern(
        'BatchNormalization', {},
        [LayerPattern('Conv2D|DepthwiseConv2D|Conv2DTranspose|Dense', {}, [])])

  def replacement(self, match_layer):
    bn_layer_node = match_layer
    conv_layer_node = match_layer.input_layers[0]

    # Only support folding when conv layer has linear activation
    if 'activation' in conv_layer_node.layer['config']:
      if conv_layer_node.layer['config']['activation'] != 'linear':
        print(
            'Vitis Warning: Only support Conv-BN folding for conv layers with linear activation, '
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

    # This config tracks which layers get quantized, and whether they have a
    # custom QuantizeConfig.
    layer_metadata = {'quantize_config': None}

    return LayerNode(
        folded_conv_layer, folded_conv_weights, metadata=layer_metadata)

  def custom_objects(self):
    return {}


class BatchNormFold(transforms.Transform):
  """Fold batchnorm's mean and variance into gamma and beta."""

  def pattern(self):
    return LayerPattern('BatchNormalization', {}, [])

  def replacement(self, match_layer):
    bn_layer_node = match_layer

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
    folded_bn_weights['moving_mean:0'] = np.zeros(bn_mm.shape)
    folded_bn_weights['moving_variance:0'] = (1 - bn_epsilon) * np.ones(
        bn_mv.shape)

    return LayerNode(
        folded_bn_layer, folded_bn_weights, metadata=match_layer.metadata)

  def custom_objects(self):
    return {}


class Conv2DActivationFold(transforms.Transform):
  """Fold activation into previous convolution layer."""

  def pattern(self):
    return LayerPattern(
        'ReLU|Activation', {},
        [LayerPattern('Conv2D|DepthwiseConv2D|Conv2DTranspose', {}, [])])

  def replacement(self, match_layer):
    act_layer_node = match_layer
    conv_layer_node = match_layer.input_layers[0]

    # Only support folding when conv layer has linear activation
    if 'activation' in conv_layer_node.layer['config']:
      if conv_layer_node.layer['config']['activation'] != 'linear':
        print(
            'Vitis Warning: Only support Conv-Activation folding for conv layers with linear activation, '
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
    return match_layer.input_layers[0]

  def custom_objects(self):
    return {}


def _calc_scale(head_weights,
                tail_weights,
                balance_method='max',
                weight_threshold=0.1):
  if balance_method == 'max':
    a = np.max(tail_weights, axis=0)
    b = np.max(head_weights, axis=0)
  elif balance_method == 'avg':
    a = np.mean(tail_weights, axis=0)
    b = np.mean(head_weights, axis=0)
  else:
    raise ValueError('Invalid Equalization method: {}'.format(balance_method))

  scale = np.sqrt(a / b)

  # Clip scale
  scale = np.clip(scale, 1e-1, 10)

  # Stop scaling for small values
  i_max = np.max(head_weights, axis=0)
  o_max = np.max(tail_weights, axis=0)
  scale[(i_max + o_max) < weight_threshold] = 1
  return scale


def _cross_layer_equalize(head_conv, tail_conv):
  """Cross Layer Equalization.

  This function re-implements the weight equalization technique proposed in the following paper.
  "Markus Nagel et al., Data-Free Quantization through Weight Equalization and Bias Correction", arXiv:1906.04721, 2019."
  """
  head_weights, tail_weights = [], []
  # Get head conv weights and bias
  if head_conv.layer['class_name'] == 'Conv2D':
    w = head_conv.weights['kernel:0']
    oc = w.shape[3]  # k * k * ic * oc for Conv2D
    head_weights.append(w.reshape(-1, oc))
    if head_conv.layer['config']['use_bias']:
      b = head_conv.weights['bias:0']
      head_weights.append(b.reshape(1, -1))
  else:
    w = head_conv.weights['depthwise_kernel:0']
    oc = w.shape[2]  # k * k * ic * 1 for DepthwiseConv2D
    head_weights.append(w.reshape(-1, oc))
    if head_conv.layer['config']['use_bias']:
      b = head_conv.weights['bias:0']
      head_weights.append(b.reshape(1, -1))

  # Get tail conv weights and bias
  if tail_conv.layer['class_name'] == 'Conv2D':
    w = tail_conv.weights['kernel:0']
    ic = w.shape[2]  # k * k * ic * oc for Conv2D
    w = w.transpose(0, 1, 3, 2)
    tail_weights.append(w.reshape(-1, ic))
  else:
    w = tail_conv.weights['depthwise_kernel:0']
    ic = w.shape[2]  # k * k * ic * 1 for DepthwiseConv2D
    tail_weights.append(w.reshape(-1, ic))

  head_weights = np.abs(np.concatenate(head_weights, axis=0))
  tail_weights = np.abs(np.concatenate(tail_weights, axis=0))

  # Calculate scale
  scale = _calc_scale(head_weights, tail_weights)

  #  print('Equalize: {} and {}.'.format(head_conv.layer['config']['name'],
  #  tail_conv.layer['config']['name']))
  # Scale head conv weights and bias
  if head_conv.layer['class_name'] == 'Conv2D':
    head_conv.weights['kernel:0'] *= scale.reshape(1, 1, 1, -1)
    if head_conv.layer['config']['use_bias']:
      head_conv.weights['bias:0'] *= scale
  else:
    head_conv.weights['depthwise_kernel:0'] *= scale.reshape(1, 1, -1, 1)
    if head_conv.layer['config']['use_bias']:
      head_conv.weights['bias:0'] *= scale

  # Scale tail conv weights and bias
  if tail_conv.layer['class_name'] == 'Conv2D':
    tail_conv.weights['kernel:0'] /= scale.reshape(1, 1, -1, 1)
  else:
    tail_conv.weights['depthwise_kernel:0'] /= scale.reshape(1, 1, -1, 1)


class ReplaceReLU6WithReLU(transforms.Transform):
  """Replace ReLU6 layers with ReLU layers, mainly used in CLE."""

  def pattern(self):
    return LayerPattern('ReLU', {'max_value': 6.0}, [])

  def replacement(self, match_layer):
    relu_layer = match_layer
    relu_layer.layer['config'].pop('max_value')
    return match_layer

  def custom_objects(self):
    return {}


class CLEBase(transforms.Transform):
  """Base class for Cross layer equalization."""

  def _get_conv_pair(self, match_layer):
    """Helper function to get conv pair."""
    return None, None

  def replacement(self, match_layer):
    head_conv, tail_conv = self._get_conv_pair(match_layer)
    if head_conv and tail_conv:
      _cross_layer_equalize(head_conv, tail_conv)
    return match_layer

  def custom_objects(self):
    return {}


class ConvConvCLE(CLEBase):
  """Cross layer equalization for conv + conv."""

  def pattern(self):
    return LayerPattern('Conv2D|DepthwiseConv2D', {},
                        [LayerPattern('Conv2D|DepthwiseConv2D', {}, [])])

  def _get_conv_pair(self, match_layer):
    tail_conv = match_layer
    head_conv = tail_conv.input_layers[0]
    return head_conv, tail_conv


class ConvActConvCLE(CLEBase):
  """Cross layer equalization for conv + act + conv."""

  def pattern(self):
    return LayerPattern('Conv2D|DepthwiseConv2D', {}, [
        LayerPattern('ReLU|Activation', {},
                     [LayerPattern('Conv2D|DepthwiseConv2D', {}, [])])
    ])

  def _get_conv_pair(self, match_layer):
    tail_conv = match_layer
    act_layer = tail_conv.input_layers[0]
    head_conv = act_layer.input_layers[0]
    return head_conv, tail_conv
