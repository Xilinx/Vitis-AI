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
"""Vitis cross layer weights equalizations."""

import tensorflow as tf
import numpy as np

from tensorflow_model_optimization.python.core.quantization.keras.vitis.graph_transformations import transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

LayerNode = transforms.LayerNode
LayerPattern = transforms.LayerPattern

keras = tf.keras
logger = common_utils.VAILogger


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
    logger.error('Invalid Equalization method: {}'.format(balance_method))

  scale = np.sqrt(a / b)

  # Clip scale
  scale = np.clip(scale, 1e-1, 10)

  # Stop scaling for small values
  i_max = np.max(head_weights, axis=0)
  o_max = np.max(tail_weights, axis=0)
  scale[(i_max + o_max) < weight_threshold] = 1
  return scale


def _cross_layer_equalize(head_conv,
                          tail_conv,
                          balance_method='max',
                          weight_threshold=0.1):
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
  scale = _calc_scale(head_weights, tail_weights, balance_method,
                      weight_threshold)

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


class CLEBase(transforms.Transform):
  """Base class for Cross layer equalization."""

  def __init__(self, forced_cle, balance_method, weight_threshold):
    self._forced_cle = forced_cle
    self._balance_method = balance_method
    self._weight_threshold = weight_threshold

  def _get_conv_pair(self, match_layer):
    """Helper function to get conv pair."""
    return None, None

  def replacement(self, match_layer):
    head_conv, tail_conv = self._get_conv_pair(match_layer)

    if head_conv and tail_conv:
      _cross_layer_equalize(head_conv, tail_conv, self._balance_method,
                            self._weight_threshold)
    return match_layer

  def custom_objects(self):
    return {}


class ConvConvCLE(CLEBase):
  """Cross layer equalization for Conv + Conv."""

  def pattern(self):
    activation_config = {} if self._forced_cle else {
        'activation': 'linear|relu'
    }
    return LayerPattern(
        'Conv2D|DepthwiseConv2D', {},
        [LayerPattern('Conv2D|DepthwiseConv2D', activation_config, [])])

  def _get_conv_pair(self, match_layer):
    tail_conv = match_layer
    head_conv = tail_conv.input_layers[0]
    logger.debug('Equalize ConvConv: {}({}) and {}({}).'.format(
        head_conv.layer['config']['name'], head_conv.layer['class_name'],
        tail_conv.layer['config']['name'], tail_conv.layer['class_name']))
    return head_conv, tail_conv


class ConvActConvCLE(CLEBase):
  """Cross layer equalization for Conv + Activation + Conv."""

  def pattern(self):
    activation_config = {} if self._forced_cle else {
        'activation': 'linear|relu'
    }
    return LayerPattern('Conv2D|DepthwiseConv2D', {}, [
        LayerPattern('Activation', activation_config, [
            LayerPattern('Conv2D|DepthwiseConv2D', {'activation': 'linear'}, [])
        ])
    ])

  def _get_conv_pair(self, match_layer):
    tail_conv = match_layer
    act_layer = tail_conv.input_layers[0]
    head_conv = act_layer.input_layers[0]
    logger.debug('Equalize ConvActConv: {}({}) and {}({}).'.format(
        head_conv.layer['config']['name'], head_conv.layer['class_name'],
        tail_conv.layer['config']['name'], tail_conv.layer['class_name']))
    return head_conv, tail_conv


class ConvReLUConvCLE(CLEBase):
  """Cross layer equalization for Conv + ReLU + Conv."""

  def pattern(self):
    relu_config = {} if self._forced_cle else {'max_value': 0}
    return LayerPattern('Conv2D|DepthwiseConv2D', {}, [
        LayerPattern('ReLU', relu_config, [
            LayerPattern('Conv2D|DepthwiseConv2D', {'activation': 'linear'}, [])
        ])
    ])

  def _get_conv_pair(self, match_layer):
    tail_conv = match_layer
    act_layer = tail_conv.input_layers[0]
    head_conv = act_layer.input_layers[0]
    logger.debug('Equalize ConvReLUConv: {}({}) and {}({}).'.format(
        head_conv.layer['config']['name'], head_conv.layer['class_name'],
        tail_conv.layer['config']['name'], tail_conv.layer['class_name']))
    return head_conv, tail_conv


class ConvReLUPadConvCLE(CLEBase):
  """Cross layer equalization for Conv + ReLU + ZeroPadding2D + Conv."""

  def pattern(self):
    relu_config = {} if self._forced_cle else {'max_value': 0}
    return LayerPattern('Conv2D|DepthwiseConv2D', {}, [
        LayerPattern('ZeroPadding2D', {}, [
            LayerPattern('ReLU', relu_config, [
                LayerPattern('Conv2D|DepthwiseConv2D', {'activation': 'linear'},
                             [])
            ])
        ])
    ])

  def _get_conv_pair(self, match_layer):
    tail_conv = match_layer
    pad_layer = tail_conv.input_layers[0]
    act_layer = pad_layer.input_layers[0]
    head_conv = act_layer.input_layers[0]
    logger.debug('Equalize ConvReLUPadConv: {}({}) and {}({}).'.format(
        head_conv.layer['config']['name'], head_conv.layer['class_name'],
        tail_conv.layer['config']['name'], tail_conv.layer['class_name']))
    return head_conv, tail_conv
