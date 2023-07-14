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
"""Conv2DTranspose layer limit classes."""

import math
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_layer_limits
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import limit_utils

register_keras_serializable = tf.keras.utils.register_keras_serializable
BaseLayerLimits = vitis_layer_limits.BaseLayerLimits
logger = common_utils.VAILogger
Limit = vitis_layer_limits.Limit
LimitType = vitis_layer_limits.LimitType


@register_keras_serializable(package='Vitis', name='Conv2DTransposeLimits')
class Conv2DTransposeLimits(BaseLayerLimits):

  def __init__(self, dpu_target, limits={}):
    """Init with dpu target."""
    super(Conv2DTransposeLimits, self).__init__(limits)
    self.conv_engine = dpu_target.get_conv_engine()
    self.target_name = dpu_target.get_name()
    self.dpu_target = dpu_target

    if self.is_supported():
      self.build_attr_limits()

  def get_layer_type(self):
    """Get current layer type."""
    return 'Conv2DTranspose'

  def is_supported(self):
    """Check if current layer type is supported by target."""
    return self.conv_engine.weight_bank != ''

  def build_attr_limits(self):
    """Build attr limits."""
    return

  def in_limits(self, layer):
    """Main entrance to check attr limits and other limits."""
    is_in_limit = True
    msgs = []

    if not self.is_supported():
      is_in_limit = False
      msgs.append('`{}` is not supported by target'.format(
          self.get_layer_type()))
      return is_in_limit, msgs

    # attr limits
    is_in_attr_limit, attr_msgs = super(Conv2DTransposeLimits,
                                        self).in_attr_limits(layer)
    # other limits
    is_in_other_limit, other_msgs = self.in_other_limits(layer)

    is_in_limit = is_in_attr_limit and is_in_other_limit
    msgs.extend(attr_msgs)
    msgs.extend(other_msgs)
    return is_in_limit, msgs

  def in_other_limits(self, layer):
    """Check limits not representable in attr limits."""
    is_in_limit = True
    msgs = []

    input_channel = layer.input_shape[-1]
    output_channel = layer.output_shape[-1]
    input_channel_parallel = self.conv_engine.input_channel_parallel
    bank_group = self.dpu_target.get_bank_group()
    output_channel_parallel = self.conv_engine.output_channel_parallel
    input_bank_name = self.conv_engine.input_bank[0]
    input_bank_depth = None
    for bank in bank_group:
      if bank.name == input_bank_name:
        input_bank = bank
        break
    input_bank_depth = input_bank.bank_depth

    # Kernel_size & Stride
    #   1 <= kernel_w/stride_w <= 16
    #   1 <= kernel_h/stride_h <= 16
    kernel_w, kernel_h = layer.kernel_size
    stride_w, stride_h = layer.strides
    ks = (kernel_w / stride_w, kernel_h / stride_h)
    ks_limit = '{}-{}'.format(1, 16)
    ks_limit = limit_utils.str_to_pair_limit(ks_limit)
    is_in_ks_limit, _ = ks_limit.in_limit(ks)
    is_in_limit &= is_in_ks_limit
    if not is_in_ks_limit:
      msg = '{}({}) exceed limit {}'.format('kernel/stride', ks, ks_limit)
      msgs.append(msg)

    # Paddings
    #   0 <= pad_left, pad_right <= (kernel_w-1)*dilation_w]
    #   0 <= pad_top, pad_bottom <= (kernel_h-1)*dilation_h]

    # Out_Size
    #   output_channel <= 256 * channel_parallel
    output_channel_limit = 256 * output_channel_parallel
    if output_channel > output_channel_limit:
      msg = '{}({}) exceed limit {}(={})'.format(
          'output_channel', output_channel, output_channel_limit,
          '256 * output_channel_parallel)')
      msgs.append(msg)
      is_in_limit = False

    return is_in_limit, msgs

  def in_act_limits(self, conv_layer, act_layer):
    """Check layer + act_layer limits."""
    is_in_limit = True
    msgs = []

    NonlinearType = self.conv_engine.nonlinear.NonlinearType
    supported_acts = self.conv_engine.nonlinear.nonlinear_type
    supported_acts_str = [NonlinearType.Name(i) for i in supported_acts]

    # Only conv-like(linear) + Activation|ReLU|LeakyReLU will go into this function now.
    if isinstance(act_layer, keras.layers.Activation):

      def get_act_type(activation):
        if hasattr(activation, '__name__'):
          return activation.__name__
        return activation.__class__.__name__

      act_type_str = get_act_type(act_layer.activation)
      if act_type_str not in supported_acts_str:
        is_in_limit = False
        msgs.append(
            'Conv2DTranspose<{}> not supported, supported act types are {}'
            .format(act_type_str, supported_acts_str))

    elif isinstance(act_layer, keras.layers.ReLU):
      if act_layer.max_value and act_layer.max_value != 6.0:
        is_in_limit = False
        msgs.append(
            'Conv2DTranspose<ReLU(max_value={})> is not supported, supported act types are {}'
            .format(act_layer.max_value, supported_acts_str))
      else:
        if act_layer.max_value == 6.0:
          act_type = NonlinearType.Value('relu_six')
        else:
          act_type = NonlinearType.Value('relu')
        if act_type not in supported_acts:
          is_in_limit = False
          msgs.append(
              'Conv2DTranspose<{}> not supported, supported act types are {}'
              .format(NonlinearType.Name(act_type), supported_acts_str))

    elif isinstance(act_layer, keras.layers.LeakyReLU):
      act_type = NonlinearType.Value('leaky_relu')
      if act_type not in supported_acts:
        is_in_limit = False
        msgs.append(
            'Conv2DTranspose<{}> not supported, supported act types are {}'
            .format('LeakyReLU', supported_acts_str))
      elif not limit_utils._is_leaky_relu_quantizable(act_layer.alpha,
                                                      26. / 256.):
        is_in_limit = False
        msgs.append(
            'Conv2DTranspose<LeakyReLU(alpha={})> is not supported, only alpha=0.1 is supported'
            .format(act_layer.alpha))

    elif isinstance(act_layer, vitis_activation.VitisSigmoid):
      act_type = NonlinearType.Value('hsigmoid')
      if act_type not in supported_acts:
        is_in_limit = False
        msgs.append(
            'Conv2DTranspose<{}> not supported, supported act types are {}'
            .format('hsigmoid', supported_acts_str))

    else:
      raise NotImplementedError()

    return is_in_limit, msgs

  def get_config(self):
    """Get config for serialization."""
    return {'dpu_target': self.dpu_target}
