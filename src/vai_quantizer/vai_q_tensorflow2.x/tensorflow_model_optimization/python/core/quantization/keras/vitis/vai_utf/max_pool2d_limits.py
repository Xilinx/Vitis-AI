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
"""MaxPooling2D layer limit classes."""

import tensorflow as tf
from tensorflow import keras

from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_layer_limits
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import limit_utils

register_keras_serializable = tf.keras.utils.register_keras_serializable
BaseLayerLimits = vitis_layer_limits.BaseLayerLimits
logger = common_utils.VAILogger
Limit = vitis_layer_limits.Limit
LimitType = vitis_layer_limits.LimitType


@register_keras_serializable(package='Vitis', name='MaxPool2DLimits')
class MaxPool2DLimits(BaseLayerLimits):

  def __init__(self, dpu_target, limits={}):
    """Init with dpu target."""
    super(MaxPool2DLimits, self).__init__(limits)
    self.pool_engine = dpu_target.get_pool_engine()
    self.target_name = dpu_target.get_name()

    if self.is_supported():
      self.build_attr_limits()

  def get_layer_type(self):
    """Get current layer type."""
    return 'MaxPooling2D'

  def is_supported(self):
    """Check if current layer type is supported by target."""
    PoolType = self.pool_engine.PoolType
    return PoolType.Value('max') in self.pool_engine.pool_type

  def build_attr_limits(self):
    """Build attr limits."""
    # max kernel_size
    max_kernel_size_limit = limit_utils.str_to_pair_limit(
        self.pool_engine.max_limit.kernel_size)
    self.add_attr_limit('pool_size', max_kernel_size_limit)
    # max strides
    max_strides_limit = limit_utils.str_to_pair_limit(
        self.pool_engine.max_limit.stride)
    self.add_attr_limit('strides', max_strides_limit)

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
    is_in_attr_limit, attr_msgs = super(MaxPool2DLimits,
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
    return is_in_limit, msgs

  def in_act_limits(self, layer, act_layer):
    """Check layer + act_layer limits."""
    is_in_limit = True
    msgs = []

    NonlinearType = self.pool_engine.nonlinear.NonlinearType
    supported_acts = self.pool_engine.nonlinear.nonlinear_type
    supported_acts_str = [NonlinearType.Name(i) for i in supported_acts]

    # Only Pool + Activation|ReLU will go into this function now.
    if isinstance(act_layer, keras.layers.Activation):

      def get_act_type(activation):
        if hasattr(activation, '__name__'):
          return activation.__name__
        return activation.__class__.__name__

      act_type_str = get_act_type(act_layer.activation)
      if act_type_str not in supported_acts_str:
        is_in_limit = False
        msgs.append(
            'AveragePooling2D<{}> not supported, supported act types are {}'
            .format(act_type_str, supported_acts_str))

    elif isinstance(act_layer, keras.layers.ReLU):
      if act_layer.max_value and act_layer.max_value != 6.0:
        is_in_limit = False
        msgs.append(
            'AveragePooling2D<ReLU(max_value={})> is not supported, supported act types are {}'
            .format(act_layer.max_value, supported_acts_str))
      else:
        if act_layer.max_value == 6.0:
          act_type = NonlinearType.Value('relu_six')
        else:
          act_type = NonlinearType.Value('relu')
        if act_type not in supported_acts:
          is_in_limit = False
          msgs.append(
              'AveragePooling2D<{}> not supported, supported act types are {}'
              .format(NonlinearType.Name(act_type), supported_acts_str))

    else:
      raise NotImplementedError()

    return is_in_limit, msgs

  def get_config(self):
    """Get config for serialization."""
    return {'dpu_target': self.dpu_target}
