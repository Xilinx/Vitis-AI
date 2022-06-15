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
"""Add layer limit classes."""

import math
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_layer_limits
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import limit_utils

register_keras_serializable = tf.keras.utils.register_keras_serializable
BaseLayerLimits = vitis_layer_limits.BaseLayerLimits
logger = common_utils.VAILogger
Limit = vitis_layer_limits.Limit
LimitType = vitis_layer_limits.LimitType


@register_keras_serializable(package='Vitis', name='AddLimits')
class AddLimits(BaseLayerLimits):

  def __init__(self, dpu_target, limits={}):
    """Init with dpu target."""
    super(AddLimits, self).__init__(limits)
    self.eltwise_engine = dpu_target.get_eltwise_engine()
    self.target_name = dpu_target.get_name()
    self.dpu_target = dpu_target

    if self.is_supported():
      self.build_attr_limits()

  def get_layer_type(self):
    """Get current layer type."""
    return 'Add'

  def is_supported(self):
    """Check if current layer type is supported by target."""
    ElewType = self.eltwise_engine.ElewType
    return ElewType.Value('add') in self.eltwise_engine.elew_type

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
    is_in_attr_limit, attr_msgs = super(AddLimits, self).in_attr_limits(layer)
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

    NonlinearType = self.eltwise_engine.nonlinear.NonlinearType
    supported_acts = self.eltwise_engine.nonlinear.nonlinear_type
    supported_acts_str = [NonlinearType.Name(i) for i in supported_acts]

    # Only Add/Mul + Activation|ReLU will go into this function now.
    if isinstance(act_layer, keras.layers.Activation):

      def get_act_type(activation):
        if hasattr(activation, '__name__'):
          return activation.__name__
        return activation.__class__.__name__

      act_type_str = get_act_type(act_layer.activation)
      if act_type_str not in supported_acts_str:
        is_in_limit = False
        msgs.append('Add<{}> not supported, supported act types are {}'.format(
            act_type_str, supported_acts_str))

    elif isinstance(act_layer, keras.layers.ReLU):
      if act_layer.max_value and act_layer.max_value != 6.0:
        is_in_limit = False
        msgs.append(
            'Add<ReLU(max_value={})> is not supported, supported act types are {}'
            .format(act_layer.max_value, supported_acts_str))
      else:
        if act_layer.max_value == 6.0:
          act_type = NonlinearType.Value('relu_six')
        else:
          act_type = NonlinearType.Value('relu')
        if act_type not in supported_acts:
          is_in_limit = False
          msgs.append(
              'Add<{}> not supported, supported act types are {}'.format(
                  NonlinearType.Name(act_type), supported_acts_str))

    elif isinstance(act_layer, keras.layers.LeakyReLU):
      act_type = NonlinearType.Value('leaky_relu')
      if act_type not in supported_acts:
        is_in_limit = False
        msgs.append('Add<{}> not supported, supported act types are {}'.format(
            'LeakyReLU', supported_acts_str))
      elif not limit_utils._is_leaky_relu_quantizable(act_layer.alpha,
                                                      26. / 256.):
        is_in_limit = False
        msgs.append(
            'Add<LeakyReLU(alpha={})> is not supported, only alpha=0.1 is supported'
            .format(act_layer.alpha))

    else:
      raise NotImplementedError()

    return is_in_limit, msgs

  def get_config(self):
    """Get config for serialization."""
    return {'dpu_target': self.dpu_target}
