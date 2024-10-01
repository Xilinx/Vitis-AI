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
"""Vitis activation layers."""

import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

__all__ = ['VitisSigmoid','VitisSoftmax','VitisLayernorm']

register_keras_serializable = tf.keras.utils.register_keras_serializable
serialize_keras_object = tf.keras.utils.serialize_keras_object
deserialize_keras_object = tf.keras.utils.deserialize_keras_object
logger = common_utils.VAILogger


@register_keras_serializable(package='Vitis', name='VitisSigmoid')
class VitisSigmoid(tf.keras.layers.Layer):
  """Vitis sigmoid layer.

  This is an simplified sigmoid layer to mimic the hardware sigmoid layer behaviour.
  """

  def __init__(self, **kwargs):
    """Create a Vitis sigmoid Layer.

    Args:
      quantizer: `Quantizer` used to quantize tensors.
      **kwargs: Additional keyword arguments to be passed to the keras layer.
    """
    super(VitisSigmoid, self).__init__(**kwargs)

  def call(self, inputs):

    def hard_sigmoid_dpu(x):
      """A hardware friendly version of sigmoid function.

         hard_sigmoid: out = relu6(x + 3.) * 1. / 6.
         hard_sigmoid_dpu: out = relu6(x + 3.) * 2731 / 2 ^ 14
      """
      x = tf.cast(x, tf.float32)
      x_out = tf.keras.activations.relu(x + 3, max_value=6.)
      x_out = x_out * 2731. / 16384.
      return x_out

    return hard_sigmoid_dpu(inputs)

  def get_config(self):
    return super(VitisSigmoid, self).get_config()

  @classmethod
  def from_config(cls, config):
    config = config.copy()
    return cls(**config)


def _types_dict():
  return {
      'VitisSigmoid': VitisSigmoid,
  }

@register_keras_serializable(package='Vitis', name='VitisSoftmax')
class VitisSoftmax(tf.keras.layers.Layer):
  """Vitis softmax layer.

  This is an simplified softmax layer to mimic the hardware softmax layer behaviour.
  """

  def __init__(self, axis, **kwargs):
    """Create a Vitis softmax Layer.

    Args:
      quantizer: `Quantizer` used to quantize tensors.
      **kwargs: Additional keyword arguments to be passed to the keras layer.
    """
    self.axis = int(axis)
    super(VitisSoftmax, self).__init__(**kwargs)
  

  def call(self, inputs):
    #compute inv of x with dim
    def compute_inv(x):
      exp_mask     = 0x7F800000
      mantissa_mask= 0x007FFFFF
      mantissa_Q   = 0x00008000
      x_f = tf.dtypes.cast(x, tf.float32)
      B_x = tf.bitcast(x_f, tf.uint32)
      exponent = bitwise_ops.right_shift((B_x & exp_mask), 23)
      mantissa = tf.where(((B_x & mantissa_Q)==0), bitwise_ops.right_shift((B_x & mantissa_mask),16), bitwise_ops.right_shift((B_x & mantissa_mask),16)+1)
      inv_exponent = 253-exponent
      inv_mantissa = tf.round(256*128/(128+mantissa)-128)
      inv_x_val = bitwise_ops.left_shift(tf.cast(inv_exponent, tf.int32),23) + bitwise_ops.left_shift(tf.cast(inv_mantissa, tf.int32),16)
      inv_x = tf.bitcast(inv_x_val, tf.float32)
      return inv_x

    def aie_add_v16(v16): # input: tensor (L, 16)
      v8 = v16[:, 0:8] + v16[:, 8:]
      v4 = v8[:, 0:4] + v8[:, 4:]
      v2 = v4[:, 0:2] + v4[:, 2:]
      v1 = v2[:, 0] + v2[:, 1]
      #v1 = v16[:, 0] + v16[:, 1]

      return v1  # output: tensor (L,)


    def hard_softmax_dpu(axis, x):
      """A hardware friendly version of softmax function.
      """
      #the last reduce num 
      reduce_count_16 = 16

      #it is for quantize tensor simulate
      #x = tf.cast(x, tf.uint8)

      exp_appr = tf.cast(tf.cast(tf.math.exp(x), tf.bfloat16), tf.float32)
      #get the shape of exp_appr tensor for reduce 16
      tran_shape = [-1]
      #-2 is without first dim and last dim
      for i in range(exp_appr._rank()-2):
        tran_shape.append(tf.shape(exp_appr)[i+1])
      #use // for truediv
      if((exp_appr.shape[-1])%reduce_count_16 != 0):
        logger.error('softmax axis is not times of 16')
         
      tran_shape.append(tf.shape(exp_appr)[-1]//reduce_count_16)
      tran_shape.append(reduce_count_16)
      #reshape the tensor for reuduce 16
      exp_appr_tran = tf.reshape(exp_appr, shape=tran_shape)
      #use axis=-2 for without transpose 
      sum_part = tf.math.reduce_sum(exp_appr_tran, axis=-2 )
      #change the sum_part for 2 dims
      sum_part = tf.reshape(sum_part,(-1,reduce_count_16))
      #reduce for 16 num
      sum_v32 = aie_add_v16(sum_part)
      #reshape the exp result  to origin dims-1
      sum_v32 = tf.reshape(sum_v32, tran_shape[:-2])
      #expand_dims for broadcast
      sum_v32 = tf.expand_dims(sum_v32,-1)
      #get the inv
      sum_inv = compute_inv(sum_v32)
      #calculate the result of multiply
      sm_res = tf.math.multiply(exp_appr,sum_inv)
      x_out = tf.cast(tf.cast(sm_res, tf.bfloat16), tf.float32)
      # below is quantize tensor used, for round method and pow
      #the quantize process
      #x_out = tf.pow(x_out, shift_up)
      #use round_method 3 in json
      #x_out = tf.math.floor(x_out)
      #use narrow_range
      #x_out = tf.clip_by_value(x_out, -127, 127)
      return x_out

    return hard_softmax_dpu(self.axis, inputs)

  def get_config(self):
    base_config = super(VitisSoftmax, self).get_config()
    config = {
        'axis': self.axis
    }
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    config = config.copy()
    axis = config.pop('axis')
    return cls(
        axis=axis,
        **config)


def _types_dict():
  return {
      'VitisSoftmax': VitisSoftmax,
  }

@register_keras_serializable(package='Vitis', name='VitisLayernorm')
class VitisLayernorm(tf.keras.layers.LayerNormalization):
  """Vitis layernorm layer.

  This is an simplified layernorm layer to mimic the hardware layernorm layer behaviour.
  """

  def __init__(self, axis, epsilon, **kwargs):
    """Create a Vitis layernorm Layer.

    Args:
      quantizer: `Quantizer` used to quantize tensors.
      **kwargs: Additional keyword arguments to be passed to the keras layer.
    """
    
    super(VitisLayernorm, self).__init__(**kwargs)
    #override the axis and epsilon
    self.axis = axis
    self.epsilon = epsilon

  def call(self, inputs):
    #compute inv of x with dim
    def newton_iteration(y, iterator, x2):
      threehalfs = 1.5
      for i in range(iterator):
        y = tf.cast(y, tf.float32)
        v3_0 = y*y
        v3_0 = tf.cast(tf.cast(v3_0,tf.bfloat16), tf.float32)
        v3_1 = -x2*v3_0
        v3_1 = tf.cast(tf.cast(v3_1,tf.bfloat16), tf.float32)
        v3_2 = tf.add(threehalfs, v3_1)
        v3_2 = tf.cast(tf.cast(v3_2,tf.bfloat16), tf.float32)
        y = y *v3_2
        y = tf.cast(y,tf.bfloat16)
      return y
    def Q_rsqrt_float(number):
      x2 = tf.cast(tf.cast(number,tf.bfloat16), tf.float32)
      x2 = x2 * 0.5
      x2 = tf.cast(tf.cast(x2,tf.bfloat16), tf.float32)
      y = tf.cast(number,tf.bfloat16)
      i = tf.bitcast(y, tf.int16)
      aaa = bitwise_ops.right_shift(i,1)
      bbb = i & 1
      ccc = tf.where((aaa%2 != 0), aaa+bbb, aaa)
      i = 0x5f37 - ccc
      y = tf.bitcast(i, tf.bfloat16)
      y = newton_iteration(y, 4, x2)
      return y

    def aie_add_v8(v8): # input: tensor (L, 16)
      v4 = v8[:, 0:4] + v8[:, 4:]
      v2 = v4[:, 0:2] + v4[:, 2:]
      v1 = v2[:, 0] + v2[:, 1]

      return v1  # output: tensor (L,)


    def hard_layernorm_dpu(axis, epsilon,gamma, beta, x):
      """A hardware friendly version of softmax function.
      """
      #the last reduce num 
      reduce_count_8 = 8

      #it is for quantize tensor simulate
      #x = tf.cast(x, tf.uint8)

      inp = tf.cast(tf.cast(x, tf.bfloat16), tf.float32)
      #get the shape of inp tensor for reduce 16
      tran_shape = [-1]
      #-2 is without first dim and last dim
      for i in range(inp._rank()-2):
        tran_shape.append(tf.shape(inp)[i+1])
      #use // for truediv
      if((inp.shape[-1])%reduce_count_8 != 0):
        logger.error('softmax axis is not times of 16')
         
      tran_shape.append(tf.shape(inp)[-1]//reduce_count_8)
      tran_shape.append(reduce_count_8)
      #reshape the tensor for reuduce 16
      inp_tran = tf.reshape(inp, shape=tran_shape)
      #use axis=-2 for without transpose 
      sum_part = tf.math.reduce_sum(inp_tran, axis=-2 )
      #change the sum_part for 2 dims
      sum_part = tf.reshape(sum_part,(-1,reduce_count_8))
      #reduce for 8 num
      sum_aie = aie_add_v8(sum_part)
      #reshape the exp result  to origin dims-1
      #sum_v8 = tf.reshape(sum_v8, tran_shape[:-2])
      #expand_dims for broadcast
      #sum_aie = tf.expand_dims(sum_v8,-1)
      numel_inv = 1/(tran_shape[-1]*tran_shape[-2])
      mean_aie = sum_aie * tf.cast(numel_inv,tf.float32)
      mean =  tf.reshape(mean_aie, tran_shape[:-2])
      mean = tf.expand_dims(mean,-1)
      # x-u
      sub =  inp - mean
      sub = tf.cast(tf.cast(sub, tf.bfloat16), tf.float32)
      #delta square
      square = tf.math.square(sub)
      #reduce sum
      var = tf.reduce_mean(square, axis=axis)
      #add small epsilon
      var = tf.math.add(var, epsilon)
      #not using for testing
      #var_sqrt = tf.sqrt(var)
      #isqrt = tf.math.divide(1.0, var_sqrt)
      isqrt = Q_rsqrt_float(var)
      isqrt = tf.cast(tf.cast(isqrt, tf.bfloat16), tf.float32)
      isqrt = tf.expand_dims(isqrt, -1)
      mul = tf.multiply(sub,isqrt)
      mul = tf.cast(tf.cast(mul,tf.bfloat16),tf.float32)

      #gamma * input + beta
      gamma_res = tf.multiply(gamma, mul)
      ln_res = tf.add(gamma_res, beta)

      # below is quantize tensor used, for round method and pow
      #the quantize process
      #x_out = tf.pow(x_out, shift_up)
      #use round_method 3 in json
      #x_out = tf.math.floor(x_out)
      #use narrow range false
      #x_out = tf.clip_by_value(x_out, -128, 127)
      return ln_res
    return hard_layernorm_dpu(self.axis, self.epsilon, self.gamma, self.beta, inputs)

  def get_config(self):
    base_config = super(VitisLayernorm, self).get_config()
    config = {
        'axis': self.axis,
        'epsilon': self.epsilon
    }
    return dict(list(base_config.items()) + list(config.items()))


  @classmethod
  def from_config(cls, config):
    config = config.copy()
    axis = config.pop('axis')
    epsilon = config.pop('epsilon')
    return cls(
        axis=axis,
        epsilon=epsilon,
        **config)



def _types_dict():
  return {
      'VitisLayernorm': VitisLayernorm,
  }
