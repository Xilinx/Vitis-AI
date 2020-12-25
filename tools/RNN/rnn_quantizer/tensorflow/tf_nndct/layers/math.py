

#
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
#

import tensorflow as tf

from tensorflow.python.keras.layers import core
from tensorflow.python.ops import math_ops

from tf_nndct.graph import OpTypes
from tf_nndct.quantization.utils import QuantizedModule
from tf_nndct.quantization.ops import scaleop
from tf_nndct.quantization.ops import table_lookup
from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.utils import NndctOption

from .sigmoid_table import *
from .tanh_table import *

@QuantizedModule(OpTypes.DENSE)
class Dense(core.Dense):
  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super().__init__(units,
                     activation=activation,
                     use_bias=use_bias,
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer,
                     kernel_regularizer=kernel_regularizer,
                     bias_regularizer=bias_regularizer,
                     activity_regularizer=activity_regularizer,
                     kernel_constraint=kernel_constraint,
                     bias_constraint=bias_constraint,
                     **kwargs)

    self.quant_mode, self.quantizer = maybe_get_quantizer()
    if self.quant_mode > 0:
      self._quant_vars = {}
    self.valid_inputs = None
    self.valid_output = None
    self.quant_info = None
    self.params_name = None
    self.node = None
    self.quant_vars_initialized = False
    self.params_quantized = False

  def build(self, input_shape):
    super().build(input_shape)

  def _quant_vars_init(self):
    if self.valid_inputs is not None:
      self._quant_vars['input_pos'] = self.quantizer.create_fp_tensor(
          name='input_pos',
          fp_name=self.valid_inputs[0])
      #self._quant_vars['input_pos_stat'] = self.quantizer.create_fp_stat_tensor()
    self._quant_vars['kernel_pos'] = self.quantizer.create_fp_tensor(
        name='kernel_pos',
        fp_name=self.params_name[0],
        tensor_type='param')
    if self.use_bias:
      self._quant_vars['bias_pos'] = self.quantizer.create_fp_tensor(
          name='bias_pos',
          fp_name=self.params_name[1],
          tensor_type='param')
    self._quant_vars['output_pos'] = self.quantizer.create_fp_tensor(
        name='output_pos',
        fp_name=self.valid_output[0])
    #self._quant_vars['output_pos_stat'] = self.quantizer.create_fp_stat_tensor()

  @property
  def quant_results(self):
    res = []
    # input
    if self.valid_inputs is not None:
      res.append((self.valid_inputs[0],
                 [self.quantizer.bita,
                 int(self._quant_vars['input_pos'].numpy())]))
    # kernel
    res.append((self.params_name[0],
               [self.quantizer.bitw,
               int(self._quant_vars['kernel_pos'].numpy())]))
    # bias
    if self.use_bias:
      res.append((self.params_name[1],
                 [self.quantizer.bitw,
                 int(self._quant_vars['bias_pos'].numpy())]))
    # output
    res.append((self.valid_output[0],
               [self.quantizer.bita,
               int(self._quant_vars['output_pos'].numpy())]))

    return res

  def call(self, x):
    if self.valid_output is not None and not self.quant_vars_initialized:
      self._quant_vars_init()
      self.quant_vars_initialized = True

    if self.params_name is not None and not self.params_quantized:
      self.params_quantized = True
      self.kernel = self.quantizer.get_fp_and_quantize(
                        self.kernel,
                        self.params_name[0],
                        self._quant_vars['kernel_pos'],
                        node=self.node,
                        tensor_type='param')

      if self.use_bias:
        self.bias = self.quantizer.get_fp_and_quantize(
                        self.bias,
                        self.params_name[1],
                        self._quant_vars['bias_pos'],
                        node=self.node,
                        tensor_type='param')

    if self.valid_inputs is not None:
      x = self.quantizer.get_fp_and_quantize(
              x,
              self.valid_inputs[0],
              self._quant_vars['input_pos'],
              tensor_type='output')

    if self.quant_mode == 2 and self.quantizer.is_lstm:
      bakFlag = self.use_bias
      self.use_bias = False
      output = super().call(x)
      if self.valid_output is not None:
        output = self.quantizer.get_fp_and_quantize(
                     output,
                     self.valid_output[0],
                     self._quant_vars['output_pos'],
                     tensor_type='output')
      self.use_bias = bakFlag
      if self.use_bias:
        output = tf.math.add(output, self.bias)
    else:
      output = super().call(x)

    if self.valid_output is not None:
      output = self.quantizer.get_fp_and_quantize(
                   output,
                   self.valid_output[0],
                   self._quant_vars['output_pos'],
                   tensor_type='output')

    return output

@QuantizedModule(OpTypes.ADD)
class Add(tf.Module):

  def __init__(self):
    super().__init__()
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    if self.quant_mode > 0:
      self._quant_vars = {}
    self.valid_inputs = None
    self.valid_output = None
    self.quant_info = None
    self.params_name = None
    self.node = None
    self.quant_vars_initialized = False

  def _quant_vars_init(self):
    if self.valid_inputs is not None:
      self._quant_vars['input_pos'] = self.quantizer.create_fp_tensor(
          name='input_pos',
          fp_name=self.valid_inputs[0])
    self._quant_vars['output_pos'] = self.quantizer.create_fp_tensor(
        name='output_pos',
        fp_name=self.valid_output[0])

  def __call__(self, x, y):
    if self.valid_output is not None and not self.quant_vars_initialized:
      self._quant_vars_init()
      self.quant_vars_initialized = True

    if self.valid_inputs is not None:
      if self.node.in_nodes[0] == self.valid_inputs[0]:
        x = self.quantizer.get_fp_and_quantize(
                x,
                self.valid_inputs[0],
                self._quant_vars['input_pos'],
                tensor_type='output')
      else:
        y = self.quantizer.get_fp_and_quantize(
                y,
                self.valid_inputs[0],
                self._quant_vars['input_pos'],
                tensor_type='output')
    output = tf.math.add(x, y)
    if self.valid_output is not None:
      output = self.quantizer.get_fp_and_quantize(
                   output,
                   self.valid_output[0],
                   self._quant_vars['output_pos'],
                   tensor_type='output')
    return output


@QuantizedModule(OpTypes.MULTIPLY)
class Multiply(tf.Module):

  def __init__(self):
    super().__init__()
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    if self.quant_mode > 0:
      self._quant_vars = {}
    self.valid_inputs = None
    self.valid_output = None
    self.quant_info = None
    self.params_name = None
    self.node = None
    self.quant_vars_initialized = False

  def _quant_vars_init(self):
    if self.valid_inputs is not None:
      self._quant_vars['input_pos'] = self.quantizer.create_fp_tensor(
          name='input_pos',
          fp_name=self.valid_inputs[0])
    self._quant_vars['output_pos'] = self.quantizer.create_fp_tensor(
        name='output_pos',
        fp_name=self.valid_output[0])

  def __call__(self, x, y):
    if self.valid_output is not None and not self.quant_vars_initialized:
      self._quant_vars_init()
      self.quant_vars_initialized = True

    if self.valid_inputs is not None:
      if self.node.in_nodes[0] == self.valid_inputs[0]:
        x = self.quantizer.get_fp_and_quantize(
                x,
                self.valid_inputs[0],
                self._quant_vars['input_pos'],
                tensor_type='output')
      else:
        y = self.quantizer.get_fp_and_quantize(
                y,
                self.valid_inputs[0],
                self._quant_vars['input_pos'],
                tensor_type='output')
    output = math_ops.multiply(x, y)
    if self.valid_output is not None:
      output = self.quantizer.get_fp_and_quantize(
                   output,
                   self.valid_output[0],
                   self._quant_vars['output_pos'],
                   tensor_type='output')
    return output

SIGMOID_TABLE = nndct_tf_sigmoid_table()

@QuantizedModule(OpTypes.SIGMOID)
class Sigmoid(tf.Module):

  def __init__(self):
    super().__init__()
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    if self.quant_mode > 0:
      self._quant_vars = {}
    self.valid_inputs = None
    self.valid_output = None
    self.quant_info = None
    self.params_name = None
    self.node = None

  def __call__(self, x):
    if NndctOption.nndct_quant_off.value:
      output = math_ops.sigmoid(x)
    elif self.quant_mode > 0 and self.node is not None:
      input_name = self.node.in_nodes[0]
      fragpos = self.quantizer.get_bnfp(input_name, False)[1]
      output = table_lookup(x, SIGMOID_TABLE.table, fragpos, type=0)
    else:
      output = math_ops.sigmoid(x)

    return output


TANH_TABLE = nndct_tf_tanh_table()

@QuantizedModule(OpTypes.TANH)
class Tanh(tf.Module):

  def __init__(self):
    super().__init__()
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    if self.quant_mode > 0:
      self._quant_vars = {}
    self.valid_inputs = None
    self.valid_output = None
    self.quant_info = None
    self.params_name = None
    self.node = None

  def __call__(self, x):

    if NndctOption.nndct_quant_off.value:
      output = math_ops.tanh(x)
    elif self.quant_mode > 0 and self.node is not None:
      input_name = self.node.in_nodes[0]
      fragpos = self.quantizer.get_bnfp(input_name, False)[1]
      output = table_lookup(x, TANH_TABLE.table, fragpos, type=1)
    else:
      output = math_ops.tanh(x)

    return output
