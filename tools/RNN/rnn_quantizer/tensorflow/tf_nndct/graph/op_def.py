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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum, unique, auto

from tf_nndct.graph import OpTypes
from tf_nndct.graph import base_op
from tf_nndct.graph import dtypes
from tf_nndct.graph import ops

class TFGeneric(ops.Operation):
  """A generic op that can represent any keras layer."""

  @unique
  class AttrName(base_op.AutoName):
    LAYER_CLASS = auto()

  def __init__(self, *args, **kwargs):
    super(TFGeneric, self).__init__(OpTypes.GENERIC, *args, **kwargs)
    self._define_attr(self.AttrName.LAYER_CLASS, value_type=type, size=1)

#TODO(yuwang): Use _define_attr to define attr.
class TFInput(ops.Operation):

  @unique
  class AttrName(ops.AutoName):
    SHAPE = auto()
    DTYPE = auto()

  def __init__(self, *args, **kwargs):
    super(TFInput, self).__init__(OpTypes.INPUT, *args, **kwargs)
    self._attr_value_mem = {self.AttrName.SHAPE: [], self.AttrName.DTYPE: []}

    self._attrs[self.AttrName.SHAPE] = ops.Attr(
        name=self.AttrName.SHAPE,
        value_type=int,
        size=None,
        value_mem=self._attr_value_mem[self.AttrName.SHAPE],
        occurence_type=ops.OccurenceType.REQUIRED,
    )
    self._attrs[self.AttrName.DTYPE] = ops.Attr(
        name=self.AttrName.DTYPE,
        value_type=dtypes.DType,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.DTYPE],
        occurence_type=ops.OccurenceType.REQUIRED,
    )

  @property
  def shape(self):
    return self.attr['shape']

  @shape.setter
  def shape(self, value):
    self.attr['shape'] = value

  @property
  def dtype(self):
    return self.attr['dtype']

  @dtype.setter
  def dtype(self, value):
    self.attr['dtype'] = value

class TFDense(base_op.Dense):

  @unique
  class AttrName(ops.AutoName):
    BIAS_TERM = auto()
    IN_DIM = auto()
    OUT_DIM = auto()
    ACTIVATION = auto()

  @unique
  class ParamName(ops.AutoName):
    WEIGHTS = 'kernel'
    BIAS = 'bias'

  def __init__(self, *args, **kwargs):
    super(TFDense, self).__init__(OpTypes.DENSE, *args, **kwargs)
    self._define_attr(self.AttrName.ACTIVATION, value_type=str, size=1)

  @property
  def use_bias(self):
    return self.attr['bias_term']

  @use_bias.setter
  def use_bias(self, value):
    self.attr['bias_term'] = value

  @property
  def units(self):
    return self.attr['out_dim']

  @units.setter
  def units(self, value):
    self.attr['out_dim'] = value

class TFBatchNorm(base_op.BatchNorm):

  @unique
  class ParamName(base_op.AutoName):
    GAMMA = auto()
    BETA = auto()
    MOVING_MEAN = auto()
    MOVING_VAR = 'moving_variance'

  def __init__(self, *args, **kwargs):
    super(TFBatchNorm, self).__init__(OpTypes.BATCH_NORM, *args, **kwargs)

  @property
  def epsilon(self):
    return self.attr['epsilon']

  @epsilon.setter
  def epsilon(self, value):
    self.attr['epsilon'] = value

  @property
  def scale(self):
    return self.attr['scale']

  @scale.setter
  def scale(self, value):
    self.attr['scale'] = value

  @property
  def center(self):
    return self.attr['center']

  @center.setter
  def center(self, value):
    self.attr['center'] = value

  @property
  def axis(self):
    return self.attr['axis']

  @axis.setter
  def axis(self, value):
    self.attr['axis'] = value

class TFConv2D(base_op.Conv2d):

  @unique
  class AttrName(ops.AutoName):
    KERNEL = auto()
    STRIDE = auto()
    DILATION = auto()
    PAD_MODE = auto()
    PAD = auto()
    GROUP = auto()
    BIAS_TERM = auto()
    IN_DIM = auto()
    OUT_DIM = auto()
    ACTIVATION = auto()

  @unique
  class ParamName(ops.AutoName):
    WEIGHTS = 'kernel'
    BIAS = 'bias'

  def __init__(self, *args, **kwargs):
    super(TFConv2D, self).__init__(OpTypes.CONV2D, *args, **kwargs)
    self._define_attr(self.AttrName.ACTIVATION, value_type=str, size=1)

  @property
  def filters(self):
    return self.attr['out_dim']

  @filters.setter
  def filters(self, value):
    self.attr['out_dim'] = value

  @property
  def kernel_size(self):
    return self.attr['kernel']

  @kernel_size.setter
  def kernel_size(self, value):
    self.attr['kernel'] = value

  @property
  def strides(self):
    return self.attr['stride']

  @strides.setter
  def strides(self, value):
    self.attr['stride'] = value

  @property
  def padding(self):
    mode = self.attr['pad_mode']
    return 'same' if mode == 1 else 'valid'

  @padding.setter
  def padding(self, value):
    mode = 1 if value.lower() == 'same' else 2
    self.attr['pad_mode'] = mode

  @property
  def dilation_rate(self):
    return self.attr['dilation']

  @dilation_rate.setter
  def dilation_rate(self, value):
    self.attr['dilation'] = value

  @property
  def groups(self):
    return self.attr['group']

  @groups.setter
  def groups(self, value):
    self.attr['group'] = value

  @property
  def use_bias(self):
    return self.attr['bias_term']

  @use_bias.setter
  def use_bias(self, value):
    self.attr['bias_term'] = value

class TFEmbedding(ops.Operation):

  @unique
  class ParamName(ops.AutoName):
    EMBEDDINGS = 'embeddings'

  def __init__(self, *args, **kwargs):
    super(TFEmbedding, self).__init__(OpTypes.EMBEDDING, *args, **kwargs)

class TFRNNLayer(ops.Operation):

  @unique
  class AttrName(ops.AutoName):
    LAYER_CLASS = auto()

  def __init__(self, *args, **kwargs):
    super(TFRNNLayer, self).__init__(OpTypes.RNN_LAYER, *args, **kwargs)
    self._define_attr(self.AttrName.LAYER_CLASS, value_type=type, size=1)

class TFRNN(ops.Operation):

  @unique
  class AttrName(ops.AutoName):
    CELL = auto()

  def __init__(self, *args, **kwargs):
    super(TFRNN, self).__init__(OpTypes.RNN, *args, **kwargs)
    self._define_attr(self.AttrName.CELL, value_type=ops.Operation, size=1)

class TFStackedRNNCells(ops.Operation):

  @unique
  class AttrName(ops.AutoName):
    CELLS = auto()

  def __init__(self, *args, **kwargs):
    super(TFStackedRNNCells, self).__init__(OpTypes.STACKED_RNN_CELLS, *args,
                                            **kwargs)

    self._define_attr(self.AttrName.CELLS, value_type=ops.Operation, size=None)

class TFSimpleRNN(ops.Operation):

  @unique
  class ParamName(ops.AutoName):
    KERNEL = 'kernel'
    RECURRENT_KERNEL = 'recurrent_kernel'
    BIAS = 'bias'

  def __init__(self, *args, **kwargs):
    super(TFSimpleRNN, self).__init__(OpTypes.SIMPLE_RNN, *args, **kwargs)

class TFLSTM(ops.Operation):

  @unique
  class ParamName(ops.AutoName):
    KERNEL = 'kernel'
    RECURRENT_KERNEL = 'recurrent_kernel'
    BIAS = 'bias'

  def __init__(self, *args, **kwargs):
    super(TFLSTM, self).__init__(OpTypes.LSTM, *args, **kwargs)

class TFGRU(ops.Operation):

  @unique
  class ParamName(ops.AutoName):
    KERNEL = 'kernel'
    RECURRENT_KERNEL = 'recurrent_kernel'
    BIAS = 'bias'

  def __init__(self, *args, **kwargs):
    super(TFGRU, self).__init__(OpTypes.GRU, *args, **kwargs)

class TFBidirectional(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFBidirectional, self).__init__(OpTypes.BIDIRECTIONAL_RNN, *args,
                                          **kwargs)

class TFMultiply(base_op.BinaryOp):

  def __init__(self, *args, **kwargs):
    super(TFMultiply, self).__init__(OpTypes.MULTIPLY, *args, **kwargs)
