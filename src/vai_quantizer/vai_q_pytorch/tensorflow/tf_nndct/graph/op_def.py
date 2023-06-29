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

class TFDense(ops.Operation, base_op.Dense):

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

class TFBatchNorm(ops.Operation, base_op.BatchNorm):

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

class TFConv2D(ops.Operation, base_op.Conv2d):

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

class TFSeparableConv2D(ops.Operation, base_op.SeparableConv2D):

  @unique
  class AttrName(ops.AutoName):
    KERNEL = auto()
    STRIDE = auto()
    DILATION = auto()
    PAD_MODE = auto()
    PAD = auto()
    GROUP = auto()
    DEPTH_MULTIPLIER = auto()
    BIAS_TERM = auto()
    IN_DIM = auto()
    OUT_DIM = auto()
    ACTIVATION = auto()

  @unique
  class ParamName(ops.AutoName):
    DEPTHWISE_WEIGHT = 'depthwise_kernel'
    POINTWISE_WEIGHT = 'pointwise_kernel'
    BIAS = 'bias'

  def __init__(self, *args, **kwargs):
    super(TFSeparableConv2D, self).__init__(OpTypes.SEPARABLECONV2D, *args,
                                            **kwargs)
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
  def depth_multiplier(self):
    return self.attr['depth_multiplier']

  @depth_multiplier.setter
  def depth_multiplier(self, value):
    self.attr['depth_multiplier'] = value

  @property
  def use_bias(self):
    return self.attr['bias_term']

  @use_bias.setter
  def use_bias(self, value):
    self.attr['bias_term'] = value

class TFConv2DTranspose(ops.Operation, base_op.Conv2d):
  # as defined in /tensorflow/python/keras/layers/convolutional.py
  # Conv2DTranspose is a sub_class of Conv2D
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
    super(TFConv2DTranspose, self).__init__(OpTypes.CONVTRANSPOSE2D, *args,
                                            **kwargs)
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

class TFConv3D(ops.Operation, base_op.Conv3d):

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
    OUTPUT_PAD = auto()
    ACTIVATION = auto()

  @unique
  class ParamName(ops.AutoName):
    WEIGHTS = 'kernel'
    BIAS = 'bias'

  def __init__(self, *args, **kwargs):
    super(TFConv3D, self).__init__(OpTypes.CONV3D, *args, **kwargs)
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
    return self.attr['pad_mode']

  @padding.setter
  def padding(self, value):
    # mode = 1 if value.lower() == 'same' else 2
    self.attr['pad_mode'] = value

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

  # no outpadding in original tf conv

class TFConv3DTranspose(ops.Operation, base_op.Conv3d):

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
    OUTPUT_PAD = auto()
    ACTIVATION = auto()

  @unique
  class ParamName(ops.AutoName):
    WEIGHTS = 'kernel'
    BIAS = 'bias'

  def __init__(self, *args, **kwargs):
    super(TFConv3DTranspose, self).__init__(OpTypes.CONVTRANSPOSE3D, *args,
                                            **kwargs)
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
    return self.attr['pad_mode']

  @padding.setter
  def padding(self, value):
    # mode = 1 if value.lower() == 'same' else 2
    self.attr['pad_mode'] = value

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

class TFDepthwiseConv2D(TFConv2D):

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
    DEPTH_MULTIPLIER = auto()

  @unique
  class ParamName(ops.AutoName):
    WEIGHTS = 'depthwise_kernel'
    BIAS = 'bias'

  def __init__(self, *args, **kwargs):
    super(TFDepthwiseConv2D, self).__init__(*args, **kwargs)
    self.set_optype(OpTypes.DEPTHWISE_CONV2D)
    self._define_attr(self.AttrName.DEPTH_MULTIPLIER, value_type=int, size=1)

  @property
  def depth_multiplier(self):
    return self.attr['depth_multiplier']

  @depth_multiplier.setter
  def depth_multiplier(self, value):
    self.attr['depth_multiplier'] = value

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

class TFMultiply(ops.Operation, base_op.BinaryOp):

  def __init__(self, *args, **kwargs):
    super(TFMultiply, self).__init__(OpTypes.MULTIPLY, *args, **kwargs)

class TFConst(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFConst, self).__init__(OpTypes.CONST, *args, **kwargs)

class TFCast(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFCast, self).__init__(OpTypes.CAST, *args, **kwargs)

class TFCast(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFCast, self).__init__(OpTypes.CAST, *args, **kwargs)

class TFAdd(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFAdd, self).__init__(OpTypes.ADD, *args, **kwargs)

class TFRescaling(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFRescaling, self).__init__(OpTypes.RESCALING, *args, **kwargs)

class TFNormalization(ops.Operation):

  @unique
  class ParamName(ops.AutoName):
    MEAN = 'mean'
    VARIANCE = 'variance'
    COUNT = 'count'

  def __init__(self, *args, **kwargs):
    super(TFNormalization, self).__init__(OpTypes.NORM, *args, **kwargs)

class TFMultiplyLayer(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFMultiplyLayer, self).__init__(OpTypes.MULTIPLYLAYER, *args,
                                          **kwargs)

class TFSubtract(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFSubtract, self).__init__(OpTypes.SUB, *args, **kwargs)

class TFBiasAdd(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFBiasAdd, self).__init__(OpTypes.BIAS_ADD, *args, **kwargs)

class TFIdentity(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFIdentity, self).__init__(OpTypes.IDENTITY, *args, **kwargs)

class TFNoOp(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFNoOp, self).__init__(OpTypes.NOOP, *args, **kwargs)

class TFSigmoid(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFSigmoid, self).__init__(OpTypes.SIGMOID, *args, **kwargs)

class TFTanh(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFTanh, self).__init__(OpTypes.TANH, *args, **kwargs)

class TFSwish(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFSwish, self).__init__(OpTypes.SWISH, *args, **kwargs)

class TFElu(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFElu, self).__init__(OpTypes.ELU, *args, **kwargs)

class TFExponential(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFExponential, self).__init__(OpTypes.EXPONENTIAL, *args, **kwargs)

class TFGelu(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFGelu, self).__init__(OpTypes.GELU, *args, **kwargs)

class TFHardSigmoid(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFHardSigmoid, self).__init__(OpTypes.HSIGMOID, *args, **kwargs)

class TFSelu(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFSelu, self).__init__(OpTypes.SELU, *args, **kwargs)

class TFSoftPlus(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFSoftPlus, self).__init__(OpTypes.SOFTPLUS, *args, **kwargs)

class TFSoftSign(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFSoftSign, self).__init__(OpTypes.SOFTSIGN, *args, **kwargs)

class TFGather(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFGather, self).__init__(OpTypes.GATHER, *args, **kwargs)

class TFRFFT(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFRFFT, self).__init__(OpTypes.RFFT, *args, **kwargs)

class TFComplexAbs(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFComplexAbs, self).__init__(OpTypes.COMPLEX_ABS, *args, **kwargs)

class TFAngle(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFAngle, self).__init__(OpTypes.ANGLE, *args, **kwargs)

class TFExp(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFExp, self).__init__(OpTypes.EXP, *args, **kwargs)

class TFIRFFT(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFIRFFT, self).__init__(OpTypes.IRFFT, *args, **kwargs)

class TFPad(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFPad, self).__init__(OpTypes.PAD, *args, **kwargs)

class TFTranspose(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFTranspose, self).__init__(OpTypes.TRANSPOSE, *args, **kwargs)

class TFSum(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFSum, self).__init__(OpTypes.SUM, *args, **kwargs)

class TFReshape(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFReshape, self).__init__(OpTypes.RESHAPE, *args, **kwargs)

class TFRelu(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFRelu, self).__init__(OpTypes.RELU, *args, **kwargs)

class TFSoftmax(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFSoftmax, self).__init__(OpTypes.SOFTMAX, *args, **kwargs)

class TFStridedSlice(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFStridedSlice, self).__init__(OpTypes.STRIDED_SLICE, *args, **kwargs)

class TFConcat(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFConcat, self).__init__(OpTypes.CONCAT, *args, **kwargs)
