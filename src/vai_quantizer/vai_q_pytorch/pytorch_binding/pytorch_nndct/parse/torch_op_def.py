

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

from enum import auto, unique
from typing import Dict, List, Callable, Optional, Union, Any, Set

import pytorch_nndct.utils as utils
from nndct_shared.base import NNDCT_CONSTANT, NNDCT_OP
from nndct_shared.nndct_graph import Operation
from nndct_shared.nndct_graph import operator_definition as base_op
from nndct_shared.nndct_graph import Tensor
from nndct_shared.utils import transformed_axis, DataFormat, AutoName

class TorchFlatten(base_op.Flatten):

  def __init__(self, *args, **kwargs):
    super(TorchFlatten, self).__init__(NNDCT_OP.FLATTEN, *args, **kwargs)
    utils.op_register(NNDCT_OP.FLATTEN, 'flatten')

  @property
  def start_dim(self):
    return self._attr_value_mem[self.AttrName.START_DIM][0]

  @start_dim.setter
  def start_dim(self, value):
    self._attr_value_mem[self.AttrName.START_DIM][:] = [value]

  @property
  def end_dim(self):
    return self._attr_value_mem[self.AttrName.END_DIM][0]

  @end_dim.setter
  def end_dim(self, value):
    self._attr_value_mem[self.AttrName.END_DIM][:] = [value]


class TorchAdd(base_op.BinaryOp):
  # TODO: Change class Operation to base_op.BinaryOp
  def __init__(self, *args, **kwargs):
    super(TorchAdd, self).__init__(NNDCT_OP.ADD, *args, **kwargs)
    utils.op_register(NNDCT_OP.ADD, 'add')

  @property
  def input(self):
    return self.get_attr(self.AttrName.INPUT)

  @input.setter
  def input(self, input):
    self.set_attr(self.AttrName.INPUT, input)

  @property
  def other(self):
    return self.get_attr(self.AttrName.OTHER)

  @other.setter
  def other(self, other):
    self.set_attr(self.AttrName.OTHER, other)
    
class TorchReLU(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchReLU, self).__init__(NNDCT_OP.RELU, *args, **kwargs)
    utils.op_register(NNDCT_OP.RELU, 'ReLU')


class TorchLeakyReLU(base_op.LeakyReLU):

  def __init__(self):
    super().__init__()
    utils.op_register(NNDCT_OP.LEAKY_RELU, 'LeakyReLU')

  @property
  def negative_slope(self):
    return self.get_attr(self.AttrName.ALPHA)

  @negative_slope.setter
  def negative_slope(self, value):
    self.set_attr(self.AttrName.ALPHA, value)


class TorchPReLU(base_op.PReLU):
  
  def __init__(self, *args, **kwargs):
    super(TorchPReLU, self).__init__(*args, **kwargs)
    utils.op_register(NNDCT_OP.PRELU, 'PReLU')

  @property
  def num_parameters(self):
    return self.get_attr(self.AttrName.NUM_PARAMETERS)

  @num_parameters.setter
  def num_parameters(self, value):
    self.set_attr(self.AttrName.NUM_PARAMETERS, value)


class TorchGELU(base_op.GELU):

  def __init__(self, *args, **kwargs):
    super(TorchGELU, self).__init__(*args, **kwargs)
    utils.op_register(NNDCT_OP.GELU, 'GELU')

  @property
  def approximate(self):
    return self.get_attr(self.AttrName.APPROXIMATE)

  @approximate.setter
  def approximate(self, value):
    self.set_attr(self.AttrName.APPROXIMATE, value)


class TorchMish(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchMish, self).__init__(NNDCT_OP.MISH, *args, **kwargs)
    utils.op_register(NNDCT_OP.MISH, 'Mish')


class TorchTanh(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchTanh, self).__init__(NNDCT_OP.TANH, *args, **kwargs)
    utils.op_register(NNDCT_OP.TANH, 'Tanh')


class TorchHardTanh(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchHardTanh, self).__init__(NNDCT_OP.HARDTANH, *args, **kwargs)
    utils.op_register(NNDCT_OP.HARDTANH, 'Hardtanh')


class TorchInput(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchInput, self).__init__(NNDCT_OP.INPUT, *args, **kwargs)
    utils.op_register(NNDCT_OP.INPUT, NNDCT_OP.INPUT)


class TorchReturn(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchReturn, self).__init__(NNDCT_OP.RETURN, *args, **kwargs)
    utils.op_register(NNDCT_OP.RETURN, NNDCT_OP.RETURN)

class TorchLinear(base_op.Dense):

  @unique
  class ParamName(AutoName):
    WEIGHTS = "weight"
    BIAS = auto()

  def __init__(self, *args, **kwargs):
    super(TorchLinear, self).__init__(NNDCT_OP.DENSE, *args, **kwargs)
    utils.op_register(NNDCT_OP.DENSE, 'Linear')

  @property
  def bias(self):
    return self._attr_value_mem[self.AttrName.BIAS_TERM][0]

  @bias.setter
  def bias(self, value):
    self._attr_value_mem[self.AttrName.BIAS_TERM][:] = [bool(value)]

  @property
  def in_features(self):
    return self._attr_value_mem[self.AttrName.IN_DIM][0]

  @in_features.setter
  def in_features(self, value):
    self._attr_value_mem[self.AttrName.IN_DIM][:] = [value]

  @property
  def out_features(self):
    return self._attr_value_mem[self.AttrName.OUT_DIM][0]

  @out_features.setter
  def out_features(self, value):
    self._attr_value_mem[self.AttrName.OUT_DIM][:] = [value]


class TorchBatchNorm(base_op.BatchNorm):

  @unique
  class ParamName(base_op.AutoName):
    GAMMA = "weight"
    BETA = "bias"
    MOVING_MEAN = "mean"
    MOVING_VAR = "var"

  def __init__(self):
    super().__init__(NNDCT_OP.BATCH_NORM)
    utils.op_register(NNDCT_OP.BATCH_NORM, "BatchNorm", class_type=utils.TorchOpClassType.NN_MODULE)

  @property
  def eps(self):
    return self._attr_value_mem[self.AttrName.EPSILON][0]

  @eps.setter
  def eps(self, value):
    self._attr_value_mem[self.AttrName.EPSILON][:] = [value]

  @property
  def num_features(self):
    return self._attr_value_mem[self.AttrName.OUT_DIM][0]

  @num_features.setter
  def num_features(self, value):
    self._attr_value_mem[self.AttrName.OUT_DIM][:] = [value]


class TorchInstanceNorm(base_op.InstanceNorm):

  @unique
  class ParamName(base_op.AutoName):
    GAMMA = "weight"
    BETA = "bias"

  def __init__(self):
    super().__init__(NNDCT_OP.INSTANCE_NORM)
    utils.op_register(NNDCT_OP.INSTANCE_NORM, "InstanceNorm", class_type=utils.TorchOpClassType.NN_MODULE)

  @property
  def eps(self):
    return self.get_attr(self.AttrName.EPS)

  @eps.setter
  def eps(self, value):
    self.set_attr(self.AttrName.EPS, value)

  @property
  def num_features(self):
    return self.get_attr(self.AttrName.NUM_FEATURES)

  @num_features.setter
  def num_features(self, value):
    self.set_attr(self.AttrName.NUM_FEATURES, value)

  @property
  def affine(self):
    return self.get_attr(self.AttrName.AFFINE)

  @affine.setter
  def affine(self, value):
    self.set_attr(self.AttrName.AFFINE, value)


class TorchGroupNorm(base_op.GroupNorm):

  @unique
  class ParamName(base_op.AutoName):
    GAMMA = "weight"
    BETA = "bias"

  def __init__(self):
    super().__init__(NNDCT_OP.GROUP_NORM)
    utils.op_register(NNDCT_OP.GROUP_NORM, "GroupNorm")

  @property
  def eps(self):
    return self.get_attr(self.AttrName.EPS)

  @eps.setter
  def eps(self, value):
    self.set_attr(self.AttrName.EPS, value)

  @property
  def num_groups(self):
    return self.get_attr(self.AttrName.NUM_GROUPS)

  @num_groups.setter
  def num_groups(self, value):
    self.set_attr(self.AttrName.NUM_GROUPS, value)

  @property
  def affine(self):
    return self.get_attr(self.AttrName.AFFINE)

  @affine.setter
  def affine(self, value):
    self.set_attr(self.AttrName.AFFINE, value)

  @property
  def num_channels(self):
    return self.get_attr(self.AttrName.NUM_CHANNELS)

  @num_channels.setter
  def num_channels(self, value):
    self.set_attr(self.AttrName.NUM_CHANNELS, value)


class _TorchConv1d(base_op.Conv1d):

  def __init__(self, op_type, *args, **kwargs):
    super().__init__(op_type, *args, **kwargs)

  @property
  def kernel_size(self):
    return self.get_attr(self.AttrName.KERNEL)

  @kernel_size.setter
  def kernel_size(self, value):
    self.set_attr(self.AttrName.KERNEL, value)

  @property
  def dilation(self):
    return self.get_attr(self.AttrName.DILATION)

  @dilation.setter
  def dilation(self, value):
   self.set_attr(self.AttrName.DILATION, value)

  @property
  def padding(self):
    return [self.get_attr(self.AttrName.PAD)[0]]

  @padding.setter
  def padding(self, value):
    
    self.set_attr(self.AttrName.PAD, [value[0], value[0]])
    self.set_attr(self.AttrName.PAD_MODE, 'FLOOR')

  @property
  def stride(self):
    return self.get_attr(self.AttrName.STRIDE)

  @stride.setter
  def stride(self, value):
    self.set_attr(self.AttrName.STRIDE, value)

  @property
  def in_channels(self):
    return self.get_attr(self.AttrName.IN_DIM)

  @in_channels.setter
  def in_channels(self, value):
    self.set_attr(self.AttrName.IN_DIM, value)

  @property
  def out_channels(self):
    return self.get_attr(self.AttrName.OUT_DIM)

  @out_channels.setter
  def out_channels(self, value):
    self.set_attr(self.AttrName.OUT_DIM, value)

  @property
  def groups(self):
    return self.get_attr(self.AttrName.GROUP)

  @groups.setter
  def groups(self, value):
    self.set_attr(self.AttrName.GROUP, value)

  @property
  def bias(self):
    return self.get_attr(self.AttrName.BIAS_TERM)

  @bias.setter
  def bias(self, value):
    self.set_attr(self.AttrName.BIAS_TERM, bool(value))


class TorchConv1d(_TorchConv1d):

  def __init__(self, nndct_op_type, *args, **kwargs):
    super().__init__(nndct_op_type, *args, **kwargs)
    utils.op_register(nndct_op_type, "Conv1d")


class TorchConvTranspose1d(_TorchConv1d):

  def __init__(self, nndct_op_type, *args, **kwargs):
    super().__init__(nndct_op_type, *args, **kwargs)
    utils.op_register(nndct_op_type, "ConvTranspose1d")


class _TorchConv2d(base_op.Conv2d):

  @unique
  class ParamName(base_op.AutoName):
    WEIGHTS = "weight"
    BIAS = auto()

  def __init__(self, op_type, *args, **kwargs):
    super(_TorchConv2d, self).__init__(op_type, *args, **kwargs)

  @property
  def kernel_size(self):
    return self._attr_value_mem[self.AttrName.KERNEL][::-1]

  @kernel_size.setter
  def kernel_size(self, value):
    self._attr_value_mem[self.AttrName.KERNEL][:] = value[::-1]

  @property
  def dilation(self):
    return self._attr_value_mem[self.AttrName.DILATION][::-1]

  @dilation.setter
  def dilation(self, value):
    self._attr_value_mem[self.AttrName.DILATION][:] = value[::-1]

  @property
  def padding(self):
    return [
        self._attr_value_mem[self.AttrName.PAD][2],
        self._attr_value_mem[self.AttrName.PAD][0]
    ]

  @padding.setter
  def padding(self, value):
    self._attr_value_mem[self.AttrName.PAD_MODE][:] = [0]
    self._attr_value_mem[self.AttrName.PAD][:] = [
        value[1], value[1], value[0], value[0]
    ]

  @property
  def stride(self):
    return self._attr_value_mem[self.AttrName.STRIDE][::-1]

  @stride.setter
  def stride(self, value):
    self._attr_value_mem[self.AttrName.STRIDE][:] = value[::-1]

  @property
  def in_channels(self):
    return self._attr_value_mem[self.AttrName.IN_DIM][0]

  @in_channels.setter
  def in_channels(self, value):
    self._attr_value_mem[self.AttrName.IN_DIM][:] = [value]

  @property
  def out_channels(self):
    return self._attr_value_mem[self.AttrName.OUT_DIM][0]

  @out_channels.setter
  def out_channels(self, value):
    self._attr_value_mem[self.AttrName.OUT_DIM][:] = [value]

  @property
  def groups(self):
    return self._attr_value_mem[self.AttrName.GROUP][0]

  @groups.setter
  def groups(self, value):
    self._attr_value_mem[self.AttrName.GROUP][:] = [value]

  @property
  def bias(self):
    return self._attr_value_mem[self.AttrName.BIAS_TERM][0]

  @bias.setter
  def bias(self, value):
    self._attr_value_mem[self.AttrName.BIAS_TERM][:] = [bool(value)]


class TorchConv2d(_TorchConv2d):

  def __init__(self, nndct_op_type, *args, **kwargs):
    super(TorchConv2d, self).__init__(nndct_op_type, *args, **kwargs)
    utils.op_register(nndct_op_type, "Conv2d")


class TorchConvTranspose2d(_TorchConv2d):

  def __init__(self, nndct_op_type, *args, **kwargs):
    super().__init__(nndct_op_type, *args, **kwargs)
    utils.op_register(nndct_op_type, "ConvTranspose2d")


class _TorchConv3d(base_op.Conv3d):

  def __init__(self, op_type, *args, **kwargs):
    super().__init__(op_type, *args, **kwargs)

  @property
  def kernel_size(self):
    return self.get_attr(self.AttrName.KERNEL)[::-1]

  @kernel_size.setter
  def kernel_size(self, value):
    self.set_attr(self.AttrName.KERNEL, value[::-1])

  @property
  def dilation(self):
    return self.get_attr(self.AttrName.DILATION)[::-1]

  @dilation.setter
  def dilation(self, value):
   self.set_attr(self.AttrName.DILATION, value[::-1])

  @property
  def groups(self):
    return self.get_attr(self.AttrName.GROUP)

  @groups.setter
  def groups(self, value):
    self.set_attr(self.AttrName.GROUP, value)

  @property
  def padding(self):
    pad = self.get_attr(self.AttrName.PAD)
    output_pad = self.get_attr(self.AttrName.OUTPUT_PAD)
    pad = [p + op for p, op in zip(pad, output_pad)]
    torch_pad = []
    for v in pad[::-2]:
      torch_pad.append(v)
    return torch_pad


  @padding.setter
  def padding(self, value):
    assert len(value) == 3
    pad = []
    for v in value[::-1]:
      pad.append(v)
      pad.append(v)

    output_pad = self.get_attr(self.AttrName.OUTPUT_PAD)
    pad = [p - op for p, op in zip(pad, output_pad)]
    self.set_attr(self.AttrName.PAD, pad)
    self.set_attr(self.AttrName.PAD_MODE, 'FLOOR')

  @property
  def output_padding(self):
    output_pad = []
    for v in self.get_attr(self.AttrName.OUTPUT_PAD)[::-2]:
      output_pad.append(v)
    return output_pad

  @output_padding.setter
  def output_padding(self, value):
    assert len(value) == 3
    output_pad = []
    for v in value[::-1]:
      output_pad.append(0)
      output_pad.append(v)
    self.set_attr(self.AttrName.OUTPUT_PAD, output_pad)
    pad = self.get_attr(self.AttrName.PAD)
    pad = [p - op for p, op in zip(pad, output_pad)]
    self.set_attr(self.AttrName.PAD, pad)

  @property
  def stride(self):
    return self.get_attr(self.AttrName.STRIDE)[::-1]

  @stride.setter
  def stride(self, value):
    self.set_attr(self.AttrName.STRIDE, value[::-1])

  @property
  def in_channels(self):
    return self.get_attr(self.AttrName.IN_DIM)

  @in_channels.setter
  def in_channels(self, value):
    self.set_attr(self.AttrName.IN_DIM, value)

  @property
  def out_channels(self):
    return self.get_attr(self.AttrName.OUT_DIM)

  @out_channels.setter
  def out_channels(self, value):
    self.set_attr(self.AttrName.OUT_DIM, value)


  @property
  def bias(self):
    return self.get_attr(self.AttrName.BIAS_TERM)

  @bias.setter
  def bias(self, value):
    self.set_attr(self.AttrName.BIAS_TERM, bool(value))


class TorchConv3d(_TorchConv3d):

  def __init__(self, nndct_op_type, *args, **kwargs):
    super().__init__(nndct_op_type, *args, **kwargs)
    utils.op_register(nndct_op_type, "Conv3d")


class TorchConvTranspose3d(_TorchConv3d):

  def __init__(self, nndct_op_type, *args, **kwargs):
    super().__init__(nndct_op_type, *args, **kwargs)
    utils.op_register(nndct_op_type, "ConvTranspose3d")


class TorchMaxPool(base_op.MaxPool):

  def __init__(self, *args, **kwargs):
    super(TorchMaxPool, self).__init__(NNDCT_OP.MAX_POOL, *args, **kwargs)
    utils.op_register(NNDCT_OP.MAX_POOL, "MaxPool2d")

  @property
  def kernel_size(self):
    return self._attr_value_mem[self.AttrName.KERNEL][::-1]

  @kernel_size.setter
  def kernel_size(self, value):
    self._attr_value_mem[self.AttrName.KERNEL][:] = value[::-1]

  @property
  def ceil_mode(self):
    return bool(self._attr_value_mem[self.AttrName.PAD_MODE][0])

  @ceil_mode.setter
  def ceil_mode(self, mode):
    self._attr_value_mem[self.AttrName.PAD_MODE][:] = [int(mode)]

  @property
  def padding(self):
    return [
        self._attr_value_mem[self.AttrName.PAD][2],
        self._attr_value_mem[self.AttrName.PAD][0]
    ]

  @padding.setter
  def padding(self, value):
    self._attr_value_mem[self.AttrName.PAD][:] = [
        value[1], value[1], value[0], value[0]
    ]

  @property
  def stride(self):
    return self._attr_value_mem[self.AttrName.STRIDE][::-1]

  @stride.setter
  def stride(self, value):
    self._attr_value_mem[self.AttrName.STRIDE][:] = value[::-1]

class TorchMaxPool1d(base_op.MaxPool1d):

  def __init__(self, *args, **kwargs):
    super(TorchMaxPool1d, self).__init__(NNDCT_OP.MAX_POOL1D, *args, **kwargs)
    utils.op_register(NNDCT_OP.MAX_POOL1D, "MaxPool1d")

  @property
  def kernel_size(self):
    return self.get_attr(self.AttrName.KERNEL)

  @kernel_size.setter
  def kernel_size(self, value):
    self.set_attr(self.AttrName.KERNEL, value)

  @property
  def ceil_mode(self):
    
    return bool(self.get_attr(self.AttrName.PAD_MODE))

  @ceil_mode.setter
  def ceil_mode(self, mode):
    self.set_attr(self.AttrName.PAD_MODE, int(mode))

  @property
  def padding(self):
    return [self.get_attr(self.AttrName.PAD)[0]]

  @padding.setter
  def padding(self, value):
    self.set_attr(self.AttrName.PAD, [value[0], value[0]])
   
  @property
  def stride(self):
    return self.get_attr(self.AttrName.STRIDE)

  @stride.setter
  def stride(self, value):
    self.set_attr(self.AttrName.STRIDE, value)


class TorchAvgPool(base_op.AvgPool):

  def __init__(self, *args, **kwargs):
    super(TorchAvgPool, self).__init__(NNDCT_OP.AVG_POOL, *args, **kwargs)
    utils.op_register(NNDCT_OP.AVG_POOL, "AvgPool2d")

  @property
  def kernel_size(self):
    return self._attr_value_mem[self.AttrName.KERNEL][::-1]

  @kernel_size.setter
  def kernel_size(self, value):
    self._attr_value_mem[self.AttrName.KERNEL][:] = value[::-1]

  @property
  def ceil_mode(self):
    return bool(self._attr_value_mem[self.AttrName.PAD_MODE][0])

  @ceil_mode.setter
  def ceil_mode(self, mode):
    self._attr_value_mem[self.AttrName.PAD_MODE][:] = [int(mode)]

  @property
  def padding(self):
    return [
        self._attr_value_mem[self.AttrName.PAD][2],
        self._attr_value_mem[self.AttrName.PAD][0]
    ]

  @padding.setter
  def padding(self, value):
    self._attr_value_mem[self.AttrName.PAD][:] = [
        value[1], value[1], value[0], value[0]
    ]

  @property
  def stride(self):
    return self._attr_value_mem[self.AttrName.STRIDE][::-1]

  @stride.setter
  def stride(self, value):
    self._attr_value_mem[self.AttrName.STRIDE][:] = value[::-1]

  @property
  def count_include_pad(self):
    return self.get_attr(self.AttrName.COUNT_INCLUDE_PAD)

  @count_include_pad.setter
  def count_include_pad(self, value):
    self.set_attr(self.AttrName.COUNT_INCLUDE_PAD, bool(value))


class TorchAdaptiveAvgPool(base_op.UnaryOp):
  def __init__(self, *args, **kwargs):
    super(TorchAdaptiveAvgPool, self).__init__(NNDCT_OP.ADAPTIVEAVGPOOL2D, *args, **kwargs)
    utils.op_register(NNDCT_OP.ADAPTIVEAVGPOOL2D, "AdaptiveAvgPool2d")
    

class TorchSize(base_op.Shape):

  def __init__(self, *args, **kwargs):
    super(TorchSize, self).__init__(NNDCT_OP.SHAPE, *args, **kwargs)
    utils.op_register(NNDCT_OP.SHAPE, 'size')

  @property
  def dim(self):
    return self.get_attr(self.AttrName.AXIS)
  

  @dim.setter
  def dim(self, value):
    self.set_attr(self.AttrName.AXIS, value)
    


class TorchCat(base_op.Concat):

  def __init__(self, *args, **kwargs):
    super(TorchCat, self).__init__(NNDCT_OP.CONCAT, *args, **kwargs)
    utils.op_register(NNDCT_OP.CONCAT, 'cat')

  @property
  def dim(self):
    return self.get_attr(self.AttrName.AXIS)
   
  @dim.setter
  def dim(self, value):
    self.set_attr(self.AttrName.AXIS, value)
 

class TorchView(base_op.Reshape):

  def __init__(self):
    super(TorchView, self).__init__(NNDCT_OP.RESHAPE)
    utils.op_register(NNDCT_OP.RESHAPE, 'reshape')

  @property
  def shape(self):
    return self.get_attr(self.AttrName.SHAPE)

  @shape.setter
  def shape(self, value):
    if isinstance(value, (tuple, list)):
      value = list(value)
    else:
      value = [value]

    self.set_attr(self.AttrName.SHAPE, value)


class TorchDropout(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchDropout, self).__init__(NNDCT_OP.DROPOUT, *args, **kwargs)
    utils.op_register(NNDCT_OP.DROPOUT, 'Dropout')


class TorchPermuteInvarOp(base_op.PermuteInvariantOp):

  def __init__(self, nndct_op_type, torch_op_type, *args, **kwargs):
    super().__init__(nndct_op_type, *args, **kwargs)
    utils.op_register(nndct_op_type, torch_op_type)

  @property
  def dim(self):
    return tuple(self.get_attr(self.AttrName.DIMS))
   

  @dim.setter
  def dim(self, value):
    if isinstance(value, (tuple, list)):
      value = list(value)
    else:
      value = [value]

    self.set_attr(self.AttrName.DIMS, value)

  @property
  def keepdim(self):
    return self._attr_value_mem[self.AttrName.KEEP_DIMS][0]

  @keepdim.setter
  def keepdim(self, value):
    self._attr_value_mem[self.AttrName.KEEP_DIMS][:] = [bool(value)]


class TorchPermute(base_op.Permute):

  def __init__(self, *args, **kwargs):
    super(TorchPermute, self).__init__(NNDCT_OP.PERMUTE, *args, **kwargs)
    utils.op_register(NNDCT_OP.PERMUTE, 'permute')

  @property
  def dims(self):
    return self.get_attr(self.AttrName.ORDER)
  

  @dims.setter
  def dims(self, value):
    self.set_attr(self.AttrName.ORDER, value)
  

class TorchTranspose(base_op.Permute):

  def __init__(self, *args, **kwargs):
    super(TorchTranspose, self).__init__(NNDCT_OP.TRANSPOSE, *args, **kwargs)
    utils.op_register(NNDCT_OP.TRANSPOSE, 'transpose')
    

class TorchContiguous(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchContiguous, self).__init__(NNDCT_OP.CONTIGUOUS, *args, **kwargs)
    utils.op_register(NNDCT_OP.CONTIGUOUS, 'contiguous')


class TorchChunk(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchChunk, self).__init__(NNDCT_OP.CHUNK, *args, **kwargs)
    utils.op_register(NNDCT_OP.CHUNK, 'chunk')


class TorchInterpolate(base_op.Resize):
  def __init__(self):
    super().__init__()
    utils.op_register(NNDCT_OP.RESIZE, 'interpolate')
    # self._scale_factor_bc = [1.0, 1.0]
 

  @property
  def size(self):
    size = self.get_attr(self.AttrName.SIZE)[:]
    if size[0] == 0 and size[1] == 0:
      return None
    else:
      return size

  @size.setter
  def size(self, size):
    self.set_attr(self.AttrName.SIZE, size[:])

  @property
  def scale_factor(self):
    scale = self.get_attr(self.AttrName.SCALE)[::-1]
    if scale[0] == 1.0 and scale[1] == 1.0:
      return None
    else:
      return scale

  @scale_factor.setter
  def scale_factor(self, factor):
    if isinstance(factor, float):
        scale_factor = 2 * [factor]
    else:
        scale_factor = factor[::-1]
    self.set_attr(self.AttrName.SCALE, scale_factor)


  @property
  def mode(self):
    mode = self.get_attr(self.AttrName.MODE)
    return f"'{mode.lower()}'"

  @mode.setter
  def mode(self, mode):
    if mode not in ["'nearest'", "'bilinear'"]:
      raise RuntimeError(f"Don't support {mode} mode in upsampling.")
    # mode = 0 if mode == "'nearest'" else 3
    # self.set_attr(self.AttrName.MODE, mode)
    self.set_attr(self.AttrName.MODE, mode.strip("'").upper())


class TorchResizeLinear(TorchInterpolate):
  def __init__(self):
    super().__init__()

  @property
  def align_corners(self):
    return self.get_attr(self.AttrName.ALIGN_CORNERS)

  @align_corners.setter
  def align_corners(self, value):
    self.set_attr(self.AttrName.ALIGN_CORNERS, bool(value))
    self.set_attr(self.AttrName.HALF_PIXEL_CENTERS, not(bool(value)))


class TorchInterpolate3d(base_op.Resize3d):
  def __init__(self):
    super().__init__()
    utils.op_register(NNDCT_OP.RESIZE_3D, 'interpolate')
    # self._scale_factor_bc = [1.0, 1.0]

  @property
  def size(self):
    size = self.get_attr(self.AttrName.SIZE)
    if all([s == 0 for s in size]):
      return None
    else:
      return [size[2]] + size[:2]

  @size.setter
  def size(self, size):
    self.set_attr(self.AttrName.SIZE, size[1:] + [size[0]])

  @property
  def scale_factor(self):
    scale = self.get_attr(self.AttrName.SCALE)
    if all([s == 1.0 for s in scale]):
      return None
    else:
      return [scale[2]] + scale[:2]

  @scale_factor.setter
  def scale_factor(self, factor):
    if isinstance(factor, float):
        scale_factor = 3 * [factor]
    else:
        scale_factor = factor[1:] + [factor[0]]
    self.set_attr(self.AttrName.SCALE, scale_factor)


  @property
  def mode(self):
    mode = self.get_attr(self.AttrName.MODE)
    return f"'{mode.lower()}'"

  @mode.setter
  def mode(self, mode):
    if mode != "'trilinear'":
      raise RuntimeError(f"Don't support {mode} mode in upsampling3d.")
    self.set_attr(self.AttrName.MODE, mode.strip("'").upper())


class TorchResizeTrilinear(TorchInterpolate3d):
  def __init__(self):
    super().__init__()

  @property
  def align_corners(self):
    return self.get_attr(self.AttrName.ALIGN_CORNERS)

  @align_corners.setter
  def align_corners(self, value):
    self.set_attr(self.AttrName.ALIGN_CORNERS, bool(value))
    self.set_attr(self.AttrName.HALF_PIXEL_CENTERS, not(bool(value)))

class TorchConst(base_op.Constant):
  def __init__(self):
    super().__init__(NNDCT_OP.CONST)
    utils.op_register(NNDCT_OP.CONST, 'tensor')

  @property
  def data(self):
    return self.get_attr(self.AttrName.DATA)

  @data.setter
  def data(self, data):
    self.set_attr(self.AttrName.DATA, data)
    

class TorchTensor(base_op.Constant):
  def __init__(self):
    super().__init__(NNDCT_OP.TENSOR)
    utils.op_register(NNDCT_OP.TENSOR, 'tensor')

  @property
  def data(self):
    return self.get_attr(self.AttrName.DATA)

  @data.setter
  def data(self, data):
    self.set_attr(self.AttrName.DATA, data)



class TorchMul(base_op.BinaryOp):

  def __init__(self, *args, **kwargs):
    super(TorchMul, self).__init__(NNDCT_OP.MULTIPLY, *args, **kwargs)
    utils.op_register(NNDCT_OP.MULTIPLY, 'mul')

  @property
  def input(self):
    return self.get_attr(self.AttrName.INPUT)

  @input.setter
  def input(self, input):
    self.set_attr(self.AttrName.INPUT, input)

  @property
  def other(self):
    return self.get_attr(self.AttrName.OTHER)

  @other.setter
  def other(self, other):
    self.set_attr(self.AttrName.OTHER, other)
  
class TorchDiv(base_op.BinaryOp):

  def __init__(self, *args, **kwargs):
    super(TorchDiv, self).__init__(NNDCT_OP.DIV, *args, **kwargs)
    utils.op_register(NNDCT_OP.DIV, 'div')

  @property
  def input(self):
    return self.get_attr(self.AttrName.INPUT)

  @input.setter
  def input(self, input):
    self.set_attr(self.AttrName.INPUT, input)

  @property
  def other(self):
    return self.get_attr(self.AttrName.OTHER)

  @other.setter
  def other(self, other):
    self.set_attr(self.AttrName.OTHER, other)
    
    
class TorchCast(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchCast, self).__init__(NNDCT_OP.CAST, *args, **kwargs)
    utils.op_register(NNDCT_OP.CAST, 'to')


class TorchFloor(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchFloor, self).__init__(NNDCT_OP.FLOOR, *args, **kwargs)
    utils.op_register(NNDCT_OP.FLOOR, 'floor')




class TorchBinaryOp(base_op.BinaryOp):
  def __init__(self, nndct_op_type, torch_op_type, force_to_primitive=False):
    super().__init__(nndct_op_type)
    utils.op_register(nndct_op_type, torch_op_type, force_to_primitive=force_to_primitive)

  @property
  def input(self):
    return self.get_attr(self.AttrName.INPUT)

  @input.setter
  def input(self, input):
    self.set_attr(self.AttrName.INPUT, input)

  @property
  def other(self):
    return self.get_attr(self.AttrName.OTHER)

  @other.setter
  def other(self, other):
    self.set_attr(self.AttrName.OTHER, other)


class TorchUnaryOp(base_op.UnaryOp):
  def __init__(self, nndct_op_type, torch_op_type, force_to_primitive=False):
    super().__init__(nndct_op_type)
    utils.op_register(nndct_op_type, torch_op_type, force_to_primitive=force_to_primitive)

  @property
  def input(self):
    return self.get_attr(self.AttrName.INPUT)

  @input.setter
  def input(self, input):
    self.set_attr(self.AttrName.INPUT, input)


class TorchFloorDiv(Operation):
  def __init__(self, *args, **kwargs):
    super(TorchFloorDiv, self).__init__(NNDCT_OP.FLOOR_DIV, *args, **kwargs)
    utils.op_register(NNDCT_OP.FLOOR_DIV, 'floor_divide')


class TorchSoftmax(base_op.Softmax):

  def __init__(self, *args, **kwargs):
    super().__init__()
    utils.op_register(NNDCT_OP.SOFTMAX, 'Softmax')

  @property
  def dim(self):
    return self._attr_value_mem[self.AttrName.AXIS][0]
   

  @dim.setter
  def dim(self, value):
    self._attr_value_mem[self.AttrName.AXIS][:] = [value]


class TorchExp(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchExp, self).__init__(NNDCT_OP.EXP, *args, **kwargs)
    utils.op_register(NNDCT_OP.EXP, 'exp')


class TorchDetach(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchDetach, self).__init__(NNDCT_OP.DETACH, *args, **kwargs)
    utils.op_register(NNDCT_OP.DETACH, 'detach')




class TorchRsub(base_op.Sub):

  def __init__(self):
    super().__init__(NNDCT_OP.RSUB)
    utils.op_register(NNDCT_OP.RSUB, 'sub')

  @property
  def input(self):
    return self.get_attr(self.AttrName.INPUT)

  @input.setter
  def input(self, input):
    self.set_attr(self.AttrName.INPUT, input)

  @property
  def other(self):
    return self.get_attr(self.AttrName.OTHER)

  @other.setter
  def other(self, other):
    self.set_attr(self.AttrName.OTHER, other)


class TorchSelect(base_op.CustomOp):

  def __init__(self, *args, **kwargs):
    super(TorchSelect, self).__init__(NNDCT_OP.SELECT, *args, **kwargs)
    utils.op_register(NNDCT_OP.SELECT, 'select')


class TorchSigmoid(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchSigmoid, self).__init__(NNDCT_OP.SIGMOID, *args, **kwargs)
    utils.op_register(NNDCT_OP.SIGMOID, 'Sigmoid')


class TorchRepeat(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchRepeat, self).__init__(NNDCT_OP.REPEAT, *args, **kwargs)
    utils.op_register(NNDCT_OP.REPEAT, 'repeat')


class TorchInplaceCopy(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchInplaceCopy, self).__init__(NNDCT_OP.INPLACE_COPY, *args,
                                           **kwargs)
    utils.op_register(NNDCT_OP.INPLACE_COPY, 'copy_')


# class TorchExpand(Operation):

#   def __init__(self, *args, **kwargs):
#     super(TorchExpand, self).__init__(NNDCT_OP.EXPAND, *args, **kwargs)
#     utils.op_register(NNDCT_OP.EXPAND, 'expand')


class TorchEmpty(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchEmpty, self).__init__(NNDCT_OP.EMPTY, *args, **kwargs)
    utils.op_register(NNDCT_OP.EMPTY, 'empty')


class TorchUnsqueeze(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchUnsqueeze, self).__init__(NNDCT_OP.UNSQUEEZE, *args, **kwargs)
    utils.op_register(NNDCT_OP.UNSQUEEZE, 'unsqueeze')



class TorchLstm(base_op.Lstm):

  def __init__(self, *args, **kwargs):
    super(TorchLstm, self).__init__(NNDCT_OP.BASIC_LSTM, *args, **kwargs)
    utils.op_register(NNDCT_OP.BASIC_LSTM, 'LSTM')

  @property
  def input_size(self):
    return self._attr_value_mem[self.AttrName.INPUT_SIZE][0]

  @input_size.setter
  def input_size(self, value):
    self._attr_value_mem[self.AttrName.INPUT_SIZE][:] = [value]

  @property
  def hidden_size(self):
    return self._attr_value_mem[self.AttrName.HIDDEN_SIZE][0]

  @hidden_size.setter
  def hidden_size(self, value):
    self._attr_value_mem[self.AttrName.HIDDEN_SIZE][:] = [value]

  @property
  def bidirectional(self):
    return self._attr_value_mem[self.AttrName.BIDIRECTIONAL][0]

  @bidirectional.setter
  def bidirectional(self, value):
    self._attr_value_mem[self.AttrName.BIDIRECTIONAL][:] = [bool(value)]

  @property
  def num_layers(self):
    return self._attr_value_mem[self.AttrName.NUM_LAYERS][0]

  @num_layers.setter
  def num_layers(self, value):
    self._attr_value_mem[self.AttrName.NUM_LAYERS][:] = [value]

  @property
  def batch_first(self):
    return self._attr_value_mem[self.AttrName.BATCH_FIRST][0]

  @batch_first.setter
  def batch_first(self, value):
    self._attr_value_mem[self.AttrName.BATCH_FIRST][:] = [bool(value)]


class TorchGru(base_op.Gru):
  def __init__(self, *args, **kwargs):
    super(TorchGru, self).__init__(NNDCT_OP.BASIC_GRU, *args, **kwargs)
    utils.op_register(NNDCT_OP.BASIC_GRU, 'GRU')

  @property
  def input_size(self):
    return self._attr_value_mem[self.AttrName.INPUT_SIZE][0]

  @input_size.setter
  def input_size(self, value):
    self._attr_value_mem[self.AttrName.INPUT_SIZE][:] = [value]

  @property
  def hidden_size(self):
    return self._attr_value_mem[self.AttrName.HIDDEN_SIZE][0]

  @hidden_size.setter
  def hidden_size(self, value):
    self._attr_value_mem[self.AttrName.HIDDEN_SIZE][:] = [value]

  @property
  def bidirectional(self):
    return self._attr_value_mem[self.AttrName.BIDIRECTIONAL][0]

  @bidirectional.setter
  def bidirectional(self, value):
    self._attr_value_mem[self.AttrName.BIDIRECTIONAL][:] = [bool(value)]

  @property
  def num_layers(self):
    return self._attr_value_mem[self.AttrName.NUM_LAYERS][0]

  @num_layers.setter
  def num_layers(self, value):
    self._attr_value_mem[self.AttrName.NUM_LAYERS][:] = [value]

  @property
  def batch_first(self):
    return self._attr_value_mem[self.AttrName.BATCH_FIRST][0]

  @batch_first.setter
  def batch_first(self, value):
    self._attr_value_mem[self.AttrName.BATCH_FIRST][:] = [bool(value)]


class TorchSplit(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchSplit, self).__init__(NNDCT_OP.SPLIT, *args, **kwargs)
    utils.op_register(NNDCT_OP.SPLIT, 'split')


class TorchZeros(base_op.ConstFromShape):

  def __init__(self, *args, **kwargs):
    super(TorchZeros, self).__init__(NNDCT_OP.ZEROS, *args, **kwargs)
    utils.op_register(NNDCT_OP.ZEROS, 'zeros')

  @property
  def size(self):
    return self.get_attr(self.AttrName.SHAPE)

  @size.setter
  def size(self, size):
    assert isinstance(size, (list, tuple))
    self.set_attr(self.AttrName.SHAPE, list(size))


class TorchPad(base_op.Pad):
  mode_map = {"'constant'": 0,
            "'reflect'": 1,
            "'replicate'": 2}

  def __init__(self):
    super().__init__()
    utils.op_register(self.type, "pad")

  @property
  def pad(self):
    pad = self.get_attr(self.AttrName.PAD_WITH)
    # HW -> WH
    pad = pad[-4:-2] + pad[2:4]
    return pad

  @pad.setter
  def pad(self, value):
    if len(value) != 4:
      raise RuntimeError("only support 2D pad")
    # WH -> HW
    value = value[-2:] + value[:2]
    value = [0, 0] + value + [0, 0]
    self.set_attr(self.AttrName.PAD_WITH, value)

  @property
  def mode(self):
    mode = self.get_attr(self.AttrName.MODE)
    mode_map_r = {v: k for k, v in self.mode_map.items()}
    return mode_map_r[mode]

  @mode.setter
  def mode(self, mode):
    if mode not in ["'constant'", "'reflect'", "'replicate'"]:
      raise RuntimeError(f"mode `{mode}` not supported in pad.")
    # mode = 0 if mode == "'constant'" else 1
    self.set_attr(self.AttrName.MODE, self.mode_map[mode])

  @property
  def value(self):
    return self.get_attr(self.AttrName.CONSTANT_VALUES)[0]

  @value.setter
  def value(self, constant):
    self.set_attr(self.AttrName.CONSTANT_VALUES, [float(constant)] * 8)


class TorchMatmul(base_op.Matmul):

  def __init__(self, *args, **kwargs):
    super(TorchMatmul, self).__init__(NNDCT_OP.MATMUL, *args, **kwargs)
    utils.op_register(NNDCT_OP.MATMUL, 'matmul')


class TorchClamp(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchClamp, self).__init__(NNDCT_OP.CLAMP, *args, **kwargs)
    utils.op_register(NNDCT_OP.CLAMP, 'clamp')
 

# TODO
class TorchSlice(base_op.StridedSlice):

  def __init__(self):
    super().__init__()
    utils.op_register(NNDCT_OP.STRIDED_SLICE, 'strided_slice', force_to_primitive=True)

  @property
  def dim(self):
    return self.get_attr(self.AttrName.DIMS)
  
  @dim.setter
  def dim(self, dims):
    return self.set_attr(self.AttrName.DIMS, dims)

  @property
  def start(self):
    return self.get_attr(self.AttrName.BEGIN)

  # @property
  # def start(self):
  #   if self._input_ndim < 4:
  #     return self.get_attr(self.AttrName.BEGIN)
  #   else:
  #     begin = [0] * self._input_ndim
  #     for dim, pos in enumerate(self.get_attr(self.AttrName.BEGIN)):
  #       new_dim = transformed_axis(
  #           src=DataFormat.channel_first, dst=DataFormat.channel_first, ndim=self._input_ndim, dim=dim)
  #       begin[new_dim] = pos
  #     return begin

  @start.setter
  def start(self, start):
    self.set_attr(self.AttrName.BEGIN, start)
  

  # @start.setter
  # def start(self, start):
  #   if self._input_ndim < 4:
  #     begin_mask = 0
  #     for dim, pos in enumerate(start):
  #       if pos == 0:
  #         begin_mask |= 1 << dim
  #     self.set_attr(self.AttrName.BEGIN_MASK, begin_mask)
  #     self.set_attr(self.AttrName.BEGIN, start)

  #   else:
  #     begin = [0] * self._input_ndim
  #     begin_mask = 0
  #     for dim, pos in enumerate(start):
  #       new_dim = transformed_axis(
  #           src=DataFormat.channel_first, dst=DataFormat.channel_first, ndim=self._input_ndim, dim=dim)
  #       begin[new_dim] = pos

  #     for dim, pos in enumerate(begin):
  #       if pos == 0:
  #         begin_mask |= 1 << dim

  #     self.set_attr(self.AttrName.BEGIN_MASK, begin_mask)
  #     self.set_attr(self.AttrName.BEGIN, begin)
  
  @property
  def end(self):
    return self.get_attr(self.AttrName.END)
  # @property
  # def end(self):
  #   if self._input_ndim < 4:
  #     return self.get_attr(self.AttrName.END)
  #   else:
  #     end = [NNDCT_CONSTANT.INT_MAX] * self._input_ndim
  #     for dim, pos in enumerate(self.get_attr(self.AttrName.END)):
  #       new_dim = transformed_axis(
  #           src=DataFormat.channel_first, dst=DataFormat.channel_first, ndim=self._input_ndim, dim=dim)
  #       end[new_dim] = pos
  #     return end

  @end.setter
  def end(self, end):
    self.set_attr(self.AttrName.END, end)
  #   if self._input_ndim < 4:
  #     end_mask = 0
  #     for dim, pos in enumerate(end):
  #       if isinstance(pos, int) and pos >= NNDCT_CONSTANT.INT_MAX:
  #         end_mask |= 1 << dim
  #     self.set_attr(self.AttrName.END_MASK, end_mask)
  #     self.set_attr(self.AttrName.END, end)
  #   else:
  #     new_end = [NNDCT_CONSTANT.INT_MAX] * self._input_ndim
  #     end_mask = 0
  #     for dim, pos in enumerate(end):
  #       new_dim = transformed_axis(
  #           src=DataFormat.channel_first, dst=DataFormat.channel_first, ndim=self._input_ndim, dim=dim)
  #       new_end[new_dim] = pos

  #     for dim, pos in enumerate(new_end):
  #       if isinstance(pos, int) and pos >= NNDCT_CONSTANT.INT_MAX:
  #         end_mask |= 1 << dim

  #     self.set_attr(self.AttrName.END_MASK, end_mask)
  #     self.set_attr(self.AttrName.END, new_end)

  @property
  def step(self):
    return self.get_attr(self.AttrName.STRIDES)
  # @property
  # def step(self):
  #   if self._input_ndim < 4:
  #     return self.get_attr(self.AttrName.STRIDES)
  #   else:
  #     strides = [1] * self._input_ndim
  #     for dim, step in enumerate(self.get_attr(self.AttrName.STRIDES)):
  #       new_dim = transformed_axis(
  #           src=DataFormat.channel_first, dst=DataFormat.channel_first, ndim=self._input_ndim, dim=dim)
  #       strides[new_dim] = step
  #     return strides

  @step.setter
  def step(self, steps):
    self.set_attr(self.AttrName.STRIDES, steps)
  # @step.setter
  # def step(self, steps):
  #   if self._input_ndim < 4:
  #     self.set_attr(self.AttrName.STRIDES, steps)
  #   else:
  #     strides = [1] * self._input_ndim
  #     for dim, step in enumerate(steps):
  #       new_dim = transformed_axis(
  #           src=DataFormat.channel_first, dst=DataFormat.channel_first, ndim=self._input_ndim, dim=dim)
  #       strides[new_dim] = step

  #     self.set_attr(self.AttrName.STRIDES, strides)


class TorchArange(Operation):

  def __init__(self):
    super().__init__(NNDCT_OP.ARANGE)
    utils.op_register(NNDCT_OP.ARANGE, 'arange')


# class TorchSlicedInplaceCopy(Operation):

#   def __init__(self):
#     super().__init__(NNDCT_OP.SLICE_TENSOR_INPLACE_COPY)


class TorchEmbeddingBag(base_op.EmbeddingBag):
  def __init__(self):
    super().__init__(NNDCT_OP.EMBEDDING_BAG)
    utils.op_register(NNDCT_OP.EMBEDDING_BAG, "EmbeddingBag")

class TorchEmbedding(base_op.Embedding):
  def __init__(self):
    super().__init__(NNDCT_OP.EMBEDDING)
    utils.op_register(NNDCT_OP.EMBEDDING, "Embedding")


class TorchBaseOperation(base_op.CustomOp):
  def __init__(self, nndct_op_type, torch_op_type=None, force_to_primitive=False, schema=None, class_type=None):
    super().__init__(nndct_op_type)
    if torch_op_type is not None:
      utils.op_register(nndct_op_type, torch_op_type, force_to_primitive=force_to_primitive, schema=schema, class_type=class_type)

class TorchAutoInferOperation(base_op.CustomOp):
  def __init__(self, nndct_op_type, torch_op_type=None, force_to_primitive=False, schema=None, class_type=None):
    super().__init__(nndct_op_type)
    self._config_list = {}
    if torch_op_type is not None:
      utils.op_register(nndct_op_type, torch_op_type, force_to_primitive=force_to_primitive, schema=schema, class_type=class_type)

  def has_config(self, config_name):
    return config_name in self._config_list.keys()

  def get_config(self, config_name: str) -> Any:
    return self._config_list[config_name]      

  def set_config(self, config_name: str, value: Any) -> None:
    if config_name not in self._configs:
      self._configs.append(config_name)
      self._config_list[config_name] = value
      self._set_attr_user(config_name, value)
    else:
      self._release_attr_user(config_name)
      self._set_attr_user(config_name, value)
      self._config_list[config_name] = value


class TorchCustomOperation(base_op.CustomOp):
  def __init__(self, nndct_op_type, torch_op_type):
    super().__init__(nndct_op_type)
    utils.op_register(nndct_op_type, torch_op_type, class_type=utils.TorchOpClassType.CUSTOM_FUNCTION)



class TorchSqueeze(base_op.Squeeze):

  def __init__(self):
    super().__init__()
    utils.op_register(NNDCT_OP.SQUEEZE, "squeeze")

  @property
  def dim(self):
    return tuple(self.get_attr(self.AttrName.DIMS))
    

  @dim.setter
  def dim(self, value):
    if isinstance(value, (tuple, list)):
      value = list(value)
    else:
      value = [value]

    self.set_attr(self.AttrName.DIMS, value)


class TorchLayerNorm(base_op.LayerNorm):

  @unique
  class ParamName(base_op.AutoName):
    GAMMA = "weight"
    BETA = "bias"

  def __init__(self):
    super().__init__(NNDCT_OP.LAYER_NORM)
    utils.op_register(NNDCT_OP.LAYER_NORM, "LayerNorm")

  @property
  def eps(self):
    return self.get_attr(self.AttrName.EPS)

  @eps.setter
  def eps(self, value):
    self.set_attr(self.AttrName.EPS, value)

  @property
  def normalized_shape(self):
    return self.get_attr(self.AttrName.NORMALIZED_SHAPE)

  @normalized_shape.setter
  def normalized_shape(self, value):
    self.set_attr(self.AttrName.NORMALIZED_SHAPE, value)

  @property
  def elementwise_affine(self):
    return self.get_attr(self.AttrName.ELEMENTWISE_AFFINE)

  @elementwise_affine.setter
  def elementwise_affine(self, value):
    self.set_attr(self.AttrName.ELEMENTWISE_AFFINE, value)


class TorchUnknownOperation(Operation):
  def __init__(self, nndct_op_type):
    super().__init__(nndct_op_type)


class TorchPixelShuffle(base_op.PixelShuffle):
  def __init__(self):
    super().__init__(NNDCT_OP.PIXEL_SHUFFLE)
    utils.op_register(NNDCT_OP.PIXEL_SHUFFLE, "PixelShuffle")

  @property
  def upscale_factor(self):
    return self.get_attr(self.AttrName.SCALE)

  @upscale_factor.setter
  def upscale_factor(self, value):
    self.set_attr(self.AttrName.SCALE, value)


class TorchPixelUnshuffle(base_op.PixelShuffle):
  def __init__(self):
    super().__init__(NNDCT_OP.PIXEL_UNSHUFFLE)
    utils.op_register(NNDCT_OP.PIXEL_UNSHUFFLE, "PixelUnshuffle")

  @property
  def downscale_factor(self):
    return self.get_attr(self.AttrName.SCALE)

  @downscale_factor.setter
  def downscale_factor(self, value):
    self.set_attr(self.AttrName.SCALE, value)

class TorchCorrelationOperation(base_op.Correlation):
  # def __init__(self, nndct_op_type, torch_op_type, force_to_primitive=False, schema=None):
  #   super().__init__(s)
  #   utils.op_register(nndct_op_type, torch_op_type, force_to_primitive=force_to_primitive, schema=schema)
  def __init__(self, nndct_op_type, torch_op_type, force_to_primitive=False, schema=None, *args, **kwargs):
    super().__init__(nndct_op_type, *args, **kwargs)
    utils.op_register(nndct_op_type, torch_op_type, force_to_primitive=force_to_primitive, schema=schema)

  @property
  def pad_size(self):
    return self.get_attr(self.AttrName.PAD_SIZE)

  @pad_size.setter
  def pad_size(self, value):
    self.set_attr(self.AttrName.PAD_SIZE, value)


class TorchCostVolumeOperation(base_op.CostVolume):

  def __init__(self, nndct_op_type, torch_op_type, force_to_primitive=False, schema=None, *args, **kwargs):
    super().__init__(nndct_op_type, *args, **kwargs)
    utils.op_register(nndct_op_type, torch_op_type, force_to_primitive=force_to_primitive, schema=schema)

  @property
  def maxdisp(self):
    return self.get_attr(self.AttrName.MAXDISP)

  @maxdisp.setter
  def maxdisp(self, value):
    self.set_attr(self.AttrName.MAXDISP, value)

class TorchLogSoftmax(base_op.LogSoftmax):

  def __init__(self, *args, **kwargs):
    super().__init__()
    utils.op_register(NNDCT_OP.LOG_SOFTMAX, 'LogSoftmax')

  @property
  def dim(self):
    return self._attr_value_mem[self.AttrName.AXIS][0]
                         
  @dim.setter
  def dim(self, value):
    self._attr_value_mem[self.AttrName.AXIS][:] = [value]

class TorchArgMax_DIM(base_op.ArgMax_DIM):

  def __init__(self, *args, **kwargs):
    super().__init__()
    utils.op_register(NNDCT_OP.ARGMAX_DIM, 'argmax')

  @property
  def dim(self):
    return self.get_attr(self.AttrName.AXIS)
                         
  @dim.setter
  def dim(self, value):
    self.set_attr(self.AttrName.AXIS, value)
