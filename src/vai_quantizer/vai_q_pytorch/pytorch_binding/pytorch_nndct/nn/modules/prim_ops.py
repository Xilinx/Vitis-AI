

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

import os
import torch
import torch.nn.functional as F
from nndct_shared.utils import NndctOption, NndctScreenLogger
from nndct_shared.base import NNDCT_CONSTANT
from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import quantize_tensors
from nndct_shared.quantization import quant_reluk_params
from nndct_shared.quantization import quant_channel_scale_params
import pytorch_nndct.utils as py_utils
from typing import Any, Optional, Sequence, Union
from torch.autograd import Variable
# __all__ = ["Int", "strided_slice", "Input", "slice_tensor_inplace_copy"]


class _PrimModule(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.params_name = None
    self.node = None

  def forward(*args, **kwargs):
    pass


class deephi_Int(_PrimModule):

  def __init__(self):
    super().__init__()

  def forward(self, input):
    output = int(input)
    return output


@py_utils.register_quant_op
def Int(*args, **kwargs):
  return deephi_Int(*args, **kwargs)


class deephi_QuantInput(_PrimModule):
  def __init__(self):
    super().__init__()

  def forward(self, input):
    qinput = quantize_tensors([input], self.node, tensor_type='input')[0]

    output = qinput
    if NndctOption.nndct_stat.value > 2:
      print('Channel number of input data: {}'.format(output.shape[1]))
      print('Input data histogram: {}'.format( output.histc(bins = 10).cpu().detach().numpy() ))
      print('Network input channel-wise statistic [Min, Max, Mean, Std]:')
      t = output.transpose(0, 1)
      for c in range(t.shape[0]):
        print('[{}, {}, {}, {}]'.format( t[c].min(), t[c].max(), t[c].float().mean(), t[c].float().std() ))
        print('histogram: {}'.format( t[c].histc(bins = 10).cpu().detach().numpy() ))

    output = quantize_tensors([output], self.node)[0]
    return output

@py_utils.register_quant_op
def quant_input(*args, **kwargs):
  return deephi_QuantInput(*args, **kwargs)


class deephi_DequantOutput(_PrimModule):
  def __init__(self):
    super().__init__()

  def forward(self, input):
    output = input
    return output

@py_utils.register_quant_op
def dequant_output(*args, **kwargs):
  return deephi_DequantOutput(*args, **kwargs)

class deephi_Input(_PrimModule):

  def __init__(self):
    super().__init__()

  def forward(self, input):
    if self.quantizer is not None and self.quantizer.exporting and isinstance(input, (tuple, list)):
      input = input[0]

    elif isinstance(input, (tuple, list)):
      for idx in range(len(input)):
        if isinstance(input[idx], torch.Tensor) and input[idx].storage().size() != input[idx].numel():
          input[idx] = torch.clone(input[idx])
          NndctScreenLogger().warning_once(f"The element number of tensor is not equal to its storage size. Please check input tensor in node {self.node.name}!")

    qinput = quantize_tensors([input], self.node, tensor_type='input')[0]
    # check input shape
    if self.node.out_tensors[0].is_complete_tensor() and self.node.out_tensors[0].ndim == 4:
      # py_utils.blob_to_torch_format(self.node.out_tensors[0])
      if not (list(self.node.out_tensors[0].shape[1:]) == list(input.size())[1:]):
        NndctScreenLogger().warning_once(f"The shape of input ({input.shape[1:]}) should be the same with that of dummy input ({self.node.out_tensors[0].shape[1:]})")
      # py_utils.blob_to_nndct_format(self.node.out_tensors[0])
    output = qinput

    if (self.node.in_quant_part and NndctOption.nndct_stat.value > 2):
      print('Channel number of input data: {}'.format(output.shape[1]))
      print('Input data histogram: {}'.format( output.histc(bins = 10).cpu().detach().numpy() ))
      print('Network input channel-wise statistic [Min, Max, Mean, Std]:')
      t = output.transpose(0, 1)
      for c in range(t.shape[0]):
        print('[{}, {}, {}, {}]'.format( t[c].min(), t[c].max(), t[c].float().mean(), t[c].float().std() ))
        print('histogram: {}'.format( t[c].histc(bins = 10).cpu().detach().numpy() ))

    if self.node.in_quant_part:
      if isinstance(output, (tuple, list)):
        output = quantize_tensors(output, self.node)
      else:
        output = quantize_tensors([output], self.node)[0]

    return output


@py_utils.register_quant_op
def Input(*args, **kwargs):
  return deephi_Input(*args, **kwargs)


class deephi_StridedSlice(_PrimModule):
  def __init__(self):
    super().__init__()

  def forward(self, input, dim, start, end, step):
    size = input.size()
    break_symbol = ':'
    symbols = ""
    start_symbol = []
    end_symbol = []
    step_symbol = []
    for i in range(dim[0]):
      start_symbol.append(str(0))
      end_symbol.append(str(int(size[i])))
      step_symbol.append(str(1))

    for i in range(len(start)):
      start_symbol.append(str(int(start[i])))
      end_symbol.append(str(int(end[i])))
      step_symbol.append(str(int(step[i])))
    
    for i in range(len(start_symbol)):
      slice_symbol = break_symbol.join([start_symbol[i], end_symbol[i], step_symbol[i]])
      if i > 0:
        symbols += "," + slice_symbol
      else:
        symbols = slice_symbol

    eval_str = f"input[{symbols}]"
    output = eval(eval_str)
    output = quantize_tensors([output], self.node)[0]
    return output
  


@py_utils.register_quant_op
def strided_slice(*args, **kwargs):
  return deephi_StridedSlice(*args, **kwargs)


class deephi_SliceInplaceCopy(_PrimModule):
  def __init__(self):
    super().__init__()

  def forward(self, input, source, dim, index):
    index = torch.tensor([index]).to(input.device)
    output = input.index_copy_(dim, index, source.unsqueeze(dim))
    output = quantize_tensors([output], self.node)[0]
    return output


@py_utils.register_quant_op
def slice_tensor_inplace_copy(*args, **kwargs):
  return deephi_SliceInplaceCopy(*args, **kwargs)


class deephi_Index(_PrimModule):
  def __init__(self):
    super().__init__()

  def forward(self, input, index):
    if isinstance(index, (list, tuple)):
      break_symbol = ':'
      symbols = ""
      for i in range(len(index)):
        if index[i] == None:
          slice_symbol = break_symbol
        else:
          slice_symbol = "index[" + str(i) + "]"
        if i > 0:
          symbols += "," + slice_symbol
        else:
          symbols = slice_symbol
      eval_str = f"input[{symbols}]"
      output = eval(eval_str)
      output = quantize_tensors([output], self.node)[0]
    else:
      output = input[index]
    return output


@py_utils.register_quant_op
def Index(*args, **kwargs):
  return deephi_Index(*args, **kwargs)


class deephi_IndexInputInplace(_PrimModule):
  def __init__(self):
    super().__init__()

  def forward(self, input, indices, values, accumulate):
    # TODO: try to remove hard code

    if any([len(index.tolist()) == 0 for index in indices if index is not None]):
      return input

    if indices[0] is None:
      input[:, indices[1]] = values
    elif len(indices) == 2 and indices[1] is None:
      input[indices[0], :] = values
    elif all([index is not None for index in indices]):
      input[indices] = values

    return input


@py_utils.register_quant_op
def index_put_inplace(*args, **kwargs):
  return deephi_IndexInputInplace(*args, **kwargs)


class deephi_ReLUk(_PrimModule):
  def __init__(self):
    super().__init__()

  def forward(self, input:torch.Tensor, channel_max:Union[torch.Tensor, Sequence[Any], float]):
    qinput = quantize_tensors([input], self.node, tensor_type='input')[0]

    if isinstance(channel_max, (list, tuple)):
      channel_max = torch.Tensor(channel_max).to(input.device)
    elif isinstance(channel_max, float):
      channel_max = torch.Tensor([channel_max]).to(input.device)
    if self.node.in_quant_part:
      channel_max = quant_reluk_params(self.node, channel_max)

    output = F.relu(input) - F.relu(qinput-channel_max)

    if self.node.in_quant_part:
      output = quantize_tensors([output], self.node)[0]

    return output

@py_utils.register_quant_op
def Reluk(*args, **kwargs):
  return deephi_ReLUk(*args, **kwargs)


class deephi_ChannelScale(_PrimModule):
  def __init__(self):
    super().__init__()

  def forward(self, input:torch.Tensor, channel_scale:Union[torch.Tensor, Sequence[Any], float]):
    qinput = quantize_tensors([input], self.node, tensor_type='input')[0]

    if isinstance(channel_scale, (list, tuple)):
      channel_scale = torch.Tensor(channel_scale).to(input.device)
    elif isinstance(channel_scale, float):
      channel_scale = torch.Tensor([channel_scale]).to(input.device)
    '''
    if self.node.in_quant_part:
      channel_scale = quant_channel_scale_params(self.node, channel_scale)
    '''
    output = qinput * channel_scale

    if self.node.in_quant_part:
      output = quantize_tensors([output], self.node)[0]

    return output

@py_utils.register_quant_op
def Channel_Scale(*args, **kwargs):
  return deephi_ChannelScale(*args, **kwargs)


class deephi_ExpandAs(_PrimModule):
  def __init__(self):
    super().__init__()

  def forward(self, input, other):
    qinput = quantize_tensors([input], self.node, tensor_type='input')[0]
    output = qinput.expand_as(other).clone()
    output = quantize_tensors([output], self.node)[0]

    return output


@py_utils.register_quant_op
def expand_as(*args, **kwargs):
  return deephi_ExpandAs(*args, **kwargs)


class deephi_Expand(_PrimModule):
  def __init__(self):
    super().__init__()

  def forward(self, input, size):
    qinput = quantize_tensors([input], self.node, tensor_type='input')[0]
    output = qinput.expand(size).clone()
    output = quantize_tensors([output], self.node)[0]

    return output


@py_utils.register_quant_op
def expand(*args, **kwargs):
  return deephi_Expand(*args, **kwargs)


class deephi_Correlation1D_Elemwise(_PrimModule):
  def __init__(self):
    super().__init__()

  def forward(self, input_1:torch.Tensor, input_2:torch.Tensor, pad_size:Union[torch.Tensor, Sequence[Any], int]):
    qinput_1 = quantize_tensors([input_1], self.node, tensor_type='input')[0]
    qinput_2 = quantize_tensors([input_2], self.node, tensor_type='input')[0]

    if isinstance(pad_size, (list, tuple)):
      pad_size = torch.Tensor(pad_size).to(qinput_1.device)
    elif isinstance(pad_size, float):
      pad_size = torch.Tensor([pad_size]).to(qinput_1.device)

    output_dim =  pad_size + 1
    B, C, H, W = qinput_1.size()
    qinput_2 = F.pad(qinput_2, pad=(pad_size,0,0,0), mode="constant",value=0) 
    cv = []
    for i in range(output_dim - 1, -1, -1):
        cost = qinput_1 * qinput_2[:, :, :, i:(i + W)]
        cost = cost.unsqueeze(2)
        cv.append(cost)
    output = torch.cat(cv, 2)

    if self.node.in_quant_part:
      output = quantize_tensors([output], self.node)[0]

    return output

@py_utils.register_quant_op
def Correlation1d_Elemwise(*args, **kwargs):
  return deephi_Correlation1D_Elemwise(*args, **kwargs)


class deephi_Correlation2D_Elemwise(_PrimModule):
  def __init__(self):
    super().__init__()

  def forward(self, input_1:torch.Tensor, input_2:torch.Tensor, pad_size:Union[torch.Tensor, Sequence[Any], int]):
    qinput_1 = quantize_tensors([input_1], self.node, tensor_type='input')[0]
    qinput_2 = quantize_tensors([input_2], self.node, tensor_type='input')[0]

    if isinstance(pad_size, (list, tuple)):
      pad_size = torch.Tensor(pad_size).to(qinput_1.device)
    elif isinstance(pad_size, float):
      pad_size = torch.Tensor([pad_size]).to(qinput_1.device)

    output_dim = 2 * pad_size + 1
    B, C, H, W = qinput_1.size()
    qinput_2 = F.pad(qinput_2, [pad_size] * 4)
    cv = []
    for i in range(output_dim):
        for j in range(output_dim):
            cost = qinput_1 * qinput_2[:, :, i:(i + H), j:(j + W)]
            cost = cost.unsqueeze(2)
            cv.append(cost)
    output = torch.cat(cv, 2)

    if self.node.in_quant_part:
      output = quantize_tensors([output], self.node)[0]

    return output

@py_utils.register_quant_op
def Correlation2d_Elemwise(*args, **kwargs):
  return deephi_Correlation2D_Elemwise(*args, **kwargs)


class deephi_CostVolume(_PrimModule):
  def __init__(self):
    super().__init__()

  def forward(self, input_1:torch.Tensor, input_2:torch.Tensor, maxdisp:Union[torch.Tensor, Sequence[Any], int]):
    qinput_1 = quantize_tensors([input_1], self.node, tensor_type='input')[0]
    qinput_2 = quantize_tensors([input_2], self.node, tensor_type='input')[0]
    if os.environ["DUMP_XMODEL"]=='1':
        cost = Variable(torch.zeros(qinput_1.size()[0], qinput_1.size()[1]*2, maxdisp//4,  qinput_1.size()[2],  qinput_1.size()[3])).cpu()
    else:
        cost = Variable(torch.zeros(qinput_1.size()[0], qinput_1.size()[1]*2, maxdisp//4,  qinput_1.size()[2],  qinput_1.size()[3])).cuda()
    
    for i in range(maxdisp//4):
        if i > 0 :
            cost[:, :qinput_1.size()[1], i, :,i:]   = qinput_1[:,:,:,i:]
            cost[:, qinput_1.size()[1]:, i, :,i:] = qinput_2[:,:,:,:-i]
        else:
            cost[:, :qinput_1.size()[1], i, :,:]   = qinput_1
            cost[:, qinput_1.size()[1]:, i, :,:]   = qinput_2
    output = cost.contiguous()

    if self.node.in_quant_part:
      output = quantize_tensors([output], self.node)[0]

    return output

@py_utils.register_quant_op
def CostVolume(*args, **kwargs):
  return deephi_CostVolume(*args, **kwargs)



class deephi_TupleUnpack(_PrimModule):

  def __init__(self):
    super().__init__()

  def forward(self, input):
    if len(self.node.out_tensors) == 1:
      output = input[0]
    else:
      output = []
      for i, tensor in enumerate(self.node.out_tensors):
        output.append(input[i])

    if self.node.in_quant_part:
      if isinstance(output, (tuple, list)):
        output = quantize_tensors(output, self.node)
      else:
        output = quantize_tensors([output], self.node)[0]

    return output


@py_utils.register_quant_op
def TupleUnpack(*args, **kwargs):
  return deephi_TupleUnpack(*args, **kwargs)
