

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

import torch
import torch.nn.functional as F
from nndct_shared.utils import NndctOption, NndctScreenLogger
from nndct_shared.base import NNDCT_CONSTANT
from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import process_inputs_and_params
from nndct_shared.quantization import post_quant_process
from nndct_shared.quantization import quant_reluk_params
from nndct_shared.quantization import quant_channel_scale_params
import pytorch_nndct.utils as py_utils
from typing import Any, Optional, Sequence, Union
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
    [input], _ = process_inputs_and_params(
        self.node,
        self.quantizer,
        inputs=[input],
    )

    output = input
    if NndctOption.nndct_stat.value > 2:
      print('Channel number of input data: {}'.format(output.shape[1]))
      print('Input data histogram: {}'.format( output.histc(bins = 10).cpu().detach().numpy() ))
      print('Network input channel-wise statistic [Min, Max, Mean, Std]:')
      t = output.transpose(0, 1)
      for c in range(t.shape[0]):
        print('[{}, {}, {}, {}]'.format( t[c].min(), t[c].max(), t[c].mean(), t[c].std() ))
        print('histogram: {}'.format( t[c].histc(bins = 10).cpu().detach().numpy() ))

    [output] = post_quant_process(self.node, [output])
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
    [input], _ = process_inputs_and_params(
        self.node,
        self.quantizer,
        inputs=[input],
    )
    # check input shape
    if self.node.out_tensors[0].is_complete_tensor() and self.node.out_tensors[0].ndim == 4:
      # py_utils.blob_to_torch_format(self.node.out_tensors[0])
      if not (self.node.out_tensors[0].shape[1:] == list(input.size())[1:]):
        NndctScreenLogger().warning(f"The shape of input ({input.shape[1:]}) should be the same with that of dummy input ({self.node.out_tensors[0].shape[1:]})")
      # py_utils.blob_to_nndct_format(self.node.out_tensors[0])
    output = input

    if (self.node.in_quant_part and NndctOption.nndct_stat.value > 2):
      print('Channel number of input data: {}'.format(output.shape[1]))
      print('Input data histogram: {}'.format( output.histc(bins = 10).cpu().detach().numpy() ))
      print('Network input channel-wise statistic [Min, Max, Mean, Std]:')
      t = output.transpose(0, 1)
      for c in range(t.shape[0]):
        print('[{}, {}, {}, {}]'.format( t[c].min(), t[c].max(), t[c].mean(), t[c].std() ))
        print('histogram: {}'.format( t[c].histc(bins = 10).cpu().detach().numpy() ))

    if self.node.in_quant_part:
      [output] = post_quant_process(self.node, [output])

    return output


@py_utils.register_quant_op
def Input(*args, **kwargs):
  return deephi_Input(*args, **kwargs)


class deephi_StridedSlice(_PrimModule):
  def __init__(self):
    super().__init__()

  def forward(self, input, start, end, step):
    size = input.size()
    for i in range(len(start)):
      if end[i] == NNDCT_CONSTANT.INT_MAX:
        end[i] = size[i]
      elif end[i] < 0:
        end[i] = size[i] + end[i]
      indices = torch.arange(start[i], end[i], step[i]).to(input.device)
      input = torch.index_select(input, i, indices)

    output = input

    [output] = post_quant_process(self.node, [output])

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
    [output] = post_quant_process(self.node, [output])
    return output


@py_utils.register_quant_op
def slice_tensor_inplace_copy(*args, **kwargs):
  return deephi_SliceInplaceCopy(*args, **kwargs)


class deephi_Index(_PrimModule):
  def __init__(self):
    super().__init__()

  def forward(self, input, index):
    if isinstance(index, (list, tuple)) and len(index) == 2:
      if index[0] is None:
        output = input[:, index[1]]
      elif index[1] is None:
        output = input[index[0], :]
      else:
        output = input[index[0], index[1]]
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
    [input], _ = process_inputs_and_params(
        self.node,
        self.quantizer,
        inputs=[input],
    )

    if isinstance(channel_max, (list, tuple)):
      channel_max = torch.Tensor(channel_max).to(input.device)
    elif isinstance(channel_max, float):
      channel_max = torch.Tensor([channel_max]).to(input.device)
    if self.node.in_quant_part:
      channel_max = quant_reluk_params(self.node, channel_max)

    output = F.relu(input) - F.relu(input-channel_max)

    if self.node.in_quant_part:
      [output] = post_quant_process(self.node, [output])

    return output

@py_utils.register_quant_op
def Reluk(*args, **kwargs):
  return deephi_ReLUk(*args, **kwargs)


class deephi_ChannelScale(_PrimModule):
  def __init__(self):
    super().__init__()

  def forward(self, input:torch.Tensor, channel_scale:Union[torch.Tensor, Sequence[Any], float]):
    [input], _ = process_inputs_and_params(
        self.node,
        self.quantizer,
        inputs=[input],
    )

    if isinstance(channel_scale, (list, tuple)):
      channel_scale = torch.Tensor(channel_scale).to(input.device)
    elif isinstance(channel_scale, float):
      channel_scale = torch.Tensor([channel_scale]).to(input.device)
    '''
    if self.node.in_quant_part:
      channel_scale = quant_channel_scale_params(self.node, channel_scale)
    '''
    output = input * channel_scale

    if self.node.in_quant_part:
      [output] = post_quant_process(self.node, [output])

    return output

@py_utils.register_quant_op
def Channel_Scale(*args, **kwargs):
  return deephi_ChannelScale(*args, **kwargs)


class deephi_ExpandAs(_PrimModule):
  def __init__(self):
    super().__init__()

  def forward(self, input, other):
    [input], _ = process_inputs_and_params(
        self.node,
        self.quantizer,
        inputs=[input],
    )
    output = input.expand_as(other).clone()

    [output] = post_quant_process(self.node, [output])

    return output


@py_utils.register_quant_op
def expand_as(*args, **kwargs):
  return deephi_ExpandAs(*args, **kwargs)


class deephi_Expand(_PrimModule):
  def __init__(self):
    super().__init__()

  def forward(self, input, size):
    [input], _ = process_inputs_and_params(
        self.node,
        self.quantizer,
        inputs=[input],
    )
    output = input.expand(size).clone()

    [output] = post_quant_process(self.node, [output])

    return output


@py_utils.register_quant_op
def expand(*args, **kwargs):
  return deephi_Expand(*args, **kwargs)