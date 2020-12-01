

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

from nndct_shared.base import NNDCT_CONSTANT, NNDCT_OP
from nndct_shared.nndct_graph import Tensor


class OpDescriptor(object):

  @staticmethod
  def input(ctx, node, output_str):
    return "{} = args[{}]".format(output_str, int(node.name.split('_')[-1]))
    
  @staticmethod
  def rsub(ctx, node, output_str):
    other = node.node_config('other')
    if isinstance(other, Tensor):
      other = ctx.tensor_output_map.get(other.name, other.name)
    return "{output} = {other} - {input}".format(
        output=output_str,
        other=other,
        input=ctx._to_list_str(ctx._get_module_input(node)))

  @staticmethod
  def strided_slice(ctx, node, output_str):
    starts = node.node_config('start')
    ends = node.node_config('end')
    steps = node.node_config('step')
    break_symbol = ':'
    symbols = ""
    for i in range(len(starts)):
      start_symbol = str(starts[i]) if starts[i] > 0 else ''
      end_symbol = str(ends[i]) if ends[i] < NNDCT_CONSTANT.INT_MAX else ''
      step_symbol = ':' + str(steps[i]) if steps[i] > 1 else ''
      slice_symbol = start_symbol + break_symbol + end_symbol + step_symbol
      if i > 0:
        symbols += "," + slice_symbol
      else:
        symbols = slice_symbol

    return "{output} = {input_tensor}[{symbols}]".format(
        output=output_str,
        input_tensor=ctx._to_list_str(ctx._get_module_input(node)),
        symbols=symbols)

  @staticmethod
  def slice_tensor_inplace_copy(ctx, node, output_str):
    slice_tensor, input = ctx._get_module_input(node)
    dim = node.node_config('dim')
    index = node.node_config('index')
    symbols = str(index)
    for i in range(dim):
      symbols = ','.join([':', symbols])
    return "{slice_tensor}[{symbols}] = {input_tensor}".format(
        slice_tensor=slice_tensor, symbols=symbols, input_tensor=input)

  @staticmethod
  def _sequence(ctx, node, output_str):
    inputs = node.op.get_config('input')
    for idx, ip in enumerate(inputs):
      if isinstance(ip, Tensor):
        inputs[idx] = ctx.tensor_output_map[ip.name]

    return "{output} = {op_name}([{inputs}])".format(
        output=output_str,
        op_name=node.op.type,
        inputs=ctx._to_list_str(inputs))

  @staticmethod
  def list(ctx, node, output_str):
    return OpDescriptor._sequence(ctx, node, output_str)

  @staticmethod
  def index(ctx, node, output_str):

    indices = ""
    for i, index in enumerate(node.node_config('index')):
      if isinstance(index, Tensor):
        symbol = ctx.tensor_output_map.get(index.name, index.name)
      elif index is None:
        symbol = ":"

      if i > 0:
        indices += "," + symbol
      else:
        indices = symbol

    input = node.node_config('input')
    input_tensor = ctx.tensor_output_map.get(input.name, input.name)
    return "{output} = {input_tensor}[{symbols}]".format(
        output=output_str, input_tensor=input_tensor, symbols=indices)
  
  @staticmethod
  def strided_slice_inplace_copy(ctx, node, output_str):
    destination, source = ctx._get_module_input(node)
    starts = node.node_config('start')
    ends = node.node_config('end')
    steps = node.node_config('step')
    break_symbol = ':'
    symbols = ""
    for i in range(len(starts)):
      start_symbol = str(starts[i]) if starts[i] > 0 else ''
      end_symbol = str(ends[i]) if ends[i] < NNDCT_CONSTANT.INT_MAX else ''
      step_symbol = ':' + str(steps[i]) if steps[i] > 1 else ''
      slice_symbol = start_symbol + break_symbol + end_symbol + step_symbol
      if i > 0:
        symbols += "," + slice_symbol
      else:
        symbols = slice_symbol

    return "{output}[{symbols}] = {input_tensor}".format(
        output=destination,
        input_tensor=source, symbols=symbols)
   
  
  @staticmethod
  def default(ctx, node, output_str):
    return "{output} = {op_name}({inputs})".format(
        output=output_str,
        op_name=node.op.type,
        inputs=ctx._to_list_str(ctx._get_module_input(node)))


MISC_OP_DISCR_MAP = {
    NNDCT_OP.INPUT: OpDescriptor.input,
    NNDCT_OP.RSUB: OpDescriptor.rsub,
    NNDCT_OP.STRIDED_SLICE: OpDescriptor.strided_slice,
    NNDCT_OP.SLICE_TENSOR_INPLACE_COPY: OpDescriptor.slice_tensor_inplace_copy,
    NNDCT_OP.INDEX: OpDescriptor.index,
    NNDCT_OP.INT: OpDescriptor.default,
    NNDCT_OP.STRIDED_SLICE_INPLACE_COPY: OpDescriptor.strided_slice_inplace_copy
}
