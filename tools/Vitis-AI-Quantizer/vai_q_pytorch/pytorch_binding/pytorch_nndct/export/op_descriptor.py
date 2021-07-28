

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
from .code_template import CodeTemplate



class OpDescriptor(object):

  @staticmethod
  def input(ctx, node, output_str):
    return "{} = args[{}]".format(output_str, int(node.name.split('_')[-1]))
    
  @staticmethod
  def rsub(ctx, node, output_str):
    other = node.node_config('other')
    if isinstance(other, Tensor):
      other = ctx.get_output_tensor_name(other)
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
    start_symbol = []
    end_symbol = []
    step_symbol = []
    
    for i in range(len(starts)):
      start_symbol.append(ctx.infer_attr_value(starts[i]))
      end_symbol.append(ctx.infer_attr_value(ends[i]))
      step_symbol.append(ctx.infer_attr_value(steps[i]))
      
    for i in range(len(starts)):
      slice_symbol = break_symbol.join([start_symbol[i], end_symbol[i], step_symbol[i]])
      if i > 0:
        symbols += "," + slice_symbol
      else:
        symbols = slice_symbol
    # for i in range(len(starts)):
    #   start_symbol = str(starts[i]) if starts[i] > 0 else ''
    #   end_symbol = str(ends[i]) if ends[i] < NNDCT_CONSTANT.INT_MAX else ''
    #   step_symbol = ':' + str(steps[i]) if steps[i] > 1 else ''
    #   slice_symbol = start_symbol + break_symbol + end_symbol + step_symbol
    #   if i > 0:
    #     symbols += "," + slice_symbol
    #   else:
    #     symbols = slice_symbol
    input_str = ctx.infer_attr_value(node.node_config('input'))
    return "{output} = {input_tensor}[{symbols}]".format(
        output=output_str,
        input_tensor=input_str,
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
        inputs[idx] = ctx.get_output_tensor_name(ip)

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
        symbol = ctx.get_output_tensor_name(index)
      elif index is None:
        symbol = ":"

      if i > 0:
        indices += "," + symbol
      else:
        indices = symbol

    input = node.node_config('input')
    input_tensor = ctx.get_output_tensor_name(input)
    return "{output} = {input_tensor}[{symbols}]".format(
        output=output_str, input_tensor=input_tensor, symbols=indices)
  
  @staticmethod
  def strided_slice_inplace_copy(ctx, node, output_str):
  
    destination = node.node_config('destination')
    source = node.node_config('source')
    starts = node.node_config('start')
    ends = node.node_config('end')
    steps = node.node_config('step')
    break_symbol = ':'
    symbols = ""
    start_symbol = []
    end_symbol = []
    step_symbol = []
    for i in range(len(starts)):
      start_symbol.append(ctx.infer_attr_value(starts[i]))
      end_symbol.append(ctx.infer_attr_value(ends[i]))
      step_symbol.append(ctx.infer_attr_value(steps[i]))
      
    for i in range(len(starts)):
      if starts[i] == ends[i]:
        slice_symbol = start_symbol[i]
      else:
        slice_symbol = break_symbol.join([start_symbol[i], end_symbol[i], step_symbol[i]])
      if i > 0:
        symbols += "," + slice_symbol
      else:
        symbols = slice_symbol
  
    destination_str = ctx.infer_attr_value(destination)
    source_str = ctx.infer_attr_value(source)
    return "{output}[{symbols}] = {input_tensor}".format(
        output=destination_str,
        input_tensor=source_str, symbols=symbols)
    
  @staticmethod
  def index_put_inplace(ctx, node, output_str):
    # destination, _, source = ctx._get_module_input(node)
    destination = node.node_config('input')
    source = node.node_config('values')
    indices = node.node_config('indices')
    indices_symbol = ''
    sep_symbol = ','
    break_symbol = ':'
    for i, index in enumerate(indices):
      index = break_symbol if index is None else ctx.get_output_tensor_name(index)
      if i > 0:
        indices_symbol += sep_symbol + index 
      else:
        indices_symbol = index
    
    destination_str = ctx.infer_attr_value(destination)
    source_str = ctx.infer_attr_value(source)
    ctx.set_name_alias_for_output(output_str, destination_str)
    return "{output}[{symbols}] = {input_tensor}".format(
        output=destination_str,
        input_tensor=source_str, symbols=indices_symbol)  
    
  @staticmethod
  def loop(ctx, node, output_str):
    loop_pattern = None
    if node.node_config("is_while_loop"):
      raise NotImplementedError()
    else:
      loop_pattern = CodeTemplate("""$loop_outputs = $loop_vars
        for $iter_var in range(0, $max_trip_count):
            $block_inputs = $loop_outputs
            $body
            $loop_outputs = $body_ret
      """)
    loop_outputs = output_str
    loop_vars = node.node_config("initial_loop_vars")
    assert len(loop_vars) == len(ctx._get_module_output(node))
    
    def loop_var_to_str(var):
      if isinstance(var, list):
        start_str = '['
        end_str = ']'
        var_lst = []
        for ele in var:
          var_lst.append(loop_var_to_str(ele))
        return start_str + ",".join(var_lst) + end_str
      else:
        return ctx.get_output_tensor_name(var)
    
    loop_vars_str = ",".join([loop_var_to_str(var) for var in loop_vars]) 
      
    body_str = ""
    block_inputs_idx = 0
    iter_var_str = ''
    block_inputs = []
    max_trip_count = node.node_config("max_trip_count")
    if isinstance(max_trip_count, Tensor):
      max_trip_count = ctx.get_output_tensor_name(max_trip_count) 
        
    for inner_node in node.blocks[0].nodes:
      if inner_node.op.type == NNDCT_OP.INPUT:
        output_str = ctx._to_list_str(ctx._get_module_output(inner_node))
        if block_inputs_idx == 0:
          iter_var_str = output_str
        else:
          if isinstance(ctx._get_module_output(inner_node), list) and len(ctx._get_module_output(inner_node)) > 1:
            output_str = f"({output_str})"
          block_inputs.append(output_str)
        block_inputs_idx += 1
      else:
        forward_str, output_str = ctx._get_forward_str(inner_node)
        body_str += forward_str + '\n'
        
    block_inputs_str = ",".join(block_inputs)
    
    def get_ret_val_str(ret_val):
      if isinstance(ret_val, list):
        ret_val_str = ""
        head_str = "["
        tail_str = "]"
        for val in ret_val:
          ret_val_str += get_ret_val_str(val) + ","
        return head_str + ret_val_str + tail_str
      elif isinstance(ret_val, Tensor):
        return ctx.get_output_tensor_name(ret_val)
      
    body_ret_str = ",".join([get_ret_val_str(ret_val) for ret_val in node.blocks[0].return_struct[1:]])

    
    return loop_pattern.substitute(loop_outputs=loop_outputs, 
                                   loop_vars=loop_vars_str, 
                                   iter_var=iter_var_str,
                                   max_trip_count=max_trip_count,
                                   block_inputs=block_inputs_str,
                                   body=body_str,
                                   body_ret=body_ret_str)
    
  @staticmethod
  def list_add(ctx, node, output_str):
    inputs = node.node_config("input")
    others = node.node_config("other")
    input_str = ""
    if isinstance(inputs, list):
      input_str += "["
      for inp in inputs:
        input_str += ctx.get_output_tensor_name(inp)
      input_str += "]"
    else:
      input_str += ctx.get_output_tensor_name(inputs)
      
    
    others_str = ""
    if isinstance(others, list):
      others_str += "["
      for other in others:
        others_str += ctx.get_output_tensor_name(other)
      others_str += "]"
    else:
      others_str += ctx.get_output_tensor_name(others)
      
    return f"{output_str} = {input_str} + {others_str}"
       

  @staticmethod
  def floor_div(ctx, node, output_str):
    inputs = node.node_config("input")
    others = node.node_config("other")
    return f"{output_str} = {ctx.get_output_tensor_name(inputs)} // {ctx.get_output_tensor_name(others)}"
  
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
    NNDCT_OP.STRIDED_SLICE_INPLACE_COPY: OpDescriptor.strided_slice_inplace_copy,
    NNDCT_OP.INDEX_INPUT_INPLACE: OpDescriptor.index_put_inplace,
    NNDCT_OP.LOOP: OpDescriptor.loop,
    NNDCT_OP.LIST_ADD: OpDescriptor.list_add,
    NNDCT_OP.FLOOR_DIV: OpDescriptor.floor_div,
}
