

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
    dim = node.node_config("dim")
    starts = node.node_config('start')
    ends = node.node_config('end')
    steps = node.node_config('step')
    break_symbol = ':'
    symbols = ""
    start_symbol = []
    end_symbol = []
    step_symbol = []
    
    for i in range(dim[0]):
      start_symbol.append(str(0))
      end_symbol.append(str(NNDCT_CONSTANT.INT_MAX))
      step_symbol.append(str(1))
      
    for i in range(len(starts)):
      start_symbol.append(ctx.infer_attr_value(starts[i]))
      end_symbol.append(ctx.infer_attr_value(ends[i]))
      step_symbol.append(ctx.infer_attr_value(steps[i]))
      
    for i in range(len(start_symbol)):
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
  def _sequence(ctx, node, op_desc_str, output_str):
    inputs = node.op.get_config('input')
    for idx, ip in enumerate(inputs):
      if isinstance(ip, Tensor):
        inputs[idx] = ctx.get_output_tensor_name(ip)

    return "{output} = {op_name}([{inputs}])".format(
        output=output_str,
        op_name=op_desc_str,
        inputs=ctx._to_list_str(inputs))

  @staticmethod
  def list(ctx, node, output_str):
    return OpDescriptor._sequence(ctx, node, 'list', output_str)

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
    dims = node.node_config('dim')
    break_symbol = ':'
    symbols = ""
    start_symbol = []
    end_symbol = []
    step_symbol = []
    
    for i in range(dims[0]):
      start_symbol.append(str(0))
      end_symbol.append(str(NNDCT_CONSTANT.INT_MAX))
      step_symbol.append(str(1))

    for i in range(len(starts)):
      start_symbol.append(ctx.infer_attr_value(starts[i]))
      end_symbol.append(ctx.infer_attr_value(ends[i]))
      step_symbol.append(ctx.infer_attr_value(steps[i]))
      
    for i in range(len(start_symbol)):
      if start_symbol[i] == end_symbol[i]:
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
    loop_outputs = output_str
    loop_vars = node.node_config("initial_loop_vars")
    loop_vars_str = ctx.infer_attr_value(loop_vars[0] if len(loop_vars) == 1 else loop_vars)
    assert len(loop_vars) == len(ctx._get_module_output(node))
    init_condition_str = ctx.infer_attr_value(node.node_config("initial_condition"))
      
    body_str = ""
    block_inputs_idx = 0
    iter_var_str = ''
    block_inputs = []
    iter_start_str = str(0)
    max_trip_count = node.node_config("max_trip_count")
    max_trip_count_str = ctx.infer_attr_value(max_trip_count)
        
    for inner_node in node.blocks[0].nodes:
      if inner_node.op.type == NNDCT_OP.RETURN:
        continue
      if inner_node.op.type in [NNDCT_OP.INPUT, NNDCT_OP.TUPLE_INPUT]:
        output_str = ctx._to_list_str(ctx._get_module_output(inner_node))
        if block_inputs_idx == 0:
          iter_var_str = output_str
        else:
          if isinstance(ctx._get_module_output(inner_node), list) and len(ctx._get_module_output(inner_node)) > 1:
            output_str = f"({output_str})"
          block_inputs.append(output_str)
        block_inputs_idx += 1
      elif inner_node.op.type == NNDCT_OP.DERIVE_LOOP_INDEX:
        iter_start_str = str(inner_node.node_config("start"))
        output_str = ctx._to_list_str(ctx._get_module_output(inner_node))
        iter_var_str = output_str
      else:
        
        forward_str, output_str = ctx._get_forward_str(inner_node)
        body_str += forward_str + '\n'
        
    block_inputs_str = ",".join(block_inputs)
    body_ret_str = ctx.infer_attr_value(node.blocks[0].return_node.node_config("input")[1:])

    # body_ret_str = ",".join([ctx.infer_attr_value(ret_val) for ret_val in node.blocks[0].return_node.in_tensors[1:]])
    iter_end_str = "+".join([max_trip_count_str, iter_start_str])
    iter_conditon_str = ctx.infer_attr_value(node.blocks[0].return_node.in_tensors[0])
    loop_pattern = None
    if node.node_config("is_while_loop"):
      
      loop_pattern = CodeTemplate("""\
$loop_outputs = $loop_vars
condition = $initial_condition
while condition:
    $block_inputs = $loop_outputs
    $body
    $loop_outputs = $body_ret
    condition = $iter_condition
      """)
      return loop_pattern.substitute(loop_outputs=loop_outputs, 
                                     loop_vars=loop_vars_str,
                                     initial_condition=init_condition_str,
                                     block_inputs=block_inputs_str,
                                     body = body_str,
                                     body_ret = body_ret_str,
                                     iter_condition=iter_conditon_str)
    else:
      loop_pattern = CodeTemplate("""\
$loop_outputs = $loop_vars
for $iter_var in range($iter_start, $iter_end):
    $block_inputs = $loop_outputs
    $body
    $loop_outputs = $body_ret
  """)
      return loop_pattern.substitute(loop_outputs=loop_outputs, 
                                    loop_vars=loop_vars_str, 
                                    iter_var=iter_var_str,
                                    iter_start=iter_start_str,
                                    iter_end=iter_end_str,
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
    input_str = ctx.infer_attr_value(node.node_config("input"))
    other_str = ctx.infer_attr_value(node.node_config("other"))
    return f"{output_str} = {input_str} // {other_str}"
  
  @staticmethod
  def sequence_unpack(ctx, node, output_str):
    if len(node.out_tensors) == 1:
      return f"{output_str}, = {ctx._to_list_str(ctx._get_module_input(node))}"
    else:
      return f"{output_str} = {ctx._to_list_str(ctx._get_module_input(node))}"


  @staticmethod
  def slice(ctx, node, output_str):
    start = node.node_config('start')
    end = node.node_config('end')
    step = node.node_config('step')
    dim = node.node_config('dim')
    break_symbol = ':'
    symbols = ""
    starts = []
    ends = []
    steps = []
    
    for i in range(dim + 1):
      if i != dim:
        starts.append(str(0))
        ends.append(str(NNDCT_CONSTANT.INT_MAX))
        steps.append(str(1))
      else:
        starts.append(ctx.infer_attr_value(start))
        ends.append(ctx.infer_attr_value(end))
        steps.append(ctx.infer_attr_value(step))
        
    for i in range(dim + 1):
      slice_symbol = break_symbol.join([starts[i], ends[i], steps[i]])
      if i > 0:
        symbols += "," + slice_symbol
      else:
        symbols = slice_symbol

    input_str = ctx.infer_attr_value(node.node_config("input"))
    return "{output} = {input_tensor}[{symbols}]".format(
      output=output_str,
      input_tensor=input_str,
      symbols=symbols)
  
  @staticmethod
  def length(ctx, node, output_str):
    return "{output} = len({input})".format(output=output_str, input=ctx._to_list_str(ctx._get_module_input(node)))
  
  
  @staticmethod
  def If(ctx, node, output_str):
    if_pattern = CodeTemplate("""\
if ($condition):
    $block_0_body
    $if_out = $ret_0
else:
    $block_1_body
    $if_out = $ret_1
    """)
    if_out_str = output_str
    condition_str = ctx.infer_attr_value(node.node_config("condition"))
    assert len(node.blocks) == 2
    blocks = [""] * 2
    block_ret = [""] * 2
    for i, block in enumerate(node.blocks):
      for inner_node in block.nodes:
        if inner_node.op.type == NNDCT_OP.RETURN:
          continue
        forward_str, output_str = ctx._get_forward_str(inner_node)
        blocks[i] += forward_str + '\n'
      
      block_ret[i] = ",".join([ctx.infer_attr_value(ret_val) for ret_val in block.return_node.in_tensors])

    block_0_body, block_1_body = blocks
    ret_0_str, ret_1_str = block_ret
    return if_pattern.substitute(condition=condition_str, 
                                 block_0_body=block_0_body, 
                                 block_1_body=block_1_body,
                                 if_out=if_out_str,
                                 ret_0=ret_0_str,
                                 ret_1=ret_1_str 
                                 )
    
    
    
    
  @staticmethod
  def lt(ctx, node, output_str):
    input_str = ctx.infer_attr_value(node.node_config("input"))
    other_str = ctx.infer_attr_value(node.node_config("other"))
    return "{output} = {input} < {other}".format(output=output_str, input=input_str, other=other_str)
     
  
  @staticmethod
  def eq(ctx, node, output_str):
    input_str = ctx.infer_attr_value(node.node_config("input"))
    other_str = ctx.infer_attr_value(node.node_config("other"))
    return "{output} = {input} == {other}".format(output=output_str, input=input_str, other=other_str)

  @staticmethod
  def return_(ctx, node, output_str):
    return_str = ctx.infer_attr_value(node.node_config("input"))
    return "return {}".format(return_str)

  @staticmethod
  def tuple_index(ctx, node, output_str):
    input_str = ctx.infer_attr_value(node.node_config("input"))
    index_str = ctx.infer_attr_value(node.node_config("index"))
    return "{output} = {input}[{index}]".format(output=output_str, input=input_str, index=index_str)


  @staticmethod
  def _tuple(ctx, node, output_str):
    return OpDescriptor._sequence(ctx, node, output_str)

  @staticmethod
  def device(ctx, node, output_str):
    input_str = ctx.infer_attr_value(node.node_config("input"))
    return "{output} = {input}.device".format(output=output_str, input=input_str)
    return op

  
  @staticmethod
  def dtype(ctx, node, output_str):
    input_str = ctx.infer_attr_value(node.node_config("input"))
    return "{output} = {input}.dtype".format(output=output_str, input=input_str)
    return op


  @staticmethod
  def int_(ctx, node, output_str):
    return "{output} = {op_name}({inputs})".format(
        output=output_str,
        op_name="int",
        inputs=ctx._to_list_str(ctx._get_module_input(node)))

  @staticmethod
  def constant_with_reshape(ctx, node, output_str):
    data_str = ctx.infer_attr_value(node.node_config("data"))
    data_shape_str = ctx.infer_attr_value(node.node_config("data_shape"))
    dtype_str = ctx.infer_attr_value(node.node_config("dtype"))
    device_str = ctx.infer_attr_value(node.node_config("device"))
    inputs = "torch.tensor(data=" + data_str + ", dtype=" + dtype_str + ", device=" + device_str + ").reshape(" + data_shape_str + ")"
    return "{output} = {inputs}".format(output=output_str, inputs=inputs)
    


MISC_OP_DISCR_MAP = {
    NNDCT_OP.INPUT: OpDescriptor.input,
    NNDCT_OP.TUPLE_INPUT: OpDescriptor.input,
    NNDCT_OP.SLICE_TENSOR_INPLACE_COPY: OpDescriptor.slice_tensor_inplace_copy,
    NNDCT_OP.INDEX: OpDescriptor.index,
    NNDCT_OP.INT: OpDescriptor.int_,
    NNDCT_OP.STRIDED_SLICE_INPLACE_COPY: OpDescriptor.strided_slice_inplace_copy,
    NNDCT_OP.INDEX_INPUT_INPLACE: OpDescriptor.index_put_inplace,
    NNDCT_OP.LOOP: OpDescriptor.loop,
    NNDCT_OP.LIST_ADD: OpDescriptor.list_add,
    NNDCT_OP.FLOOR_DIV: OpDescriptor.floor_div,
    NNDCT_OP.TUPLE_UNPACK: OpDescriptor.sequence_unpack,
    NNDCT_OP.SLICE: OpDescriptor.slice,
    NNDCT_OP.LENGTH: OpDescriptor.length,
    NNDCT_OP.IF: OpDescriptor.If,
    NNDCT_OP.SCALAR_LESS_THAN: OpDescriptor.lt,
    NNDCT_OP.SCALAR_EQUAL: OpDescriptor.eq,
    NNDCT_OP.RETURN: OpDescriptor.return_,
    NNDCT_OP.LIST: OpDescriptor.list,
    NNDCT_OP.TUPLE: OpDescriptor._tuple,
    NNDCT_OP.TUPLE_INDEX: OpDescriptor.tuple_index,
    NNDCT_OP.DEVICE: OpDescriptor.device,
    NNDCT_OP.DTYPE: OpDescriptor.dtype,
    NNDCT_OP.CONSTANT_WITH_RESHAPE: OpDescriptor.constant_with_reshape
}
