from nndct_shared.base import NNDCT_CONSTANT
from nndct_shared.nndct_graph import Tensor

class OpDescriptor:

  @staticmethod
  def input(ctx, node, output_str):
    return "{} = args[{}]".format(output_str, int(node.name.split('_')[-1]))

  @staticmethod
  def rsub(ctx, node, output_str):
    other = node.node_config('other')
    if isinstance(other, Tensor):
      other = ctx.self.tensor_output_map(other.name, other.name)
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
  def default(ctx, node, output_str):
    return "{output} = {op_name}({inputs})".format(
        output=output_str,
        op_name=node.op.type,
        inputs=ctx._to_list_str(ctx._get_module_input(node)))
