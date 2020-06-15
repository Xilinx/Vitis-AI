from nndct_shared.base.key_names import FrameworkType
from nndct_shared.nndct_graph import Graph, Tensor
from nndct_shared.utils import NndctDebugLogger, NndctOption

from .op_dispatcher import *
from .trace_helper import TorchGraphHandler
from .utils import *


class TorchParser(object):

  def __call__(self, graph_name, module, input_args):
    torch_graph_handler = TorchGraphHandler()
    raw_graph = torch_graph_handler.build_torch_graph(graph_name, module, input_args)
    self._nndct_graph = Graph(graph_name=raw_graph.name)
    node_convertor = NodeConvertor()
    op_creator = OpCreator()
    for raw_node in raw_graph.nodes:
      nndct_node = node_convertor(self, raw_graph, raw_node)
      if nndct_node:
        self._nndct_graph.add_node(nndct_node)
        nndct_node.op = op_creator(self, raw_graph, raw_node)

    for ret_value_name in raw_graph.ret_values().keys():
      end_tensor = self._nndct_graph.tensor(get_full_name(self._nndct_graph.name, ret_value_name))
      self._nndct_graph.add_end_tensor(end_tensor)

    self._convert_blob_tensor_type()
    self._nndct_graph.connect_nodes()
    self._load_data(module)
    if NndctOption.nndct_parse_debug.value >= 2:
      NndctDebugLogger.write(f"nndct raw graph:\n{self._nndct_graph}")
    # print(f"nndct graph:{self._nndct_graph}")
    return self._nndct_graph

  def _convert_blob_tensor_type(self):
    r"""convert torch tensor info to nndct tensor info"""
    for blob_tensor in self._nndct_graph.tensors.values():
      tensor_util.convert_blob_tensor_format(blob_tensor,
                                             tensor_util.FrameworkType.TORCH,
                                             tensor_util.FrameworkType.NNDCT)
      blob_tensor.dtype = convert_dtype(blob_tensor.dtype)

  def _load_data(self, module):
    for node in self._nndct_graph.nodes: 
      if node.op.type in [NNDCT_OP.BASIC_LSTM, NNDCT_OP.BASIC_GRU]:
        for nndct_param, param_tensors in node.op.params.items():
          for tensor in param_tensors:
            data = module.state_dict()[get_short_name(tensor.name)].cpu().numpy()
            tensor.from_ndarray(data)
            tensor = tensor_util.convert_parameter_tensor_format(
              tensor, FrameworkType.TORCH, FrameworkType.NNDCT)
        #combine bias_ih and bias_hh item
        
        if node.op.type == NNDCT_OP.BASIC_LSTM:
          for bias_term in [node.op.ParamName.BIAS, node.op.ParamName.BIAS_REVERSE]:
            if bias_term in node.op.params and len(node.op.params[bias_term]) > 0:
              if len(node.op.params[bias_term]) % 2 != 0:
                raise RuntimeError("The num of bias should be even")
              i = 0
              bias_list = []
              while i != len(node.op.params[bias_term]):
                bias_ih = node.op.params[bias_term][i]
                bias_hh = node.op.params[bias_term][i + 1]
                tensor_name = f"bias_{i//2}" if bias_term == node.op.ParamName.BIAS else f"bias_{i//2}_reverse"
                bias = Tensor(name=get_full_name(self._nndct_graph.name, tensor_name), data=bias_ih.data + bias_hh.data)
                bias_list.append(bias)
                i = i + 2
              node.op.set_param(bias_term, bias_list)
      elif node.op.type == NNDCT_OP.CONVTRANSPOSE2D:
        for param_name, tensor in node.op.params.items():
          data = module.state_dict()[get_short_name(tensor.name)].cpu().numpy()
          if param_name == node.op.ParamName.WEIGHTS:
            data = np.copy(data).transpose(1, 0, 2, 3)
            data = np.ascontiguousarray(data)

          tensor.from_ndarray(data)
          tensor = tensor_util.convert_parameter_tensor_format(
              tensor, FrameworkType.TORCH, FrameworkType.NNDCT)
          
      elif node.op.type == NNDCT_OP.DEPTHWISE_CONV2D:
        for param_name, tensor in node.op.params.items():
          data = module.state_dict()[get_short_name(tensor.name)].cpu().numpy()
          if param_name == node.op.ParamName.WEIGHTS:
            in_channels = node.node_config("in_channels")
            out_channels = node.node_config("out_channels")
            kernel_size = node.node_config("kernel_size")
            channel_mutiplier = int(out_channels/in_channels)
            data = np.copy(data).reshape((channel_mutiplier, in_channels, *kernel_size))
        
          tensor.from_ndarray(data)
          tensor = tensor_util.convert_parameter_tensor_format(
              tensor, FrameworkType.TORCH, FrameworkType.NNDCT)
      else:
        for param_name, tensor in node.op.params.items():
          data = module.state_dict()[get_short_name(tensor.name)].cpu().numpy()
          tensor.from_ndarray(data)
          tensor = tensor_util.convert_parameter_tensor_format(
              tensor, FrameworkType.TORCH, FrameworkType.NNDCT)

  def get_blob_tensor_by_name(self, name):
    name = get_full_name(self._nndct_graph.name,name)
    return self._nndct_graph.tensor(name)

  def get_nndct_value(self, torch_value):
    r"""
    three simple types of value : nndct tensor/plain value/None
    """

    def _get_simple_value(value):
      if value.is_none():
        return None
      elif value.is_plain_value():
        return value.data
      else:
        return self.get_blob_tensor_by_name(value.name)

    if isinstance(torch_value, list):
      return [_get_simple_value(value) for value in torch_value]
    else:
      return _get_simple_value(torch_value)
