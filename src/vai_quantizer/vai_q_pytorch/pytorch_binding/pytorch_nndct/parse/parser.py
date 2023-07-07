

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


from collections import ChainMap, defaultdict
from enum import Enum
from tqdm import tqdm
import torch

from nndct_shared.base.key_names import FrameworkType
from nndct_shared.nndct_graph import (Graph, Tensor, Block, Node,
                                      reorder_multi_subgraph_nodes)
from nndct_shared.utils import NndctDebugLogger, NndctOption, NndctScreenLogger, QError, QWarning, QNote
from pytorch_nndct.utils import build_aten_torch_ops_table

from .op_dispatcher import *
from .parse_utils import *
from .torch_op_def import TorchUnknownOperation
from .trace_helper import TorchGraphHandler


def unknown_op_type_check(graph: Graph):
  unkown_ops = set()
  custom_ops = set()
  for node in graph.all_nodes():
    if isinstance(node.op, TorchUnknownOperation):
      unkown_ops.add(node.op.type)
    elif node.has_custom_op():
      custom_ops.add(node.op.type)        
  for op in custom_ops:
      NndctScreenLogger().warning2user(QWarning.FLOAT_OP, f"The quantizer recognize new op `{op}` as a float operator by default.")

#   if custom_ops:
#     NndctScreenLogger().info(f"You can make these new ops quantizable by add them to custom_quant_ops, \
# e.g. quantizer= torch_quantizer(..., custom_quant_ops=['{list(custom_ops)[0]}',...])")

  NndctScreenLogger().check2user(QError.UNSUPPORTED_OPS, f"Unsupported Ops: {unkown_ops}.", len(unkown_ops)==0)

class TorchParser(object):
  def __init__(self):
    self.visited_blob_tensors = {}
    self.visited_param_tensors = {}
    self.node_input_args = defaultdict(list)
    self.node_params = defaultdict(list)
    self.cur_graph = None
    self.cur_block = None
    # self.converted_node = set()
    build_aten_torch_ops_table()
  
  
 
  
  def __call__(self, graph_name, module, input_args):
    graph_handler = TorchGraphHandler()
    # graph_handler = create_graph_handler(module)
    raw_graph = graph_handler.build_torch_graph(graph_name, module, input_args) 
    GLOBAL_MAP.set_map(NNDCT_KEYS.DEVICE, self._get_device_info(module, input_args))
    NndctScreenLogger().info("Processing ops...")
    nndct_graph = self._convert_graph(raw_graph)
    unknown_op_type_check(nndct_graph)  
    self._convert_blob_tensor_type(nndct_graph)
    self._load_data(nndct_graph, module)
    if NndctOption.nndct_parse_debug.value >= 2:
      NndctDebugLogger.write(f"nndct raw graph:\n{nndct_graph}")
    return nndct_graph
  
  def _convert_params(self, raw_graph):
    for value in raw_graph.param_values():
      param_tensor = self._convert_tensor(value)
      self.cur_graph.add_param_name(param_tensor.name)
      self.cur_graph.add_tensor(param_tensor)
      
  def _bind_free_params(self, nndct_node):
    def unpack_op_params(params):
      unpacked_params = []
      for param in params:
        if isinstance(param, list):
          unpacked_params.extend(param)
        else:
          unpacked_params.append(param)
      return unpacked_params
      
    for param_tensor in self.node_params[nndct_node]:
      if param_tensor not in unpack_op_params(list(nndct_node.op.params.values())):
        param_name = get_formal_name(param_tensor.name)
        param_type = Enum(param_name, [(param_name, param_name)])
        nndct_node.op.set_param(param_type[param_name], param_tensor)
  
  
        
  def _convert_graph(self, raw_graph):
    nndct_graph = Graph(graph_name=raw_graph.name)
    self.cur_graph = nndct_graph
    graph_input = self._convert_node(raw_graph.head_node)
    graph_return = self._convert_node(raw_graph.return_node)
    top_block = Block(nndct_graph, None, graph_input, graph_return)
    self.cur_block = top_block
    nndct_graph.set_top_block(top_block)
    
    self._convert_params(raw_graph)
    
    pbar = tqdm(list(raw_graph.nodes), bar_format="{bar:50}{r_bar}")
    for raw_node in pbar:
      pbar.set_postfix_str(f"OpInfo: name = {raw_node.name}, type = {raw_node.kind}")
      pbar.update()
      nndct_node = self._convert_node(raw_node)
      if not nndct_node.in_node_list():
        nndct_graph.append_node(nndct_node)

      for sub_block in raw_node.blocks:
        cur_block = self.cur_block
        self.cur_block = None
        node_block = self._convert_block(nndct_graph, nndct_node, sub_block)
        self.cur_block = cur_block
        nndct_node.add_block(node_block)    
      self._bind_free_params(nndct_node)
      
    # for node in nndct_graph.nodes:
    #   print(node.name, node.in_nodes, node.out_nodes, node.topo_position)
    return nndct_graph
  
  def _convert_block(self, nndct_graph, nndct_block_node, raw_block):
    block_input = self._convert_node(raw_block.head_node, nndct_graph.name)
    block_return = self._convert_node(raw_block.return_node, nndct_graph.name)
    nndct_block = Block(nndct_graph, nndct_block_node, block_input, block_return)
    self.cur_block = nndct_block
    for raw_node in raw_block.nodes:
      nndct_node = self._convert_node(raw_node)
      if not nndct_node.in_node_list():
        nndct_block.append_node(nndct_node)
      for raw_block in raw_node.blocks:
        cur_block = self.cur_block
        self.cur_block = None
        node_block = self._convert_block(nndct_graph, nndct_node, raw_block)
        self.cur_block = cur_block
        nndct_node.add_block(node_block)
      self._bind_free_params(nndct_node)
      
    return nndct_block
      
  
  def _convert_node(self, raw_node, scope=None):
    if scope is None:
      assert self.cur_graph 
      node_scope = self.cur_graph.name
    else:
      node_scope = scope
      
    nndct_node = Node(
        name=get_full_name(node_scope, raw_node.name),
        dtype=self.convert_dtype(raw_node.dtype),
        )
    nndct_node.source_range = raw_node.source_range
    nndct_node.scope_name = raw_node.scope_name
    if nndct_node.name in self.cur_graph:
      return self.cur_graph.node(nndct_node.name)
  
    # nndct_node.raw_kind = raw_node.kind
    # self.converted_node.add(raw_node)
    nndct_node.schema = raw_node.schema
    nndct_node.is_custom_extension = raw_node.is_custom_pyop
    nndct_node.caller = raw_node.pyobj
    nndct_node.owning_block = self.cur_block
    nndct_node.owning_graph = self.cur_graph
    for out in raw_node.outputs:
      full_name = get_full_name(node_scope, out.name)
      if self.cur_graph and self.cur_graph.is_tensor_in_graph(full_name):
        nndct_node.add_out_tensor(self.cur_graph.tensor(full_name))
      else:
        nndct_tensor = self._convert_tensor(out, node_scope)
        nndct_node.add_out_tensor(nndct_tensor)

    for ip in raw_node.flatten_inputs:
      if ip.name is None:
          continue
      full_name = get_full_name(node_scope, ip.name) 
      if self.cur_graph and self.cur_graph.is_tensor_in_graph(full_name):
        nndct_node.add_in_tensor(self.cur_graph.tensor(full_name))
      elif not raw_node.outputs:
        # For Return node
        nndct_tensor = self._convert_tensor(ip, node_scope)
        nndct_node.add_in_tensor(nndct_tensor)
        
      if self.cur_graph and full_name in self.cur_graph.param_names():
        self.node_params[nndct_node].append(self.cur_graph.tensor(full_name))
      
  
    node_input_args = []
    if not raw_node.inputs:
      node_input_args.extend(
          [self.get_nndct_value(i) for i in raw_node.outputs])
    else:
      node_input_args.extend(
          [self.get_nndct_value(i) for i in raw_node.inputs])
      
    nndct_node.op = self._create_op(raw_node.kind, nndct_node, node_input_args)
    
    return nndct_node
  
  def _convert_tensor(self, value, scope=None):
    if scope is None:
      assert self.cur_graph 
      value_scope = self.cur_graph.name
    else:
      value_scope = scope
      
    if isinstance(value.data, torch.Tensor):
      nndct_tensor = Tensor(
          name=get_full_name(value_scope, value.name),
          shape=value.shape,
          dtype=value.dtype,
          data=value.data.cpu().numpy(),
          layout=value.layout,
          device=value.device,
          requires_grad=value.requires_grad)
    else:
      nndct_tensor = Tensor(
          name=get_full_name(value_scope, value.name),
          shape=value.shape,
          dtype=value.dtype,
          data=value.data,
          layout=value.layout,
          device=value.device,
          requires_grad=value.requires_grad)
    return nndct_tensor
  
  @staticmethod
  def _create_op(node_kind, nndct_node, node_input_args):
    op_creator = OpCreator()
    op_creator.cur_node = nndct_node
    op_type = op_creator.op_convert_map.get(node_kind, node_kind)
    try:
      if hasattr(op_creator, op_type):
        op = getattr(op_creator, op_type)(*node_input_args)
      elif nndct_node.is_custom_extension:
        op = op_creator.custom_op(nndct_node, op_type, *node_input_args)
      else:
        op = op_creator.default(nndct_node, op_type, *node_input_args)
    except Exception as e:
      NndctScreenLogger().warning(f"The op `{node_kind}` parse error.\nException:`{str(e)}`")
      op = op_creator.default(nndct_node, op_type, *node_input_args)
    return op    
  
  @staticmethod  
  def convert_dtype(dtype):
    r"""convert torch dtype to nndct dtype"""
    return {
        'torch.float': 'float32',
        'torch.double': 'float64',
        'torch.int': 'int32',
        'torch.long': 'int64'
    }.get(dtype, dtype)
  

  def _convert_blob_tensor_type(self, graph):
    r"""convert torch tensor info to nndct tensor info"""
    for blob_tensor in graph.tensors:
      # tensor_util.convert_blob_tensor_format(blob_tensor,
      #                                        tensor_util.FrameworkType.TORCH,
      #                                        tensor_util.FrameworkType.NNDCT)
      blob_tensor.dtype = self.convert_dtype(blob_tensor.dtype)
  

  @staticmethod
  def _get_tensor_data_from_module(module, tensor):
    if len(tensor.uses) == 1 and tensor.data is not None and isinstance(tensor.data, np.ndarray):
      return np.copy(tensor.data)
    else:
      return module.state_dict()[get_short_name(tensor.name)].cpu().numpy()

  @classmethod
  def _load_data(cls, graph, module):
    for node in graph.nodes:
      if node.op.type in [NNDCT_OP.BASIC_LSTM, NNDCT_OP.BASIC_GRU]:
        for nndct_param, param_tensors in node.op.params.items():
          for tensor in param_tensors:
            data = cls._get_tensor_data_from_module(module, tensor)
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
                bias = Tensor(name=get_full_name(graph.name, tensor_name), data=bias_ih.data + bias_hh.data)
                bias_list.append(bias)
                i = i + 2
              node.op.set_param(bias_term, bias_list)
              
      elif node.op.type in [NNDCT_OP.CONVTRANSPOSE2D, NNDCT_OP.CONVTRANSPOSE3D]:
        for param_name, tensor in node.op.params.items():
          data = cls._get_tensor_data_from_module(module, tensor)
          if param_name == node.op.ParamName.WEIGHTS:
            data = np.copy(data).swapaxes(0, 1)
            data = np.ascontiguousarray(data)

          tensor.from_ndarray(data)
          tensor = tensor_util.convert_parameter_tensor_format(
              tensor, FrameworkType.TORCH, FrameworkType.NNDCT)

      elif node.op.type in [NNDCT_OP.DEPTHWISE_CONV2D, NNDCT_OP.DEPTHWISE_CONV3D]:
        for param_name, tensor in node.op.params.items():
          data = cls._get_tensor_data_from_module(module, tensor)
          if param_name == node.op.ParamName.WEIGHTS:
            in_channels = node.node_config("in_channels")
            out_channels = node.node_config("out_channels")
            kernel_size = node.node_config("kernel_size")
            channel_mutiplier = int(out_channels/in_channels)
            data = np.copy(data).reshape((channel_mutiplier, in_channels, *kernel_size))

          tensor.from_ndarray(data)
          tensor = tensor_util.convert_parameter_tensor_format(
              tensor, FrameworkType.TORCH, FrameworkType.NNDCT)
      elif node.op.type in [NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D]:
          for param_name, tensor in node.op.params.items():
            data = cls._get_tensor_data_from_module(module, tensor)
            if param_name == node.op.ParamName.WEIGHTS:
              # data = np.copy(data).transpose(1, 0, 2, 3)
              # data = np.ascontiguousarray(data)
              in_channels = node.node_config("in_channels")
              out_channels = node.node_config("out_channels")
              kernel_size = node.node_config("kernel_size")
              channel_mutiplier = int(out_channels / in_channels)
              data = np.copy(data).reshape((in_channels, channel_mutiplier, *kernel_size))
              data = np.copy(data).swapaxes(0, 1)
              data = np.ascontiguousarray(data)

            tensor.from_ndarray(data)
            tensor = tensor_util.convert_parameter_tensor_format(
              tensor, FrameworkType.TORCH, FrameworkType.NNDCT)
           
          
      elif node.blocks:
        for block in node.blocks:
          cls._load_data(block, module)
      else:
        for param_name, tensor in node.op.params.items():
          data = cls._get_tensor_data_from_module(module, tensor)
          tensor.from_ndarray(data)
          if node.has_bound_params():
            tensor = tensor_util.convert_parameter_tensor_format(
                tensor, FrameworkType.TORCH, FrameworkType.NNDCT)

  def _get_device_info_from_inputs(self, inputs):
    if isinstance(inputs, torch.Tensor):
      return inputs.device.type

    elif isinstance(inputs, (tuple, list)):
      for ip in inputs:
        device = self._get_device_info_from_inputs(ip)
        if device is not None:
          return device
    else:
      return None
       
  def _get_device_info(self, module, inputs):
    if module.state_dict():
      for _, item in module.state_dict().items():
        if isinstance(item, torch.Tensor):
          return item.device.type
    else:
      return self._get_device_info_from_inputs(inputs)
      
  def get_nndct_value(self, value):
    if isinstance(value, (tuple, list)):
      values = []
      for ele in value:
        values.append(self.get_nndct_value(ele))
      return type(value)(values)
    else:
      if value.is_none():
        return None
      elif value.is_plain_value():
        return value.data
      else:
        #tensor_map = ChainMap(self.visited_blob_tensors, self.visited_param_tensors)
        return self.cur_graph.tensor(get_full_name(self.cur_graph.name, value.name))
