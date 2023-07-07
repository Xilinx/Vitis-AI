

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
from nndct_shared.nndct_graph import (Graph, Tensor,
                                      reorder_multi_subgraph_nodes, collect_all_blocks)
from nndct_shared.utils import NndctDebugLogger, NndctOption, NndctScreenLogger
from pytorch_nndct.utils import build_aten_torch_ops_table

from .op_dispatcher import *
from .parse_utils import *
from .torch_op_def import TorchUnknownOperation
from .trace_helper import TorchGraphHandler


def unknown_op_type_check(graph: Graph):
  graphs = [graph] + list(graph.block_subgraphs())
  unkown_ops = set()
  custom_ops = set()
  for graph in graphs:
    for node in graph.nodes:
      if isinstance(node.op, TorchUnknownOperation):
        unkown_ops.add(node.op.type)
      elif node.has_custom_op():
        custom_ops.add(node.op.type)
        
  for op in custom_ops:
      NndctScreenLogger().warning(f"The quantizer recognize new op `{op}` as a float operator by default.")

#   if custom_ops:
#     NndctScreenLogger().info(f"You can make these new ops quantizable by add them to custom_quant_ops, \
# e.g. quantizer= torch_quantizer(..., custom_quant_ops=['{list(custom_ops)[0]}',...])")

  NndctScreenLogger().check(f"Unsupported Ops: {unkown_ops}", len(unkown_ops)==0)

class TorchParser(object):
  def __init__(self):
    self.visited_blob_tensors = {}
    self.visited_param_tensors = {}
    self.node_input_args = defaultdict(list)
    self.node_params = defaultdict(list)
    build_aten_torch_ops_table()
    
  def __call__(self, graph_name, module, input_args):
    # torch_graph_handler = TorchGraphHandler()
    graph_handler = create_graph_handler(module)
    raw_graph, raw_params = graph_handler.build_torch_graph(graph_name, module, input_args) 
   
    self._convert_params(raw_params, raw_graph.name)
    NndctScreenLogger().info("Processing ops...")
    nndct_graph = self._convert_graph(raw_graph, self._get_device_info(module, input_args))
    unknown_op_type_check(nndct_graph)  
    graphs = [nndct_graph]
    #graphs.extend(list(nndct_graph.block_subgraphs()))
    collect_all_blocks(nndct_graph, graphs)
    reorder_multi_subgraph_nodes(graphs)      
    self._convert_blob_tensor_type(nndct_graph)
    self._load_data(nndct_graph, module)
    if NndctOption.nndct_parse_debug.value >= 2:
      NndctDebugLogger.write(f"nndct raw graph:\n{nndct_graph}")
    # print(f"nndct graph:{self._nndct_graph}")
    return nndct_graph

  def _convert_params(self, raw_params, param_scope_name=""):
    param_tensor_convertor = TensorConvertor()
    for name, value in raw_params.items():
      param_tensor = param_tensor_convertor(param_scope_name, value)
      self.visited_param_tensors[value.name] = param_tensor
  
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
        
  def _convert_graph(self, raw_graph, device_type):
    nndct_graph = Graph(graph_name=raw_graph.name)
    node_convertor = NodeConvertor()
    op_creator = OpCreator(device_type)
    pbar = tqdm(list(raw_graph.nodes), bar_format="{bar:50}{r_bar}")
    #for raw_node in raw_graph.nodes:
    for raw_node in pbar:
      pbar.set_postfix_str(f"OpInfo: name = {raw_node.name}, type = {raw_node.kind}")
      pbar.update()
      nndct_node = node_convertor(self, raw_node, node_scope=nndct_graph.name)
      if nndct_node:
        nndct_graph.add_node(nndct_node)
        nndct_node.op = op_creator(self, nndct_node)
        for i, block in enumerate(raw_node.blocks):
          nndct_block = self._convert_graph(block, device_type)
          nndct_node.add_block(nndct_block)     
            
        self._bind_free_params(nndct_node)
     
    def _construct_ret_struct(values, return_struct):
      for value in values:
        if isinstance(value, list):
          inner_list = []
          _construct_ret_struct(value, inner_list)
          return_struct.append(inner_list)
        else:
          end_tensor = self.get_nndct_value(value)
          assert end_tensor is not None
          return_struct.append(end_tensor)
          nndct_graph.add_end_tensor(end_tensor)
            
    nndct_graph.return_struct = []
    _construct_ret_struct(raw_graph.ret_values().values(), nndct_graph.return_struct)
    nndct_graph.connect_nodes()
    
    return nndct_graph
     
  @staticmethod
  def _convert_blob_tensor_type(graph):
    r"""convert torch tensor info to nndct tensor info"""
    for blob_tensor in graph.tensors:
      # tensor_util.convert_blob_tensor_format(blob_tensor,
      #                                        tensor_util.FrameworkType.TORCH,
      #                                        tensor_util.FrameworkType.NNDCT)
      blob_tensor.dtype = convert_dtype(blob_tensor.dtype)
  
  @staticmethod
  def _load_data(graph, module):
    for node in graph.nodes:
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
                bias = Tensor(name=get_full_name(graph.name, tensor_name), data=bias_ih.data + bias_hh.data)
                bias_list.append(bias)
                i = i + 2
              node.op.set_param(bias_term, bias_list)
              
      elif node.op.type in [NNDCT_OP.CONVTRANSPOSE2D, NNDCT_OP.CONVTRANSPOSE3D]:
        for param_name, tensor in node.op.params.items():
          data = module.state_dict()[get_short_name(tensor.name)].cpu().numpy()
          if param_name == node.op.ParamName.WEIGHTS:
            data = np.copy(data).swapaxes(0, 1)
            data = np.ascontiguousarray(data)

          tensor.from_ndarray(data)
          tensor = tensor_util.convert_parameter_tensor_format(
              tensor, FrameworkType.TORCH, FrameworkType.NNDCT)

      elif node.op.type in [NNDCT_OP.DEPTHWISE_CONV2D, NNDCT_OP.DEPTHWISE_CONV3D]:
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
      elif node.op.type in [NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D]:
          for param_name, tensor in node.op.params.items():
            data = module.state_dict()[get_short_name(tensor.name)].cpu().numpy()
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
          TorchParser._load_data(block, module)
      else:
        for param_name, tensor in node.op.params.items():
          data = module.state_dict()[get_short_name(tensor.name)].cpu().numpy()
          tensor.from_ndarray(data)
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
    if isinstance(value, list):
      values = []
      for ele in value:
        values.append(self.get_nndct_value(ele))
      return values
    else:
      if value.is_none():
        return None
      elif value.is_plain_value():
        return value.data
      else:
        tensor_map = ChainMap(self.visited_blob_tensors, self.visited_param_tensors)
        return tensor_map[value.name]
