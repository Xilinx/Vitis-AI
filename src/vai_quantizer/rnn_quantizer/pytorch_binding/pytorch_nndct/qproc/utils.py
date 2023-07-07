

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

import importlib.util
import os
import sys
from typing import Any, NoReturn, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import nndct_shared.utils as nndct_utils
from nndct_shared.base import FrameworkType
from nndct_shared.nndct_graph import Graph
from nndct_shared.optimization import QuantOptimizer
from pytorch_nndct.export import get_script_writer
from nndct_shared.utils import NndctDebugLogger, NndctOption, NndctScreenLogger, NNDCT_OP, permute_data
from pytorch_nndct.parse import TorchParser
from pytorch_nndct.quantization import TORCHQuantizer
from pytorch_nndct.utils import TorchSymbol
from pytorch_nndct.nn.modules import reluk, channel_scale
from nndct_shared.compile import DevGraphOptimizer
from pytorch_nndct.nn import stacked_lstm
from .ModuleHooker import ModuleHooker


def _reload_module(module_file_name: str, module_name: str) -> torch.nn.Module:

  py_module_name = "_".join(["nndct", module_name])
  spec = importlib.util.spec_from_file_location(py_module_name,
                                                module_file_name)
  py_module = importlib.util.module_from_spec(spec)
  sys.modules[py_module_name] = py_module
  spec.loader.exec_module(py_module)
  return py_module.__dict__[module_name]()


def recreate_nndct_module(graph: Graph, enable_quant: bool, export_file: str) -> torch.nn.Module:

  exporter = get_script_writer(enable_quant=enable_quant)
  exporter.write(graph, file_path=export_file)
  nndct_module = _reload_module(export_file, graph.name)
  return nndct_module


def parse_module(module: Union[torch.nn.Module, torch.jit.ScriptModule],
                 input_args: Union[torch.Tensor, Sequence[Any]],
                 enable_opt: bool = True,
                 graph_name: Optional[str] = None) -> Graph:

  if NndctOption.nndct_equalization.value:
    if NndctOption.nndct_relu6_replace.value == 'reluk':
      replace_relu6_with_reluk(module)
    elif NndctOption.nndct_relu6_replace.value == 'relu':
      replace_relu6_with_relu(module)
  
  # if NndctOption.nndct_wes.value:
  #   insert_scale_after_conv2d(module)
  
  parser = TorchParser()
  graph = parser(module._get_name() if graph_name is None else graph_name,
                 module, input_args)
  if enable_opt:
    graph = quant_optimize(graph)
    
  if NndctOption.nndct_parse_debug.value >= 3:
      NndctDebugLogger.write(f"nndct quant graph:\n{graph}")
  return graph


def quant_optimize(graph: Graph):
  def _execute_optimize(block):
    optimizer = QuantOptimizer()
    graph = optimizer(block)
    return graph
    
  for block in graph.block_subgraphs():
      quant_optimize(block)
  graph = _execute_optimize(graph)
  return graph
  
def register_output_hook(module: torch.nn.Module, record_once: bool = True) -> NoReturn:
  ModuleHooker.register_output_hook(module, record_once)


def disconnect_modeule_with_graph(module):
  ModuleHooker.detach_node_from_module(module)
  
  
def connect_module_with_graph(module: torch.nn.Module,
                              graph: Graph,
                              recover_param: bool = True,
                              recover_state_dict_keys: bool = False) -> NoReturn:
  """
  Hook graph info with modules
  Args:
      module (torch.nn.Module): rebuild module
      graph (Graph): nndct graph
      record_once (bool, optional): whether record output once or multiple times. Defaults to True.
      recover_param (bool, optional): recover parameters from graph to module. Defaults to True.
      recover_state_dict_keys (bool, optional): recover the state dict keys in rebuild module from graph. Defaults to False.
  """

  # ModuleHooker.register_output_hook(module, record_once)

  ModuleHooker.hook_module_with_node(module, graph)
  
  # if bn is fused into conv, recover_state_dict_keys should be False
  if recover_state_dict_keys:
    ModuleHooker.register_state_dict_hook(module)
  
  if recover_param:
    ModuleHooker.update_parameters(module, graph, graph2module=True)


# def connect_module_with_quantizer(module: torch.nn.Module,
#                                   quantizer: TORCHQuantizer) -> NoReturn:
#   ModuleHooker.hook_module_with_quantizer(module, quantizer)


def update_nndct_parameters(module: torch.nn.Module, graph: Graph) -> NoReturn:
  ModuleHooker.update_parameters(module, graph, graph2module=False)


def update_nndct_blob_data(module: torch.nn.Module,
                           graph: Graph,
                           time_step: Optional[int] = None) -> NoReturn:
  ModuleHooker.update_blobs_once(module, graph, time_step)
  
  def dfs(node, visited):
    visited.append(node)
    
    if node.op.type == NNDCT_OP.PERMUTE:
      in_data = node.in_tensors[0].data
      data = permute_data(in_data, node.node_attr(node.op.AttrName.ORDER))
      node.out_tensors[0].from_ndarray(data)
    
    for cn in graph.children(node):
      if cn not in visited:
        dfs(cn, visited)
    
  source_nodes = []
  for node in graph.nodes:
    if not node.in_tensors:
      source_nodes.append(node)
    
  visited = []
  for source in source_nodes:
    dfs(source, visited)
 

def set_outputs_recorder_status(module, turn_on) -> NoReturn:
  ModuleHooker.clear_record_outputs(module)
  ModuleHooker.turn_on_record_outputs(
      module) if turn_on else ModuleHooker.turn_off_record_outputs(module)


def prepare_quantizable_module(
    module: torch.nn.Module,
    input_args: Union[torch.Tensor, Sequence[Any]],
    export_folder: str,
    state_dict_file: Optional[str] = None,
    quant_mode: int = 1, 
    device: torch.device = torch.device("cuda")) -> Tuple[torch.nn.Module, Graph]:

  nndct_utils.create_work_dir(export_folder)

  if isinstance(state_dict_file, str):
    state_dict = torch.load(state_dict_file)
    module.load_state_dict(state_dict)

  export_file = os.path.join(export_folder,
                             module._get_name() + TorchSymbol.SCRIPT_SUFFIX)
  
  # switch to specified device
  module, input_args = to_device(module, input_args, device)
  
  # parse origin module to graph
  NndctScreenLogger().info(f"=>Parsing {module._get_name()}...")
  graph = parse_module(module, input_args)
  NndctScreenLogger().info(f"=>Quantizable module is generated.({export_file})")
  # recreate quantizable module from graph
  quant_module = recreate_nndct_module(graph, True, export_file).to(device)
  quant_module.train(mode=module.training)
  # hook module with graph
  connect_module_with_graph(quant_module, graph)

  return quant_module, graph


def to_device(module: torch.nn.Module, 
              input_args: Union[torch.Tensor, Sequence[Any]], 
              device: torch.device) -> Tuple[torch.nn.Module, Union[torch.Tensor, Sequence[Any]]]:
  NndctScreenLogger().info(f"=>Quant Module is in '{device.type}'.")
  if input_args is not None:
    if isinstance(input_args, torch.Tensor):
      input_args = input_args.to(device)
    else:
      is_tuple = True if isinstance(input_args, tuple) else False
      input_args = list(input_args)
      for i in range(len(input_args)):
        if isinstance(input_args[i], torch.Tensor):
          input_args[i] = input_args[i].to(device)
      if is_tuple:
        input_args = tuple(input_args)
        
  module = module.to(device)
  return module, input_args

def replace_relu6_with_relu(module: torch.nn.Module):
  def _replace_func(op):
    for op_name, c_op in op.named_children():
      if isinstance(c_op, torch.nn.ReLU6):
        op._modules[op_name] = torch.nn.ReLU(c_op.inplace)
  if any([isinstance(submodule, torch.nn.ReLU6) for submodule in module.modules()]):
    module.apply(_replace_func)
    NndctScreenLogger().warning(f"ReLU6 has been replaced by ReLU.")
    
def replace_relu6_with_reluk(module: torch.nn.Module):
  def _replace_func(op):
    for op_name, c_op in op.named_children():
      if isinstance(c_op, torch.nn.ReLU6):
        #op._modules[op_name] = torch.nn.ReLU(c_op.inplace)
        op._modules[op_name] = reluk.ReLUk(channel_max=6.0)
  if any([isinstance(submodule, torch.nn.ReLU6) for submodule in module.modules()]):
    module.apply(_replace_func)
    NndctScreenLogger().warning(f"ReLU6 has been replaced by ReLUK.")
    
def insert_scale_after_conv2d(module: torch.nn.Module):
  def _insert_func(op):
    insert_name = None
    conv2d_cnt = 0 
    find_conv2d = False 
    for op_name, c_op in op.named_children():
      if find_conv2d:
        conv2d_cnt = conv2d_cnt+1
      if isinstance(c_op, torch.nn.Conv2d) or isinstance(c_op, torch.nn.ConvTranspose2d):
        find_conv2d = True
        insert_name = op_name
      elif isinstance(c_op, torch.nn.BatchNorm2d) and (find_conv2d == True):
        insert_name = op_name
       
      if conv2d_cnt == 1:
        op._modules[insert_name] = torch.nn.Sequential(op._modules[insert_name], channel_scale.ChannelScale(channel_scale=1.0))
        find_conv2d = False
        conv2d_cnt = 0
    if find_conv2d:
      op._modules[insert_name] = torch.nn.Sequential(op._modules[insert_name], channel_scale.ChannelScale(channel_scale=1.0))
        
  if any([(isinstance(submodule, torch.nn.Conv2d) or isinstance(submodule, torch.nn.ConvTranspose2d)) for submodule in module.modules()]):
    module.apply(_insert_func)
    NndctScreenLogger().warning(f"ChannelScale has been inserted after Conv2d.")
    
def insert_scale_after_batchnorm2d(module: torch.nn.Module):
  def _insert_func(op):
    for op_name, c_op in op.named_children():
      if isinstance(c_op, torch.nn.BatchNorm2d):
        op._modules[op_name] = torch.nn.Sequential(op._modules[op_name], channel_scale.ChannelScale(channel_scale=1.0))
  if any([(isinstance(submodule, torch.nn.BatchNorm2d)) for submodule in module.modules()]):
    module.apply(_insert_func)
    NndctScreenLogger().warning(f"ChannelScale has been inserted after batchnorm2d.")

def get_deploy_graph_list(quant_model, nndct_graph):
  g_optmizer = DevGraphOptimizer(nndct_graph)
  # g_optmizer.infer_tensor_layout()
  g_optmizer.layout_tranform()
  g_optmizer.strip_redundant_ops()
  
  # for node in g_optmizer._dev_graph.nodes:
  #   print(f"{node.name}, {node.op.type}, {node.out_tensors[0].layout}")
    
  # sync model data with dev graph
  connect_module_with_graph(quant_model, g_optmizer.frozen_graph, recover_param=False)
  update_nndct_blob_data(quant_model, g_optmizer.frozen_graph)
  connect_module_with_graph(quant_model, nndct_graph, recover_param=False)
  
  g_optmizer.constant_folding()
  if NndctOption.nndct_parse_debug.value >= 3:
    NndctDebugLogger.write(f"\nfrozen dev graph:\n{g_optmizer.frozen_graph}")
  
  deploy_graphs = g_optmizer.partition_by_quant_part() 
  
  return deploy_graphs


def convert_lstm(ori_module: torch.nn.Module):
  """replace_torch_lstm_with_stacked_lstm

  """
  
  if isinstance(ori_module, torch.nn.LSTM):
    return stacked_lstm(ori_module)
  
  for n, m in ori_module.named_children():
    if isinstance(m, torch.nn.LSTM):
      setattr(ori_module, n, stacked_lstm(m))
    else:
      convert_lstm(m)    
  return ori_module
  
  
    
 
