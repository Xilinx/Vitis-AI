

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
from typing import Any, NoReturn, Optional, Sequence, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import nndct_shared.utils as nndct_utils
import pytorch_nndct.utils.jit_utils as jit_utils
import pytorch_nndct.utils.module_util as module_util
from nndct_shared.base import FrameworkType
from nndct_shared.compile import DevGraphOptimizer
from nndct_shared.nndct_graph import Graph
from nndct_shared.optimization import QuantOptimizer
from nndct_shared.utils import (GLOBAL_MAP, NNDCT_KEYS, NNDCT_OP,
                                NndctDebugLogger, NndctOption,
                                NndctScreenLogger, permute_data, permute_axes,
                                QError, QWarning, QNote)
from nndct_shared.utils.dpu_utils import get_avgpool_dpu_coeff
from pytorch_nndct.export import get_script_writer
from pytorch_nndct.nn import stacked_lstm
from pytorch_nndct.nn.modules import channel_scale, reluk
from pytorch_nndct.parse import TorchParser
from pytorch_nndct.utils import TorchSymbol
from pytorch_nndct.utils.module_util import to_device, get_module_name
from pytorch_nndct.utils.jit_utils import get_torch_version
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
  if NndctOption.nndct_convert_sigmoid_to_hsigmoid.value:
    replace_sigmoid_with_hsigmoid(module)
  if NndctOption.nndct_convert_silu_to_hswish.value:
    replace_silu_with_hswish(module)
  
  # if NndctOption.nndct_wes.value:
  #   insert_scale_after_conv2d(module)
 
  # replace affine=False with affine=True
  replace_batchnorm_affine_false_with_true(module)
  parser = TorchParser()
  graph = parser(get_module_name(module) if graph_name is None else graph_name,
                 module, input_args)
  if enable_opt:
    graph = quant_optimize(graph)
    
  if NndctOption.nndct_parse_debug.value >= 3:
      NndctDebugLogger.write(f"nndct quant graph:\n{graph}")
  return graph


def quant_optimize(graph: Graph):
  optimizer = QuantOptimizer() 
  graph = optimizer(graph, fuse_conv_bn=NndctOption.nndct_conv_bn_merge.value)
  return graph
  #   return graph
  #   graph = optimizer(block)
  # def _execute_optimize(block):
  #   optimizer = QuantOptimizer()
  #   graph = optimizer(block)
  #   return graph
    
  # for block in graph.block_subgraphs():
  #     quant_optimize(block)
  # graph = _execute_optimize(graph)
  # return graph
  
def register_input_dump_hook(module: torch.nn.Module) -> NoReturn:
  ModuleHooker.register_input_dump_hook(module)
  
def register_output_intime_hook(module: torch.nn.Module) -> NoReturn:
  ModuleHooker.register_output_intime_hook(module)
  
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
                           time_step: Optional[int] = None,
                           only_update_shape=False) -> NoReturn:
  ModuleHooker.update_blobs_once(module, graph, time_step, only_update_shape)
  permute_nodes = graph.find_nodes_by_types([NNDCT_OP.PERMUTE])
  if only_update_shape is False:
    for node in permute_nodes:
      in_data = node.in_tensors[0].data
      if in_data is not None:
        data = permute_data(in_data, node.node_attr(node.op.AttrName.ORDER))
        node.out_tensors[0].from_ndarray(data)
      else:
        NndctScreenLogger().warning(f"{node.__repr__()} has no data.")
  else:
    for node in permute_nodes:
      in_shape = node.in_tensors[0].shape
      if in_shape is not None:
        out_shape = permute_axes(in_shape, node.node_attr(node.op.AttrName.ORDER))
        node.out_tensors[0].shape = out_shape
      else:
        NndctScreenLogger().warning(f"{node.__repr__()} has no shape.")

 

def set_outputs_recorder_status(module, turn_on) -> NoReturn:
  ModuleHooker.clear_record_outputs(module)
  ModuleHooker.turn_on_record_outputs(
      module) if turn_on else ModuleHooker.turn_off_record_outputs(module)


def set_output_intime_status(module, turn_on) -> NoReturn:
  ModuleHooker.clear_op_called_times(module)
  ModuleHooker.turn_on_output_intime(
      module) if turn_on else ModuleHooker.turn_off_output_intime(module)
  

def set_input_dump_status(module, turn_on) -> NoReturn:
  ModuleHooker.clear_input_called_times(module)
  ModuleHooker.turn_on_input_dump(
      module) if turn_on else ModuleHooker.turn_off_input_dump(module)


def prepare_quantizable_module(
    module: torch.nn.Module,
    input_args: Union[torch.Tensor, Sequence[Any]],
    export_folder: str,
    state_dict_file: Optional[str] = None,
    quant_mode: int = 1, 
    device: "torch.device" = torch.device("cuda"),
    connect_qm_with_graph=True) -> Tuple[torch.nn.Module, Graph]:

  nndct_utils.create_work_dir(export_folder)

  if isinstance(state_dict_file, str):
    state_dict = torch.load(state_dict_file)
    module.load_state_dict(state_dict)

  export_file = os.path.join(export_folder, get_module_name(module) + TorchSymbol.SCRIPT_SUFFIX)
  
  # switch to specified device
  NndctScreenLogger().info(f"=>Quant Module is in '{device.type}'.")
  module, input_args = to_device(module, input_args, device)
  
  # parse origin module to graph
  NndctScreenLogger().info(f"=>Parsing {get_module_name(module)}...")
  graph = parse_module(module, input_args)
  NndctScreenLogger().info(f"=>Quantizable module is generated.({export_file})")
  # recreate quantizable module from graph
  quant_module = recreate_nndct_module(graph, True, export_file).to(device)
  quant_module.train(mode=module.training)
  
  # hook module with graph
  if connect_qm_with_graph is True:
    connect_module_with_graph(quant_module, graph)
  
  if quant_mode > 1 and NndctOption.nndct_deploy_check.value > 0:
    register_output_intime_hook(quant_module)
    set_output_intime_status(quant_module, True)
  
  if quant_mode > 1 and NndctOption.nndct_input_check.value > 0:
    register_input_dump_hook(quant_module)
    set_input_dump_status(quant_module, True)

  return quant_module, graph


def replace_relu6_with_relu(module: torch.nn.Module):
  def _replace_func(op):
    for op_name, c_op in op.named_children():
      if isinstance(c_op, torch.nn.ReLU6):
        op._modules[op_name] = torch.nn.ReLU(c_op.inplace)
  if any([isinstance(submodule, torch.nn.ReLU6) for submodule in module.modules()]):
    module.apply(_replace_func)
    NndctScreenLogger().warning2user(QWarning.REPLACE_RELU6, f"ReLU6 has been replaced by ReLU.")
    
def replace_relu6_with_reluk(module: torch.nn.Module):
  def _replace_func(op):
    for op_name, c_op in op.named_children():
      if isinstance(c_op, torch.nn.ReLU6):
        #op._modules[op_name] = torch.nn.ReLU(c_op.inplace)
        op._modules[op_name] = reluk.ReLUk(channel_max=6.0)
  if any([isinstance(submodule, torch.nn.ReLU6) for submodule in module.modules()]):
    module.apply(_replace_func)
    NndctScreenLogger().warning2user(QWarning.REPLACE_RELUK, f"ReLU6 has been replaced by ReLUK.")

def replace_sigmoid_with_hsigmoid(module: torch.nn.Module):
  def _replace_func(op):
    for op_name, c_op in op.named_children():
      if isinstance(c_op, torch.nn.Sigmoid):
        op._modules[op_name] = torch.nn.Hardsigmoid()
  if any([isinstance(submodule, torch.nn.Sigmoid) for submodule in module.modules()]):
    module.apply(_replace_func)
    NndctScreenLogger().warning2user(QWarning.REPLACE_SIGMOID, f"Sigmoid has been replaced by Hardsigmoid.")

def replace_silu_with_hswish(module: torch.nn.Module):
  def _replace_func(op):
    for op_name, c_op in op.named_children():
      if isinstance(c_op, torch.nn.SiLU):
        op._modules[op_name] = torch.nn.Hardswish()
  if (get_torch_version() >= 170) and any([isinstance(submodule, torch.nn.SiLU) for submodule in module.modules()]):
    module.apply(_replace_func)
    NndctScreenLogger().warning2user(QWarning.REPLACE_SILU, f"SiLU has been replaced by Hardswish.")
    
def replace_batchnorm_affine_false_with_true(module: torch.nn.Module):
  def _replace_func(op):
    for op_name, c_op in op.named_children():
      # by default: gamma=1, beta=0
      if isinstance(c_op, torch.nn.BatchNorm1d) and not c_op.affine:
        c_op_new = torch.nn.BatchNorm1d(num_features=c_op.num_features, eps=c_op.eps, momentum=c_op.momentum, affine=True, track_running_stats=c_op.track_running_stats)
      elif isinstance(c_op, torch.nn.BatchNorm2d) and not c_op.affine:
        c_op_new = torch.nn.BatchNorm2d(num_features=c_op.num_features, eps=c_op.eps, momentum=c_op.momentum, affine=True, track_running_stats=c_op.track_running_stats)
      elif isinstance(c_op, torch.nn.BatchNorm3d) and not c_op.affine:
        c_op_new = torch.nn.BatchNorm3d(num_features=c_op.num_features, eps=c_op.eps, momentum=c_op.momentum, affine=True, track_running_stats=c_op.track_running_stats)
      if isinstance(c_op, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)) and not c_op.affine:
        c_op_new.running_mean.data.copy_(c_op.running_mean.data)
        c_op_new.running_var.data.copy_(c_op.running_var.data)
        c_op_new = c_op_new.to(c_op.running_mean.device)
        op._modules[op_name] = c_op_new 
        NndctScreenLogger().warning2user(QWarning.BATCHNORM_AFFINE, f"{op_name} attribute affine=False has been replaced by affine=True when parsing the model.")

  if any([isinstance(submodule, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)) for submodule in module.modules()]):
    module.apply(_replace_func)
    
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

def get_deploy_graph_list(quant_model, nndct_graph, need_partition=True):
  g_optmizer = DevGraphOptimizer(nndct_graph)
  # sync model data with dev graph

  connect_module_with_graph(quant_model, g_optmizer.dev_graph, recover_param=False)
  update_nndct_blob_data(quant_model, g_optmizer.dev_graph, only_update_shape=False)
  g_optmizer.strip_redundant_ops()
  g_optmizer.update_op_attrs()
  g_optmizer.convert_shape_tensor_to_const()
  g_optmizer.constant_folding()
  g_optmizer.fuse_pad()
  g_optmizer.fuse_transpose_matmul()
  g_optmizer.layout_tranform()    
  g_optmizer.fuse_redundant_transpose()
  # connect_module_with_graph(quant_model, g_optmizer.dev_graph, recover_param=False)
  # update_nndct_blob_data(quant_model, g_optmizer.dev_graph, only_update_shape=False)
  # connect_module_with_graph(quant_model, nndct_graph, recover_param=False)  
  g_optmizer.update_node_data()
  g_optmizer.convert_reshapelike_to_reshape()
  g_optmizer.merge_permute_to_linear()
  g_optmizer.merge_consecutive_reshape()
  g_optmizer.broadcast_const_for_binary_op()
  g_optmizer.convert_rsub_to_sub()
  g_optmizer.convert_adaptive_pool_to_pool()
  # for node in g_optmizer._dev_graph.nodes:
  #   print(f"{node.name}, {node.op.type}, {node.out_tensors[0].layout}")

  if NndctOption.nndct_parse_debug.value >= 3:
    NndctDebugLogger.write(f"\nfrozen dev graph:\n{g_optmizer.dev_graph}")
  
  deploy_graphs = g_optmizer.partition_by_quant_part() if need_partition is True else []
  connect_module_with_graph(quant_model, nndct_graph, recover_param=False)  
  return deploy_graphs, g_optmizer.dev_graph


def convert_lstm(ori_module: torch.nn.Module, device):
  """replace_torch_lstm_with_stacked_lstm

  """
  
  if isinstance(ori_module, torch.nn.LSTM):
    if NndctOption.nndct_jit_script.value:
      return torch.jit.script(stacked_lstm(ori_module).to(device).eval())
    else:
      return stacked_lstm(ori_module).to(device).eval()
  
  for n, m in ori_module.named_children():
    if isinstance(m, torch.nn.LSTM):
      if NndctOption.nndct_jit_script.value:
        setattr(ori_module, n, torch.jit.script(stacked_lstm(m).to(device).eval()))
      else:
        setattr(ori_module, n, stacked_lstm(m).to(device).eval())
    else:
      convert_lstm(m, device)    
  return ori_module

def has_lstm(module):
  for mod in module.modules():
    if isinstance(mod, torch.nn.LSTM):
      return True
  return False


def _get_node_scope(node):
  if isinstance(node, str):
    return node.split("::")[0]
  else:
    return node.name.split("::")[0]

def _valid_bnfp(bnfp):
  if bnfp == None:
    return False
  return not(bnfp[0] == 67108864 and bnfp[1] == 4096)

def insert_fix_neuron_in_script_model(script_model, quantizer):
  quant_config = quantizer.quant_config
  script_graph = script_model.graph
  # device = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_DEVICE)
  # device_id = 1 if device == torch.device("cpu") else 0
  for node in quantizer.Nndctgraph.all_nodes():
    if _get_node_scope(node) != get_module_name(script_model):
      continue
    state_dict = module_util.state_dict_from_node(node)
    if state_dict:
      for tensor_name, tensor in state_dict.items():
        torch_value_name = jit_utils.nndct_name_2_jit_name(tensor_name)
        const_node =  jit_utils.find_fw_node_by_name(script_graph, torch_value_name, recursive=True)
        device_id = 1 if quantizer.graph.param_tensor(tensor_name).device.device_type=='cpu' else 0
        device = torch.device('cpu') if device_id == 1 else torch.device('cuda')
        jit_utils.set_attr_value(const_node, "value", tensor.to(device))
        if tensor_name in quant_config["param"].keys():
          bnfp = quantizer.get_quant_config(tensor_name, True, "param")
          if not _valid_bnfp(bnfp):
            continue
          method = -1 if NndctOption.nndct_use_torch_quantizer.value is True else 3
          fix_node = jit_utils.create_fix_node(script_graph, jit_utils.get_node_output_at(const_node, 0), quant_min=-bnfp[0], 
                                          quant_max=bnfp[0]-1,   
                                          scale_inv=bnfp[1], 
                                          zero_point=0, 
                                          method=method, 
                                          device_id=device_id, 
                                          inplace=quantizer.inplace,
                                          name=tensor_name,
                                          tensor_type='param')
          if fix_node is not None:
            jit_utils.insert_after_node(script_graph, const_node, fix_node, torch_value_name)              



  for nndct_node_name in quant_config["output"].keys():
    if _get_node_scope(nndct_node_name) != get_module_name(script_model):
      continue
    for idx in range(quantizer.get_quant_len(nndct_node_name, "output")):
      index = None if quantizer.get_quant_len(nndct_node_name, "output")==1 else idx
      bnfp = quantizer.get_quant_config(nndct_node_name, True, "output", idx)
      if not _valid_bnfp(bnfp):
        continue

      if quantizer.Nndctgraph.node(nndct_node_name).op.type == NNDCT_OP.INPUT:
        torch_value_name = jit_utils.nndct_name_2_jit_name(nndct_node_name)
        input_value = jit_utils.find_input_value_by_name(script_graph, torch_value_name)
        input_node = jit_utils.get_fw_graph_input_node(script_graph)
        device_id = 1 if quantizer.graph.node(nndct_node_name).out_tensors[idx].device.device_type=='cpu' else 0
        method = 4 if quantizer.lstm else 2
        if NndctOption.nndct_use_torch_quantizer.value is True:
          method = -1
        elif quantizer.lstm and NndctOption.nndct_ip_asr.value is True:
          method = 3
        fix_node = jit_utils.create_fix_node(script_graph, input_value, quant_min=-bnfp[0], 
                                            quant_max=bnfp[0]-1,   
                                            scale_inv=bnfp[1], 
                                            zero_point=0, 
                                            method=method, 
                                            device_id=device_id, 
                                            inplace=quantizer.inplace,
                                            name=nndct_node_name,
                                            tensor_type='output',
                                            index = index)
        if fix_node is not None:
          jit_utils.insert_after_node(script_graph, input_node, fix_node, torch_value_name, idx)
      else:
        torch_value_name = jit_utils.nndct_name_2_jit_name(nndct_node_name)
        fw_node = jit_utils.find_fw_node_by_name(script_graph, torch_value_name, recursive=True)
        device_id = 1 if quantizer.graph.node(nndct_node_name).out_tensors[idx].device.device_type=='cpu' else 0
        method = 4 if quantizer.lstm else 2
        if NndctOption.nndct_use_torch_quantizer.value is True:
          method = -1
        elif quantizer.lstm and NndctOption.nndct_ip_asr.value is True:
          method = 3
          if quantizer.Nndctgraph.node(nndct_node_name).op.type == NNDCT_OP.LAYER_NORM:
            method = 4
        fix_node = jit_utils.create_fix_node(script_graph, jit_utils.get_node_output_at(fw_node, idx), quant_min=-bnfp[0], 
                                            quant_max=bnfp[0]-1,   
                                            scale_inv=bnfp[1], 
                                            zero_point=0, 
                                            method=method, 
                                            device_id=device_id, 
                                            inplace=quantizer.inplace,
                                            name=nndct_node_name,
                                            tensor_type='output',
                                            index=index)
        if fix_node is not None:
          jit_utils.insert_after_node(script_graph, fw_node, fix_node, torch_value_name, idx)

  for nndct_node_name in quant_config["input"].keys():
    if _get_node_scope(nndct_node_name) != get_module_name(script_model):
      continue
    for idx in range(quantizer.get_quant_len(nndct_node_name, "input")):
      index = None if quantizer.get_quant_len(nndct_node_name, "output")==1 else idx
      bnfp = quantizer.get_quant_config(nndct_node_name, True, "input", idx)
      if not _valid_bnfp(bnfp):
        continue

      torch_value_name = jit_utils.nndct_name_2_jit_name(nndct_node_name)
      fw_node = jit_utils.find_fw_node_by_name(script_graph, torch_value_name, recursive=True)
      input_node = jit_utils.get_in_node_at(fw_node, idx)
      device_id = 1 if quantizer.graph.node(nndct_node_name).in_tensors[idx].device.device_type=='cpu' else 0
      method = 4 if quantizer.lstm else 2
      if NndctOption.nndct_use_torch_quantizer.value is True:
        method = -1
      elif quantizer.lstm and NndctOption.nndct_ip_asr.value is True:
        method = 3
      fix_node = jit_utils.create_fix_node(script_graph, jit_utils.get_node_output_at(input_node, 0), quant_min=-bnfp[0], 
                                          quant_max=bnfp[0]-1,   
                                          scale_inv=bnfp[1], 
                                          zero_point=0, 
                                          method=method, 
                                          device_id=device_id, 
                                          inplace=quantizer.inplace,
                                          name=nndct_node_name,
                                          tensor_type='input',
                                          index=index)
      if fix_node is not None:
        jit_utils.insert_before_node(script_graph, fw_node, fix_node, idx)
  return script_model

def opt_script_model_for_quant(script_model):
  g = script_model.graph
  jit_utils.remove_fused_bn(g)
  jit_utils.remove_fused_ln_sigmoid(g)
  jit_utils.remove_dropout(g)
  jit_utils.remove_dce_node(g)
  return script_model


def insert_mul_after_avgpool(script_model, quantizer):  
  def _insert_mul_node(nndct_node_name, kernel):
    scale = get_avgpool_dpu_coeff(kernel)
    torch_value_name = jit_utils.nndct_name_2_jit_name(nndct_node_name)
    fw_node = jit_utils.find_fw_node_by_name(script_graph, torch_value_name)      
    mul_node = jit_utils.create_mul_node(script_graph, jit_utils.get_node_output_at(fw_node, 0), scale)
    if mul_node is not None:
      jit_utils.insert_after_node(script_graph, fw_node, mul_node, torch_value_name)
  
  script_graph = script_model.graph
  nndct_graph = quantizer.Nndctgraph

  for nndct_node in nndct_graph.all_nodes():
    if _get_node_scope(nndct_node.name) != get_module_name(script_model):
      continue
    if nndct_node.op.type == NNDCT_OP.AVG_POOL:
      kernel = nndct_node.node_attr(nndct_node.op.AttrName.KERNEL)
      _insert_mul_node(nndct_node.name, kernel)
    elif nndct_node.op.type == NNDCT_OP.ADAPTIVEAVGPOOL2D:
      if not nndct_node.in_tensors[0].shape or (not nndct_node.out_tensors[0].shape):
        continue
      input_size = nndct_node.in_tensors[0].shape[2:]
      output_size = nndct_node.out_tensors[0].shape[2:]
      mod = [input_size[i] % output_size[i] for i in range(0, len(input_size))]
      if mod != [0] * len(mod):
        continue

      stride_h = int(input_size[0] / output_size[0])
      stride_w = int(input_size[1] / output_size[1])
      kernel_h = input_size[0] - (output_size[0] - 1) * stride_h
      kernel_w = input_size[1] - (output_size[1] - 1) * stride_w
      _insert_mul_node(nndct_node.name, [kernel_w, kernel_h])    
  return script_model


def register_input_checker(module, module_gen_from_script):
  ModuleHooker.hook_module_with_input_device_checker(module, module_gen_from_script)


def remove_quant_dequant_stub(script_model):
  g = script_model.graph
  jit_utils.remove_quant_dequant_stub(g)
  return script_model
  

def quant_model_inferenced(quant_model: Union[torch.nn.Module, List[torch.nn.Module]]) -> bool:
  if isinstance(quant_model, list):
    return all([qmod.is_inferenced for qmod in quant_model])
  else:
    return quant_model.is_inferenced


