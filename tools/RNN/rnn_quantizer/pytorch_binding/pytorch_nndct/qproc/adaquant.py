

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

import os
import numpy as np
import torch
import copy
import nndct_shared.utils as nndct_utils
from nndct_shared.base import GLOBAL_MAP, NNDCT_KEYS, NNDCT_OP
from nndct_shared.utils import NndctOption, NndctScreenLogger, set_option_value
from pytorch_nndct.quantization import TORCHQuantizer
from collections import defaultdict
import torch.nn.functional as F
from tqdm import tqdm
import  pytorch_nndct.nn as py_nn
from nndct_shared.nndct_graph import GraphSearcher
from nndct_shared.utils import PatternType

from .utils import (connect_module_with_graph,
                    disconnect_modeule_with_graph, prepare_quantizable_module,
                    register_output_hook, set_outputs_recorder_status,
                    to_device, update_nndct_blob_data, update_nndct_parameters,
                    get_deploy_graph_list)


class LayerMutiHook(object):
  def __init__(self, in_out):
    # self.inputs = []
    self.outputs = []
    
  def hook(self, layer, input, output):
    # self.inputs.append(input)
    self.outputs.append(output.detach().cpu())

class LayerHook(object):
  def __init__(self):
    # self.input = None
    self.output = None
    
  def hook(self, layer, input, output):
    # self.input = input[0]
    self.output = output
    
class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self, name, fmt=':f'):
    self.name = name
    self.fmt = fmt
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)


class AdvancedQuantProcessor(torch.nn.Module):
  r""" 
  This class re-implements the Adaquant technique proposed in the following paper.
  "Itay Hubara et al., Improving Post Training Neural Quantization: Layer-wise Calibration and Integer Programming, 
  arXiv:2006.10518, 2020."
  """
  def __init__(self, quantizer):
    super().__init__()
    self._quantizer = quantizer
    quantizer.load_param()
    self._graph = quantizer.graph
    self._quant_model = quantizer.quant_model
    self._float_model = copy.deepcopy(self._quant_model)
    for mod in self._float_model.modules():
      if hasattr(mod, "node"):
        mod.node.in_quant_part = False
      
      if hasattr(mod, "quantizer"):
        mod.quantizer = None
        
    self._data_loader = None
    self._num_batches_tracked = 0
    self._cached_outputs = defaultdict(list)
    self._cached_output = defaultdict(list)
    self._handlers = []
    self._net_input_nodes = [node for node in self._graph.nodes if node.op.type == NNDCT_OP.INPUT]
    self._float_weights = defaultdict(list)
  
  def _setup_quantizer(self, quant_mode):
    self._quantizer.quant_mode = quant_mode
     
  def calib(self):
    super().eval()
    self._setup_quantizer(quant_mode=1)

  def eval(self):
    super().eval()
    self._setup_quantizer(quant_mode=2)

  def hook_cache_output(self, hook_mods, hook_type="multiple"):
    handlers = []
    if hook_type == "multiple":
      def hook(module, input, output):
          self._cached_outputs[module].append(output.detach().cpu())
      
      for module in hook_mods:
        handlers.append(module.register_forward_hook(hook))
    else:
      def hook(module, input, output):
          self._cached_output[module] = output
      
      for module in hook_mods:
        handlers.append(module.register_forward_hook(hook))
    return handlers
      
  def collect_last_quant_nodes(self):
    def find_last_quant_node(node, visited=None, quant_nodes=None):
      if node in visited:
        return
      visited.add(node)
      if self.quantizer.configer.is_node_quantizable(node, False):
        quant_nodes.append(node)
        return
      for pn in self._graph.parents(node):
          find_last_quant_node(pn, visited=visited, quant_nodes=quant_nodes)
        
    end_nodes = [tensor.node for tensor in self._graph.end_tensors]
    last_quant_nodes = []
    visited = set()
    for node in end_nodes:
      find_last_quant_node(node, visited=visited, quant_nodes=last_quant_nodes)
    
    return last_quant_nodes
    
  # def clean_hooks(self):
  #   if self._handlers:
  #     for handle in self._handlers:
  #       handle.remove()
        
  @staticmethod
  def clean_hooks(handlers):
    for handle in handlers:
      handle.remove()
        
  def forward(self, inputs):
    return self._quant_model(inputs)

  
  def eval_loss(self, net_inputs, last_quant_mods, device):
    with torch.no_grad():
      loss = AverageMeter("loss")
      for idx, input_args in enumerate(zip(*net_inputs)):
        new_input_args = []
        for ip in input_args:
          if isinstance(ip, torch.Tensor):
            new_input_args.append(ip.to(device))
        _ = self.quant_model(*new_input_args)
        local_loss = 0
        for mod in last_quant_mods:
          cached_net_output = self._cached_outputs[mod][idx]
          output = self._cached_output[mod]
          local_loss += F.mse_loss(cached_net_output.to(device), output).item()
        loss.update(local_loss)
    return loss.avg
              
  def optimize_layer(self, node, float_layer, layer_inputs, layer_act_group, net_inputs, net_loss, last_quant_mods, device):
    batch_factor = 0.5 if layer_inputs[0].size(0) == 1 else 1

    layer = node.module
    float_data = np.fabs(float_layer.weight.cpu().detach().numpy().flatten())
    quant_data = np.fabs(layer.weight.cpu().detach().numpy().flatten())
    q_noise = np.square(float_data - quant_data).mean()
    
    sqnr = 10 * np.log10(np.square(float_data).mean() / q_noise)
    quantize_efficiency = sqnr / 8.0
    
    lr_factor = NndctOption.nndct_finetune_lr_factor.value
    lr_factor = lr_factor * batch_factor
    if quantize_efficiency > 4.5:
      lr_factor = 0.1 * lr_factor * batch_factor
  
    lr_w = lr_factor * layer.weight.std().item()
    # lr_w=1e-3
    opt_weight = torch.optim.Adam([layer.weight], lr=lr_w)
    opt_bias = None
    lr_b = 0
    if hasattr(layer, "bias") and layer.bias is not None:
      if layer.bias.flatten().shape[0] == 1: lr_b = 0.0
      else: lr_b = lr_factor * layer.bias.std().item()
      # lr_b = lr_factor * layer.bias.std().item()
      # lr_b=1e-3
      opt_bias = torch.optim.Adam([layer.bias], lr=lr_b)
      
    #print(f"learning rate: lr_w={lr_w}, lr_b={lr_b}")
    #print(f"pre quant efficiency:{quantize_efficiency}")
    iters = 20
    total_loss = AverageMeter("layer_loss")
    best_params = self.get_layer_params(layer)
    handlers = self.hook_cache_output([float_layer])
    for input_args in zip(*net_inputs):
      with torch.no_grad():
        f_model = self._float_model.to(device)
        f_model.eval()
        new_input_args = []
        for ip in input_args:
          if isinstance(ip, torch.Tensor):
            new_input_args.append(ip.to(device))
        _ = f_model(*new_input_args)
    torch.cuda.empty_cache()
    self.clean_hooks(handlers)
    
    for i in range(iters):
      for idx, layer_input in enumerate(layer_inputs):
        train_output = self._cached_outputs[float_layer][idx].to(device)
        qout = layer(layer_input.to(device))
        # train_output = train_output.to(device)
        
        if node in layer_act_group:
          act_node = layer_act_group[node]
          q_act_layer = act_node.module
          inplace = q_act_layer.inplace
          q_act_layer.inplace = False
          qout = q_act_layer(qout)
          q_act_layer.inplace = inplace
          if act_node.op.type == NNDCT_OP.RELU:
            train_output = F.relu(train_output)
          elif act_node.op.type == NNDCT_OP.RELU6:
            train_output = F.relu6(train_output)
          elif act_node.op.type == NNDCT_OP.HSIGMOID:
            train_output = F.hardsigmoid(train_output)
          elif act_node.op.type == NNDCT_OP.HSWISH:
            train_output = F.hardswish(train_output)
          else:
            raise       
        
        if NndctOption.nndct_quant_opt.value > 0:
          loss = F.mse_loss(qout, train_output) + F.mse_loss(layer.weight, float_layer.weight.detach().to(device))
        else:
          loss = F.mse_loss(qout, train_output)
          
        total_loss.update(loss.item())

        opt_weight.zero_grad()
        if opt_bias:
          opt_bias.zero_grad()

        loss.backward()
        opt_weight.step()
        if opt_bias:
          opt_bias.step()
    
    
      float_data = np.fabs(layer.weight.cpu().detach().numpy().flatten())
      layer.param_quantized = False
      handlers = self.hook_cache_output(last_quant_mods, hook_type="single")
      eval_loss = self.eval_loss(net_inputs, last_quant_mods, device)
      self.clean_hooks(handlers)
      quant_data = np.fabs(layer.weight.cpu().detach().numpy().flatten())
      q_noise = np.square(float_data - quant_data).mean()
      sqnr = 10 * np.log10(np.square(float_data).mean() / q_noise)
      quantize_efficiency = sqnr / 8.0
      #print(f"post quant efficiency:{quantize_efficiency}")
      #print(f"eval loss:{eval_loss} best loss:{net_loss}")
      if eval_loss < net_loss:
        best_params = self.get_layer_params(layer)
        net_loss = eval_loss
      else:
        self.set_layer_params(layer, best_params[0], best_params[1])
        break
    # self.set_layer_params(layer, best_params[0], best_params[1])
    #print(f"{node.name}\n{total_loss}")
    #print(f"opt net loss:{net_loss}")
    # self.clean_hooks()
    del self.cached_outputs[float_layer]
    # del cached_outputs
    torch.cuda.empty_cache()
    return net_loss

  @staticmethod
  def get_layer_params(layer):
    w = layer.weight.clone()
    b = None
    if hasattr(layer, "bias") and layer.bias is not None:
      b = layer.bias.clone()
      
    return w, b
  
  @staticmethod
  def set_layer_params(layer, weight, bias):
    layer.weight.data.copy_(weight)
    if bias is not None:
      layer.bias.data.copy_(bias)
    
  @property
  def quantizer(self):
    return self._quantizer

  @property
  def graph(self):
    return self._graph

  @property
  def quant_model(self):
    return self._quant_model

  @property
  def cached_outputs(self):
    return self._cached_outputs
    
  @property
  def cached_output(self):
    return self._cached_output

  @property
  def input_nodes(self):
    return self._net_input_nodes

  def finetune(self, run_fn, run_args):
    if self.quantizer.quant_mode == 2:
      NndctScreenLogger().warning(f"Finetune function will be ignored in test mode!")
      return
    NndctScreenLogger().info(f"=>Preparing data for fast finetuning module parameters ...")

    # backup option value
    opt_bak_param_corr = NndctOption.nndct_param_corr.value
    set_option_value("nndct_param_corr", 0)

    # cache input and output
    #print("**** cache input and output")
    last_quant_nodes = self.collect_last_quant_nodes()
    with torch.no_grad():
      hook_mods = []
      for node in self.graph.nodes:
        if node.op.type == NNDCT_OP.INPUT or \
        node in last_quant_nodes:
        # (self.quantizer.configer.is_node_quantizable(node, False) and 
        # len(node.op.params) > 0):
          hook_mods.append(node.module)
    
      handlers = self.hook_cache_output(hook_mods)
      
      set_option_value("nndct_quant_off", True)
      run_fn(*run_args)
      self.clean_hooks(handlers)
      
      # for mod in self.quant_model.modules():
      #   if hasattr(mod, "node") and mod.node.op.type in [NNDCT_OP.DENSE, NNDCT_OP.CONV2D, NNDCT_OP.DEPTHWISE_CONV2D, NNDCT_OP.CONVTRANSPOSE2D]:
      #     self._float_weights[mod.node].append(mod.weight.detach().cpu())

    torch.cuda.empty_cache()

    # calibration to get a set of quantization steps
    #print("****calibration to get float model tensor values")
    for mod in self.quant_model.modules():
      if hasattr(mod, "param_quantized"):
        setattr(mod, "param_quantized", False)

    # evaluation to get float model tensors 
    set_option_value("nndct_quant_off", False)
    with torch.no_grad():
      run_fn(*run_args)
    torch.cuda.empty_cache()
  
    #print("****Parameter finetuning")
    NndctScreenLogger().info(f"=>Fast finetuning module parameters for better quantization accuracy...")
    device = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_DEVICE)      
    graph_searcher = GraphSearcher(self.graph)
    node_sets = graph_searcher.find_nodes_from_type([
        PatternType(pattern=[NNDCT_OP.CONV2D, NNDCT_OP.HSWISH]),
        PatternType(pattern=[NNDCT_OP.CONV2D, NNDCT_OP.HSIGMOID]),
        PatternType(pattern=[NNDCT_OP.CONV2D, NNDCT_OP.RELU]),
        PatternType(pattern=[NNDCT_OP.CONV2D, NNDCT_OP.RELU6]),
        PatternType(pattern=[NNDCT_OP.DEPTHWISE_CONV2D, NNDCT_OP.HSWISH]),
        PatternType(pattern=[NNDCT_OP.DEPTHWISE_CONV2D, NNDCT_OP.HSIGMOID]),
        PatternType(pattern=[NNDCT_OP.DEPTHWISE_CONV2D, NNDCT_OP.RELU]),
        PatternType(pattern=[NNDCT_OP.DEPTHWISE_CONV2D, NNDCT_OP.RELU6]),
        PatternType(pattern=[NNDCT_OP.CONVTRANSPOSE2D, NNDCT_OP.HSWISH]),
        PatternType(pattern=[NNDCT_OP.CONVTRANSPOSE2D, NNDCT_OP.HSIGMOID]),
        PatternType(pattern=[NNDCT_OP.CONVTRANSPOSE2D, NNDCT_OP.RELU]),
        PatternType(pattern=[NNDCT_OP.CONVTRANSPOSE2D, NNDCT_OP.RELU6]),
        PatternType(pattern=[NNDCT_OP.CONV3D, NNDCT_OP.HSWISH]),
        PatternType(pattern=[NNDCT_OP.CONV3D, NNDCT_OP.HSIGMOID]),
        PatternType(pattern=[NNDCT_OP.CONV3D, NNDCT_OP.RELU]),
        PatternType(pattern=[NNDCT_OP.CONV3D, NNDCT_OP.RELU6]),
        PatternType(pattern=[NNDCT_OP.DEPTHWISE_CONV3D, NNDCT_OP.HSWISH]),
        PatternType(pattern=[NNDCT_OP.DEPTHWISE_CONV3D, NNDCT_OP.HSIGMOID]),
        PatternType(pattern=[NNDCT_OP.DEPTHWISE_CONV3D, NNDCT_OP.RELU]),
        PatternType(pattern=[NNDCT_OP.DEPTHWISE_CONV3D, NNDCT_OP.RELU6]),
        PatternType(pattern=[NNDCT_OP.CONVTRANSPOSE3D, NNDCT_OP.HSWISH]),
        PatternType(pattern=[NNDCT_OP.CONVTRANSPOSE3D, NNDCT_OP.HSIGMOID]),
        PatternType(pattern=[NNDCT_OP.CONVTRANSPOSE3D, NNDCT_OP.RELU]),
        PatternType(pattern=[NNDCT_OP.CONVTRANSPOSE3D, NNDCT_OP.RELU6]),
    ])
    
    layer_act_group = {}
    for _, node_list in node_sets.items():
      for nodeset in node_list:
        conv, act = nodeset
        layer_act_group[conv] = act
    
    # to avoid quantization steps change among parameter finetuning
    self.quantizer.quant_mode = 2
  
    net_inputs = []
    for node in self.input_nodes:
      cached_net_input = [out for out in self.cached_outputs[node.module]]
      net_inputs.append(cached_net_input)
      
    # last_quant_nodes = self.collect_last_quant_nodes()
    last_quant_mods = [node.module for node in last_quant_nodes]
      
    handlers = self.hook_cache_output(last_quant_mods, hook_type="single")
    net_loss = self.eval_loss(net_inputs, last_quant_mods, device)
    self.clean_hooks(handlers)
    # model.clean_hooks()
    torch.cuda.empty_cache()
    
    finetune_group = {}
    # hook_mods = []
    for qmod, fmod in zip(self._quant_model.modules(), self._float_model.modules()):
      if hasattr(qmod, "node"):
        if (self.quantizer.configer.is_node_quantizable(qmod.node, False) and 
          len(qmod.node.op.params) > 0):     
          finetune_group[qmod.node] = [qmod, fmod]

          # hook_mods.append(fmod)
    # self.hook_cache_output(hook_mods, hook_type="single")
          
    #for node, module_pair in finetune_group.items():
    for idx, (node, module_pair) in tqdm(enumerate(finetune_group.items()), total=len(finetune_group.items())):
      # if self.quantizer.configer.is_node_quantizable(node, False) and \
      #   len(node.op.params) > 0:
      quant_layer, float_layer = module_pair
      pn_node = self.graph.parents(node)[0]
      handlers = self.hook_cache_output([pn_node.module], hook_type="single")
      layer_inputs = []
      with torch.no_grad():
        for input_args in zip(*net_inputs):
          new_input_args = []
          for ip in input_args:
            if isinstance(ip, torch.Tensor):
              new_input_args.append(ip.to(device))
          _ = self.quant_model(*new_input_args)
          
          layer_inputs.append(self.cached_output[pn_node.module].detach().cpu())
      self.clean_hooks(handlers)
      del self.cached_output[pn_node.module]
      #print(f"Tuning {node.name}")
      net_loss = self.optimize_layer(node, float_layer, layer_inputs, layer_act_group, net_inputs, net_loss, last_quant_mods, device)
      del layer_inputs
      torch.cuda.empty_cache()
  
    # recover quantizer status
    for node in self.graph.nodes:
      for _, fp_history in self.quantizer.fp_history.items():
        if node.name in fp_history:
          fp_history[node.name].clear()
    for mod in self.quant_model.modules():
      if hasattr(mod, "param_quantized"):
        setattr(mod, "param_quantized", False)
    for mod in self.quant_model.modules():
      if hasattr(mod, "param_saved"):
        setattr(mod, "param_saved", False)
    self.quantizer.quant_mode = 1
    set_option_value("nndct_param_corr", opt_bak_param_corr)

    NndctScreenLogger().info(f"=>Export fast finetuned parameters ...")
    # export finetuned parameters
    self.quantizer.export_param()
 
    
  def quantize(self, run_fn, run_args):
    self.calib()
    self.finetune(run_fn, run_args)
 
