

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
import math
import nndct_shared.utils as nndct_utils
from nndct_shared.base import GLOBAL_MAP, NNDCT_KEYS, NNDCT_OP
from nndct_shared.utils import NndctOption, NndctScreenLogger, set_option_value, QError, QWarning, QNote
from pytorch_nndct.quantization import TORCHQuantizer
from collections import defaultdict
import torch.nn.functional as F
from tqdm import tqdm
import pytorch_nndct.nn as py_nn
from nndct_shared.nndct_graph import GraphSearcher
from nndct_shared.utils import PatternType
from pytorch_nndct.utils import TorchSymbol
from pytorch_nndct.utils.module_util import to_device
from .ModuleHooker import ModuleHooker

from .utils import (connect_module_with_graph,
                    disconnect_modeule_with_graph, prepare_quantizable_module,
                    register_output_hook, set_outputs_recorder_status,
                    update_nndct_blob_data, update_nndct_parameters,
                    get_deploy_graph_list)

from .adaquant_utils import tensor_size, tensor_size_by_num


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

  def __init__(self, name, size=1, fmt=':f'):
    self.name = name
    self.fmt = fmt
    self.val = [0] * size
    self.avg = [1000.] * size
    self.sum = [0] * size
    self.count = [0] * size

  def update(self, val, idx=0, n=1):
    self.val[idx] = val
    self.sum[idx] += val * n
    self.count[idx] += n
    self.avg[idx] = self.sum[idx] / self.count[idx]

  def __str__(self):
    #fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    #return fmtstr.format(**self.__dict__)
    return str(self.avg)

  def __len__(self):
    return len(self.val)

  # only support non-neagtive loss 
  def __lt__(self, y, reltol=1e-6):
    ret = True
    all_equal = True
    for idx in range(0, len(self.avg)):
      #if self.avg[idx] > y.avg[idx] * (1.+reltol):
      if self.avg[idx] > y.avg[idx]:
        ret = False
        all_equal = False
        break
      elif self.avg[idx] < y.avg[idx]:
        all_equal = False
    return (ret and not all_equal)

  # only support non-neagtive loss 
  def __eq__(self, y, reltol=1e-6):
    ret = True
    for idx in range(0, len(self.avg)):
      if not math.isclose(self.avg[idx], y.avg[idx], rel_tol=reltol):
        ret = False
        break
    return ret

class NoQuant(object):
  def __enter__(self):
    set_option_value("nndct_quant_off", True)
  
  def __exit__(self, *args):
    set_option_value("nndct_quant_off", False)

  
class AdaQuant(object):
  def __init__(self, processor):
    self._param_corr = None
    self._processor = processor
    
  def __enter__(self):
    self._param_corr = NndctOption.nndct_param_corr.value
    set_option_value("nndct_param_corr", 0)
    if NndctOption.nndct_calib_before_finetune.value is True:
      self._processor.set_keep_fp(True)
    
    
  def __exit__(self, *args):
    set_option_value("nndct_param_corr", self._param_corr)
    for node in self._processor.graph.nodes:
      for _, config_history in self._processor.quantizer.config_history.items():
        if node.name in config_history:
          config_history[node.name].clear()
    for mod in self._processor.quant_model.modules():
      if hasattr(mod, "param_quantized"):
        setattr(mod, "param_quantized", False)
    for mod in self._processor.quant_model.modules():
      if hasattr(mod, "param_saved"):
        setattr(mod, "param_saved", False)
        
    self._processor.setup_calib()
    # don't change tensors' quantization step in re-calibration after fast finetune
    if NndctOption.nndct_calib_before_finetune.value is False:
      self._processor.set_keep_fp(True)
   
   
    
    
class StopForward(Exception):
  pass


  
class AdvancedQuantProcessor(torch.nn.Module):
  r""" 
  This class re-implements the Adaquant technique proposed in the following paper.
  "Itay Hubara et al., Improving Post Training Neural Quantization: Layer-wise Calibration and Integer Programming, 
  arXiv:2006.10518, 2020."
  """
  def __init__(self, module, graph, quantizer, example_inputs):
    super().__init__()
    self._quantizer = quantizer
    #quantizer.load_param()
    self._graph = graph
    self._quant_model = module
    self._float_model = ModuleHooker.clone_quant_module(module, graph)
    self._example_inputs = example_inputs

    for mod in self._float_model.modules():
      if hasattr(mod, "node"):
        mod.node.in_quant_part = False
      
      if hasattr(mod, "quantizer"):
        mod.quantizer = None
   
    possible_last_quantized_nodes = self._get_possible_last_quant_nodes()
    
    self._batch_size = None
    self._cached_outputs = defaultdict(list)
    self._cached_output = defaultdict(list)
    self._mem_count = defaultdict(float)
    #self._net_input_nodes = [node for node in self._graph.nodes if node.op.type == NNDCT_OP.INPUT]
    self._net_input_nodes = self._graph.get_input_nodes()
    self._float_weights = defaultdict(list)
    
    self._last_quant_nodes = self.collect_last_quant_nodes(filter=lambda node: node.name in possible_last_quantized_nodes)

    self._torch_version = torch.__version__.split('.')
  
  def _setup_quantizer(self, quant_mode):
    self._quantizer.quant_mode = quant_mode
     
  def setup_calib(self):
    super().eval()
    self._setup_quantizer(quant_mode=1)

  def set_keep_fp(self, flag):
    self._quantizer.keep_fp = flag

  def setup_test(self):
    super().eval()
    self._setup_quantizer(quant_mode=2)

  def hook_cache_output(self, hook_mods, hook_type="multiple", monitor_mem=False, stop_forward=False):
    handlers = []
    if hook_type == "multiple":
      def hook(module, input, output):
        if module.node.op.type == NNDCT_OP.TUPLE_INPUT:
          self._cached_outputs[module].append([out.detach().cpu() for out in output])
          if monitor_mem is True:
            for out in output:
              self._mem_count[module] += tensor_size(out)   
        else:
          self._cached_outputs[module].append(output.detach().cpu())
          if monitor_mem is True:
            self._mem_count[module] += tensor_size(output)   
        if stop_forward is True:
          raise StopForward()
    else:
      def hook(module, input, output):
        self._cached_output[module] = output
        if monitor_mem is True:
          if module.node.op.type == NNDCT_OP.TUPLE_INPUT:
            for out in output:
              self._mem_count[module] += tensor_size(out)
          else:
            self._mem_count[module] += tensor_size(output)
        if stop_forward is True:
          raise StopForward()
      
    for module in hook_mods:
      handlers.append(module.register_forward_hook(hook))
    return handlers
  
  def hook_memory_monitor(self, hook_mods):
    
    def hook(module, input, output):
      self._mem_count[module] += tensor_size(output)
                                                
    handlers = []
    for module in hook_mods:
        handlers.append(module.register_forward_hook(hook))
    return handlers
  
  def hook_batch_size(self, hook_mods):
    def hook(module, input, output):
      if module.node.op.type == NNDCT_OP.DENSE:
        if isinstance(output, torch.Tensor) and output.ndim >= 2:
          self._batch_size = output.size()[0]
      else:
        if isinstance(output, torch.Tensor) and output.ndim >= 4:
          self._batch_size = output.size()[0]
                                                
    handlers = []
    for module in hook_mods:
        handlers.append(module.register_forward_hook(hook))
    return handlers
  
  def hook_stop_forward(self, hook_mods):
    def hook(module, input):
      raise StopForward()
    handlers = []
    for module in hook_mods:
        handlers.append(module.register_forward_pre_hook(hook))
    return handlers
    
  def collect_last_quant_nodes(self, filter=None):
    def find_last_quant_node(node, visited=None, quant_nodes=None):
      if node in visited:
        return
      visited.add(node)
      if self.quantizer.configer.is_node_quantizable(node, self.quantizer.lstm) and node.in_nodes:
        if filter is None or (filter is not None and filter(node) is True):
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
          elif isinstance(ip, (list, tuple)):
            ip = [item.to(device) for item in ip if isinstance(item, torch.Tensor)]
            new_input_args.append(ip)
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
          elif isinstance(ip, (list, tuple)):
            ip = [item.to(device) for item in ip if isinstance(item, torch.Tensor)]
            new_input_args.append(ip)
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
            raise NotImplementedError()
                   
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
      # print(f"eval loss:{eval_loss} best loss:{net_loss}")
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
    # print(f"iter:{i}")
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
      NndctScreenLogger().warning2user(QWarning.FINETUNE_IGNORED, f"Finetune function will be ignored in test mode!")
      return
    NndctScreenLogger().info(f"=>Preparing data for fast finetuning module parameters ...")
    
    # memory status
    total_m, *_, available_m = list(map(lambda x: x/1024, map(int, os.popen('free -t -m').readlines()[1].split()[1:])))
    NndctScreenLogger().info(f"Mem status(total mem: {total_m:.2f}G, available mem: {available_m:.2f}G).")
    
    NndctScreenLogger().info(f"=>Preparing data for fast finetuning module parameters ...")
    # backup option value
    opt_bak_param_corr = NndctOption.nndct_param_corr.value
    set_option_value("nndct_param_corr", 0)

    # cache input and output
    #print("**** cache input and output")
    last_quant_nodes = self.collect_last_quant_nodes()
    with torch.no_grad():
      cache_layers = []
      monitor_layers = []
      for node in self.graph.nodes:
        if node.op.type == NNDCT_OP.INPUT or node in last_quant_nodes:
          cache_layers.append(node.module)
        elif self.quantizer.configer.is_conv_like(node):
          monitor_layers.append(node.module)
          
      monitor_handlers = self.hook_memory_monitor(monitor_layers)
      cache_handlers = self.hook_cache_output(cache_layers, monitor_mem=True)
      set_option_value("nndct_quant_off", True)
      run_fn(*run_args)
      # memory statistics      
      total_memory_cost = 0.0
      for layer in cache_layers:
        total_memory_cost += self._mem_count[layer]
        del self._mem_count[layer]
        
      total_memory_cost += 2 * max(self._mem_count.values())
     
      self.clean_hooks(monitor_handlers + cache_handlers)
    
    torch.cuda.empty_cache()
    NndctScreenLogger().info(f"Mem cost by fast finetuning: {total_memory_cost:.2f}G.")
    if total_memory_cost > 0.8 * available_m:
        NndctScreenLogger().warning2user(QWarning.MEMORY_SHORTAGE, f"There is not enought memory for fast finetuning and this process will be ignored!.Try to use a smaller calibration dataset.")
        return 
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
            elif isinstance(ip, (list, tuple)):
              ip = [item.to(device) for item in ip if isinstance(item, torch.Tensor)]
              new_input_args.append(ip)
          _ = self.quant_model(*new_input_args)
          
          layer_inputs.append(self.cached_output[pn_node.module].detach().cpu())
      self.clean_hooks(handlers)
      del self.cached_output[pn_node.module]
      #print(f"Tuning {node.name}")
      net_loss = self.optimize_layer(node, float_layer, layer_inputs, layer_act_group, net_inputs, net_loss, last_quant_mods, device)
      # print(f"{node.name}:{net_loss}")
      del layer_inputs
      torch.cuda.empty_cache()
  
    # recover quantizer status
    for node in self.graph.nodes:
      for _, config_history in self.quantizer.config_history.items():
        if node.name in config_history:
          config_history[node.name].clear()
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
 
  def cache_net_inpouts(self, run_fn, run_args):
    total_m, *_, available_m = list(map(lambda x: x / 1024, map(int, os.popen('free -t -m').readlines()[1].split()[1:])))
    NndctScreenLogger().info(f"Mem status(total mem: {total_m:.2f}G, available mem: {available_m:.2f}G).")
    cache_layers = []
    monitor_layers = []
    batch_layers = []
    
    for node in self.graph.nodes:
      # if node.op.type == NNDCT_OP.INPUT or node in end_nodes:
      if node.op.type == NNDCT_OP.INPUT or node.op.type == NNDCT_OP.TUPLE_INPUT or node in self._last_quant_nodes:
        cache_layers.append(node.module)
      
      if self.quantizer.configer.is_conv_like(node):
        monitor_layers.append(node.module)
        if not batch_layers:
          batch_layers.append(node.module)
   
    monitor_handlers = self.hook_memory_monitor(monitor_layers)
    batch_handlers = self.hook_batch_size(batch_layers)
    cache_handlers = self.hook_cache_output(cache_layers, monitor_mem=True)
    with torch.no_grad():
      run_fn(*run_args)
      # memory statistics      
    total_memory_cost = 0.0
    for layer in cache_layers:
      total_memory_cost += self._mem_count[layer]
      del self._mem_count[layer]
      
    NndctScreenLogger().info(f"Memory cost by fast finetuning is {total_memory_cost:.2f} G.")
    if total_memory_cost > 0.8 * available_m:
        NndctScreenLogger().warning2user(QWarning.MEMORY_SHORTAGE, f"There is not enought memory for fast finetuning and this process will be ignored!.Try to use a smaller calibration dataset.")
        return 
    self.clean_hooks(monitor_handlers + cache_handlers + batch_handlers)
    net_inputs = []
    for node in self.input_nodes:
      cached_net_input = [out for out in self.cached_outputs[node.module]]
      net_inputs.append(cached_net_input)
      del self.cached_outputs[node.module]
    
    net_outputs = {}
    for node in self._last_quant_nodes:
      cached_net_output = [out for out in self.cached_outputs[node.module]]
      net_outputs[node.module] = cached_net_output
      del self.cached_outputs[node.module]
      
    torch.cuda.empty_cache() 
    return net_inputs, net_outputs
      
      
      
  def calibrate(self, run_fn, run_args):
    self.setup_calib()
    for mod in self.quant_model.modules():
      if hasattr(mod, "param_quantized"):
        setattr(mod, "param_quantized", False)

    with torch.no_grad():
      run_fn(*run_args)
    torch.cuda.empty_cache()
  
  def collect_layer_act_pair(self):
    graph_searcher = GraphSearcher(self.graph)
    patterns = []
    
    if NndctOption.nndct_ip_asr.value:
      tuning_ops = [NNDCT_OP.CONV2D, NNDCT_OP.DENSE]
      #tail_act_ops = [NNDCT_OP.ADD, NNDCT_OP.RESHAPE, NNDCT_OP.LAYER_NORM, NNDCT_OP.RELU]
      tail_act_ops = [NNDCT_OP.RELU]
    else:
      tuning_ops = [NNDCT_OP.CONV2D, NNDCT_OP.DEPTHWISE_CONV2D, NNDCT_OP.CONVTRANSPOSE2D, 
                    NNDCT_OP.CONV3D, NNDCT_OP.DEPTHWISE_CONV3D, NNDCT_OP.CONVTRANSPOSE3D]
      tail_act_ops = [NNDCT_OP.RELU, NNDCT_OP.RELU6, NNDCT_OP.HSWISH, NNDCT_OP.HSIGMOID]
    for tuning_op in tuning_ops:
      for act_op in tail_act_ops:
        patterns.append(PatternType(pattern=[tuning_op, act_op]))    
    node_sets = graph_searcher.find_nodes_from_type(patterns)
    layer_act_group = {}
    for _, node_list in node_sets.items():
      for nodeset in node_list:
        conv, act = nodeset
        layer_act_group[conv] = act
    return layer_act_group
  

      
      
  def calc_net_loss(self, net_inputs, net_outputs, device):
    last_quant_mods = [node.module for node in self._last_quant_nodes]
    handlers = self.hook_cache_output(last_quant_mods, hook_type="single")
    loss = AverageMeter("loss", size=len(last_quant_mods))
    with torch.no_grad():
      for idx, input_args in enumerate(zip(*net_inputs)):
        new_input_args = []
        for ip in input_args:
          if isinstance(ip, torch.Tensor):
            new_input_args.append(ip.to(device))
          elif isinstance(ip, (list, tuple)):
            ip = [item.to(device) for item in ip if isinstance(item, torch.Tensor)]
            new_input_args.append(ip)
        #new_input_args = [ip.to(device) for ip in input_args if isinstance(ip, torch.Tensor)]
        # _ = fmodel(*new_input_args)
        _ = self.quant_model(*new_input_args)
        
        loss_idx = 0
        for mod in last_quant_mods:
          loss_mod = math.sqrt(F.mse_loss(net_outputs[mod][idx].to(device), self._cached_output[mod]).item())
          loss.update(loss_mod, idx=loss_idx)
          loss_idx += 1
    self.clean_hooks(handlers)
    return loss
  
  
  def finetune_v2(self, run_fn, run_args):
    # check status
    if self.quantizer.quant_mode == 2:
      NndctScreenLogger().warning2user(QWarning.FINETUNE_IGNORED, f"Finetune function will be ignored in test mode!")
      return    
    
    # parameter finetuning

    #import ipdb
    #ipdb.set_trace()
    with AdaQuant(processor=self):
      # calibration to get a set of quantization steps
      NndctScreenLogger().info(f"=>Preparing data for fast finetuning module parameters ...")   
      with NoQuant():
        net_inputs, net_outputs = self.cache_net_inpouts(run_fn, run_args)
      
      NndctScreenLogger().info(f"=>Find initial quantization steps for fast finetuning...")
      self.calibrate(run_fn, run_args)
      
      NndctScreenLogger().info(f"=>Fast finetuning module parameters for better quantization accuracy...")
      self.setup_test()    
      device = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_DEVICE)  
      
      intial_net_loss = self.calc_net_loss(net_inputs, net_outputs, device)
      
      layer_act_pair = self.collect_layer_act_pair()  
      
      finetune_group = []
      for qmod, fmod in zip(self._quant_model.modules(), self._float_model.modules()):
        if hasattr(qmod, "node"):
          #if (self.quantizer.configer.is_node_quantizable(qmod.node, self.quantizer.lstm) and 
          if (self.quantizer.configer.is_conv_like(qmod.node) and
            len(qmod.node.op.params) > 0):
            finetune_group.append([qmod.node, fmod])

      net_loss = intial_net_loss
      for idx, (qnode, fmod) in tqdm(enumerate(finetune_group), total=len(finetune_group)):
        is_cached = self.is_cached(qnode, len(net_inputs[0]))
        if (is_cached and idx < len(finetune_group) / 2) or (not is_cached):
          need_cache = False
        else:
          need_cache = True
        net_loss = self.optimize_layer_v2(qnode, fmod, layer_act_pair, net_inputs, net_outputs, net_loss, device, need_cache)
      #print(f"%%%%%%%%%%%%%%%%% final opt net loss:{net_loss.avg}")

        # print(f"{qnode.name}({need_cache}):{net_loss}")
            
    #NndctScreenLogger().info(f"=>Export fast finetuned parameters ...")
    # export finetuned parameters
    #self.quantizer.export_param()
  
  def is_cached(self, node, iters):
    *_, available_m = list(map(lambda x: x/1024, map(int, os.popen('free -t -m').readlines()[1].split()[1:])))
    if self.graph.node(node.in_nodes[0]).out_tensors[0].shape is None:
      input_numel = np.prod([self._batch_size])
    else:
      input_numel = np.prod([self._batch_size] + self.graph.node(node.in_nodes[0]).out_tensors[0].shape[1:])
    if node.out_tensors[0].shape is None:
      output_numel = np.prod([self._batch_size])
    else:
      output_numel = np.prod([self._batch_size] + node.out_tensors[0].shape[1:])
    size = tensor_size_by_num((input_numel + output_numel) * iters)
    if size > 0.8 * available_m:
      return False
    else:
      return True
    
    
  @staticmethod
  def quant_eff(qdata, fdata):
    q_noise = torch.mean(torch.pow(fdata - qdata, 2))
    sqnr = 10 * torch.log10(torch.mean(torch.pow(fdata, 2)) / q_noise)
    return sqnr / 8.0
     
  def optimize_layer_v2(self, qnode, float_layer, layer_act_pair, net_inputs, net_outputs, net_loss, device, need_cache):
    batch_factor = 0.5 if self._batch_size == 1 else 1
    layer = qnode.module
    quantize_efficiency = self.quant_eff(layer.weight.data, float_layer.weight.data)
    
    weight_grad = None
    bias_grad = None
    if layer.weight.requires_grad == False:
      weight_grad = False
      layer.weight.requires_grad_(requires_grad = True)
    
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
      if layer.bias.requires_grad == False:
        bias_grad = False
        layer.bias.requires_grad_(requires_grad = True)
      if layer.bias.flatten().shape[0] == 1: lr_b = 0.0
      else: lr_b = lr_factor * layer.bias.std().item()
      # lr_b = lr_factor * layer.bias.std().item()
      # lr_b=1e-3
      opt_bias = torch.optim.Adam([layer.bias], lr=lr_b)
    
    in_shape = layer.weight.shape
    if layer.node.op.type == NNDCT_OP.DENSE:
      fake_in = torch.rand(1, layer.in_features, device=layer.weight.device)
    else:
      fake_in = torch.rand(1, layer.in_channels, *in_shape[2:], device=layer.weight.device)
    if layer(fake_in).grad_fn is None:
      NndctScreenLogger().error(f'Layer to do fast finetune does not contain grad_fn attribute, \
please remove it if there is "torch.no_grad()" in forward process')
      exit(2)
      
    #print(f"learning rate: lr_w={lr_w}, lr_b={lr_b}")
    #print(f"pre quant efficiency:{quantize_efficiency}")
    if NndctOption.nndct_ip_asr.value: 
      iters = 25
    else:
      iters = 100
    total_loss = AverageMeter("layer_loss")
    best_params = self.get_layer_params(layer)
    # torch version >= 1.6
    if int(self._torch_version[0]) >= 1 and int(self._torch_version[1]) >= 6:
      act_func_map = {
        NNDCT_OP.RELU: F.relu,
        NNDCT_OP.RELU6: F.relu6,
        NNDCT_OP.HSIGMOID: F.hardsigmoid,
        NNDCT_OP.HSWISH: F.hardswish,
      }
    else:
      act_func_map = {
        NNDCT_OP.RELU: F.relu,
        NNDCT_OP.RELU6: F.relu6,
      }
    
    prev_loss = AverageMeter("loss", size=len(net_loss))
    if not need_cache:
      for i in range(iters):
        #print(f"--------iter={i}-----------------")
        stop_handlers = self.hook_cache_output([float_layer, self.graph.node(qnode.in_nodes[0]).module], hook_type='single', stop_forward=True)
        for input_args in zip(*net_inputs):
          new_input_args = []
          for ip in input_args:
            if isinstance(ip, torch.Tensor):
              new_input_args.append(ip.to(device))
            elif isinstance(ip, (list, tuple)):
              ip = [item.to(device) for item in ip if isinstance(item, torch.Tensor)]
              new_input_args.append(ip)
          #new_input_args = [ip.to(device) for ip in input_args if isinstance(ip, torch.Tensor)]
          
          with torch.no_grad():
            try:
              _ = self._float_model(*new_input_args)
            except StopForward:
              pass
          
            try:
              _ = self.quant_model(*new_input_args)
            except StopForward:
              pass
          
          qin = self._cached_output[self.graph.node(qnode.in_nodes[0]).module]
          qout = layer(qin)
          fout = self._cached_output[float_layer]
          if qnode in layer_act_pair:
            act_node = layer_act_pair[qnode]
            q_act_layer = act_node.module
            inplace = q_act_layer.inplace
            q_act_layer.inplace = False
            qout = q_act_layer(qout)
            q_act_layer.inplace = inplace
            fout = act_func_map[act_node.op.type](fout)
            
          if NndctOption.nndct_quant_opt.value > 0:
            loss = F.mse_loss(qout, fout) + F.mse_loss(layer.weight, float_layer.weight.detach())
          else:
            loss = F.mse_loss(qout, fout)
            
          total_loss.update(loss.item())

          opt_weight.zero_grad()
          if opt_bias:
            opt_bias.zero_grad()

          loss.backward()
          #print('weights lr: {}'.format(opt_weight.param_groups[0]['lr']))
          opt_weight.step()
          if opt_bias:
            #print('bias lr: {}'.format(opt_bias.param_groups[0]['lr']))
            opt_bias.step()
            #print(f'{layer.name} bias after backward: {layer.bias[0]}', flush=True)
            
        self.clean_hooks(stop_handlers)    
        float_data = float_layer.weight.clone().detach()
        layer.param_quantized = False
        eval_loss = self.calc_net_loss(net_inputs, net_outputs, device)
        ''' 
        quant_data = layer.weight.detach()
        quantize_efficiency = self.quant_eff(quant_data, float_data)
        print(f"post quant efficiency:{quantize_efficiency}")
        print('weights/bias lr: {} / {}'.format(
            opt_weight.param_groups[0]['lr'],
            opt_bias.param_groups[0]['lr']), flush=True)
        print(f"eval loss:{eval_loss.avg} best loss:{net_loss.avg}", flush=True)
        '''
        loss_diff_ratio = ((np.array(net_loss.avg)-np.array(eval_loss.avg))/np.array(net_loss.avg))*10000
        #print(f"loss_diff_ratio :{loss_diff_ratio}", flush=True)
       
        if NndctOption.nndct_ip_asr.value:
          loss_ratio_threshold = 5.0
        else:
          loss_ratio_threshold = 0.0
        
        if (eval_loss < net_loss) and (loss_diff_ratio.max() > loss_ratio_threshold):
          #print(f"update parameters", flush=True)
          best_params = self.get_layer_params(layer)
          net_loss = eval_loss
        else:
          #print(f"update learning rate", flush=True)
          self.set_layer_params(layer, best_params[0], best_params[1])
          opt_weight.param_groups[0]['lr'] /= 2. 
          if opt_bias:
            opt_bias.param_groups[0]['lr'] /= 2.
          if ((opt_weight.param_groups[0]['lr'] < 1e-5 and
              (opt_bias == None or opt_bias.param_groups[0]['lr'] < 1e-5))
              or eval_loss == prev_loss):
            break
          #del prev_loss
          prev_loss = eval_loss
    else:
      handlers = self.hook_cache_output([float_layer, self.graph.parents(qnode)[0].module], stop_forward=True)
      with torch.no_grad():
        for input_args in zip(*net_inputs):
          new_input_args = []
          for ip in input_args:
            if isinstance(ip, torch.Tensor):
              new_input_args.append(ip.to(device))
            elif isinstance(ip, (list, tuple)):
              ip = [item.to(device) for item in ip if isinstance(item, torch.Tensor)]
              new_input_args.append(ip)
          #new_input_args = [ip.to(device) for ip in input_args if isinstance(ip, torch.Tensor)]
          try:
            _ = self._float_model(*new_input_args)
          except StopForward:
            pass
          
          try:
            _ = self.quant_model(*new_input_args)
          except StopForward:
            pass
            
      torch.cuda.empty_cache()
      self.clean_hooks(handlers)
      for i in range(iters):
        #print(f"--------iter={i}-----------------")
        for idx, layer_input in enumerate(self._cached_outputs[self.graph.parents(qnode)[0].module]):
          fout = self._cached_outputs[float_layer][idx].to(device)
          qout = layer(layer_input.to(device))
          
          if qnode in layer_act_pair:
            act_node = layer_act_pair[qnode]
            q_act_layer = act_node.module
            inplace = q_act_layer.inplace
            q_act_layer.inplace = False
            qout = q_act_layer(qout)
            q_act_layer.inplace = inplace
            fout = act_func_map[act_node.op.type](fout)   
          
          if NndctOption.nndct_quant_opt.value > 0:
            loss = F.mse_loss(qout, fout) + F.mse_loss(layer.weight, float_layer.weight.detach().to(device))
          else:
            loss = F.mse_loss(qout, fout)
            
          total_loss.update(loss.item())

          opt_weight.zero_grad()
          if opt_bias:
            opt_bias.zero_grad()

          loss.backward()
          #print('weights lr: {}'.format(opt_weight.param_groups[0]['lr']))
          opt_weight.step()
          if opt_bias:
            #print('bias lr: {}'.format(opt_bias.param_groups[0]['lr']))
            opt_bias.step()
    
        float_data = np.fabs(float_layer.weight.cpu().detach().numpy().flatten())
        layer.param_quantized = False
        eval_loss = self.calc_net_loss(net_inputs, net_outputs, device)
        '''
        print('----------------loss after backward-----------------')
        quant_data = np.fabs(layer.weight.cpu().detach().numpy().flatten())
        q_noise = np.square(float_data - quant_data).mean()
        sqnr = 10 * np.log10(np.square(float_data).mean() / q_noise)
        quantize_efficiency = sqnr / 8.0
        print(f"post quant efficiency:{quantize_efficiency}")
        print('weights/bias lr: {} / {}'.format(
            opt_weight.param_groups[0]['lr'],
            opt_bias.param_groups[0]['lr']), flush=True)
        print(f"eval loss:{eval_loss.avg} best loss:{net_loss.avg}", flush=True)
        '''
        loss_diff_ratio = ((np.array(net_loss.avg)-np.array(eval_loss.avg))/np.array(net_loss.avg))*10000
        #print(f"loss_diff_ratio :{loss_diff_ratio}", flush=True)
        
        if NndctOption.nndct_ip_asr.value:
          loss_ratio_threshold = 5.0
        else:
          loss_ratio_threshold = 0.0
        
        if (eval_loss < net_loss) and (loss_diff_ratio.max() > loss_ratio_threshold):
          #print(f"update parameters", flush=True)
          best_params = self.get_layer_params(layer)
          net_loss = eval_loss
        else:
          #print(f"update learning rate", flush=True)
          opt_weight.param_groups[0]['lr'] /= 2.
          if opt_bias:
            opt_bias.param_groups[0]['lr'] /= 2.
          self.set_layer_params(layer, best_params[0], best_params[1])
          if ((opt_weight.param_groups[0]['lr'] < 1e-5 and
              (opt_bias == None or opt_bias.param_groups[0]['lr'] < 1e-5))
              or eval_loss == prev_loss):
            break
          #del prev_loss
          prev_loss = eval_loss
    # self.set_layer_params(layer, best_params[0], best_params[1])
    #print(f"{node.name}\n{total_loss}")
      del self.cached_outputs[float_layer]
      del self.cached_outputs[self.graph.parents(qnode)[0].module]
    #print(f"%%%%%%%%%%%%%%%%% final opt net loss:{net_loss}")
    
    if weight_grad is not None:
      layer.weight.requires_grad_(requires_grad=weight_grad)
    if bias_grad is not None:
      layer.bias.requires_grad_(requires_grad=bias_grad)
    
    torch.cuda.empty_cache()
    # print(f"iter:{i}")
    return net_loss 
 
  def _get_possible_last_quant_nodes(self):
    device = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_DEVICE)
    f_model, input_args = to_device(self._float_model, self._example_inputs, device)
    handlers = ModuleHooker.register_tensor_dtype_and_shape_hook(f_model)
    f_model.eval()
    with NoQuant():
      if isinstance(input_args, tuple):
        _ = f_model(*input_args)
      else:
        _ = f_model(input_args)
    possible_last_quantized_nodes = []
    for fmod in f_model.modules():
      if hasattr(fmod, "node"):
        if (len(fmod.node.out_tensors) > 1 or 
          fmod.node.out_tensors[0].is_complete_tensor() is False or
          fmod.node.out_tensors[0].dtype not in self._quantizer.configer.QUANTIZABLE_DTYPES):
          continue
        possible_last_quantized_nodes.append(fmod.node.name)

    self.clean_hooks(handlers)
    return possible_last_quantized_nodes
