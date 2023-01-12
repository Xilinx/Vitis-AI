

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

import copy
import os
import warnings
from collections import defaultdict
from typing import Any, Optional, Sequence, Union, List

import torch

import nndct_shared.utils as nndct_utils
import pytorch_nndct.nn.modules.rnn_builder as rnn_builder
from nndct_shared.base import GLOBAL_MAP, NNDCT_KEYS, NNDCT_OP
from nndct_shared.utils import create_work_dir, option_util, NndctOption, NndctScreenLogger, QError, QWarning, QNote
#from nndct_shared.quantization import DefaultQstrategy, QstrategyFactory
from nndct_shared.compile import CompilerFactory, DeployChecker
from nndct_shared.nndct_graph import (merge_multi_subgraphs,
                                      reorder_multi_subgraph_nodes, merge_multi_graphs_to_single_graph)
from pytorch_nndct.parse import NodeTransformer
from pytorch_nndct.quantization import TORCHQuantizer
from pytorch_nndct.utils import TorchSymbol
from pytorch_nndct.quantization import RNNTorchQConfig, TorchQConfig
from .utils import (connect_module_with_graph,
                    parse_module, recreate_nndct_module,
                    set_outputs_recorder_status, update_nndct_blob_data, register_output_hook,
                    convert_lstm, prepare_quantizable_module, has_lstm, register_input_checker)

from .base import TorchQuantProcessor
from pytorch_nndct.utils.jit_utils import optimize_graph
from pytorch_nndct.utils.module_util import to_device


class LSTMTorchQuantProcessor(TorchQuantProcessor):
    
  def _check_args(self, module):
    if not isinstance(module, torch.nn.Module):
      raise TypeError(f"type of 'module' should be 'torch.nn.Module'.")
    
  def __init__(self,
               quant_mode: str,
               module: torch.nn.Module,
               input_args: Union[torch.Tensor, Sequence[Any]] = None,
               state_dict_file: Optional[str] = None,
               output_dir: str = "quantize_result",
               bitwidth_w: int = 8,
               bitwidth_a: int = 8,
               device: torch.device = torch.device("cuda"),
               lstm_app: bool = True,
               quant_config_file: Optional[str] = None):
    self._export_folder = output_dir
    # Check arguments type
    self._check_args(module)
    
    # Check device available
    if device.type == "cuda":
      #if not (torch.cuda.is_available() and "CUDA_HOME" in os.environ):
      if not (torch.cuda.is_available() and ("CUDA_HOME" or "ROCM_HOME" in os.environ)):
        device = torch.device("cpu")
        NndctScreenLogger().warning2user(QWarning.CUDA_UNAVAILABLE, f"CUDA (HIP) is not available, change device to CPU")
    
    # Transform torch module to quantized module format
    nndct_utils.create_work_dir(output_dir)

    # turn off weights equalization and bias correction
    if (hasattr(NndctOption.nndct_param_corr, '_value')):
      option_util.set_option_value("nndct_param_corr", NndctOption.nndct_param_corr._value)
    else:
      option_util.set_option_value("nndct_param_corr", False)
      
    if (hasattr(NndctOption.nndct_equalization, '_value')):
      option_util.set_option_value("nndct_equalization", NndctOption.nndct_equalization._value)
    else:
      option_util.set_option_value("nndct_equalization", False)
    
    option_util.set_option_value("nndct_cv_app", False)

    # Parse the quant config file
    QConfiger = RNNTorchQConfig()
    #if quant_config_file:
    QConfiger.parse_config_file(quant_config_file, 
                                bit_width_w = bitwidth_w, 
                                bit_width_a = bitwidth_a)
    qconfig = QConfiger.qconfig
    quantizer, qmode = self._init_quant_env(quant_mode, 
                                            output_dir,
                                            qconfig,
                                            is_lstm = True)

    GLOBAL_MAP.set_map(NNDCT_KEYS.QUANTIZER, quantizer)
    GLOBAL_MAP.set_map(NNDCT_KEYS.QUANT_MODE, qmode)
    GLOBAL_MAP.set_map(NNDCT_KEYS.QUANT_DEVICE, device)
    GLOBAL_MAP.set_map(NNDCT_KEYS.QUANT_CONFIG, qconfig)
    
    standard_RNNs, customized_RNNs = self._analyse_module(module)

    if len(standard_RNNs) == 0 and len(customized_RNNs) == 0:
      raise RuntimeError(
          f"The top module '{module._get_name()}' should have one LSTM module at least."
      )

    self._modules_info = defaultdict(dict)

    # process customized Lstm
    for layer_name, layer_module in customized_RNNs.items():
      for cell_name, cell_module in layer_module.named_children():
        lstm_direction = "forward" if layer_module.go_forward else "backward"
        full_cell_name = ".".join([layer_name, cell_name])
        layer_graph = self._get_customized_LSTM_graph(full_cell_name,
                                                      cell_module,
                                                      layer_module.input_size,
                                                      layer_module.hidden_size,
                                                      layer_module.memory_size)
        self._modules_info[full_cell_name]["layers_graph"] = [{
            lstm_direction: layer_graph
        }]
        self._modules_info[full_cell_name]["stack_mode"] = None
        self._modules_info[full_cell_name]["layer_module"] = layer_module

    # process standard Lstm
    for name, rnn_module in standard_RNNs.items():
      layers_graph = self._get_standard_RNN_graph(
          graph_name=name, lstm_module=rnn_module)
      self._modules_info[name]["layers_graph"] = layers_graph
      self._modules_info[name]["input_size"] = [rnn_module.input_size
                                                ] * rnn_module.num_layers
      self._modules_info[name]["hidden_size"] = [rnn_module.hidden_size
                                                 ] * rnn_module.num_layers
      self._modules_info[name]["memory_size"] = [rnn_module.hidden_size
                                                 ] * rnn_module.num_layers
      self._modules_info[name][
          "stack_mode"] = "bidirectional" if rnn_module.bidirectional else "unidirectional"
      self._modules_info[name][
          "batch_first"] = True if rnn_module.batch_first is True else False

      if rnn_module.mode == 'LSTM':
        self._modules_info[name]["mode"] = "LSTM"
      elif rnn_module.mode == "GRU": 
        self._modules_info[name]["mode"] = "GRU"
    # merge multi graphs into a graph
    top_graph = self._merge_subgraphs()
    
    # turn on quantizer
    #if quant_mode:
    quantizer.setup(top_graph, rnn_front_end=True, lstm=True)
    
    # write and reload quantizable cell module
    module_graph_map = self._rebuild_layer_module()
    
    # replace float module with quantizale module
    for name, info in self._modules_info.items():
      if info["stack_mode"] is not None:
        self._build_stack_lstm_module(info)
      else:
        info["QLSTM"] = list(info["layers_module"][0].values())[0]
      module = self._insert_QuantLstm_in_top_module(module, name, info)

    # move modules info into layers info
    self._convert_modules_info_to_layers(module_graph_map)

    # hook module with quantizer
    # connect_module_with_quantizer(quant_module, quantizer)
    quantizer.quant_model = module.to(device)

    self.quantizer = quantizer

  # function needs forwarding iteration control
  def quantize(self, run_fn, run_args):
    pass

  # export xmodel file to be compiled for deployment
  def export_xmodel(self, output_dir, deploy_check):
    self.dump_xmodel(output_dir, deploy_check)
    
  @staticmethod
  def _analyse_module(top_module):
    standard_RNNs = {}
    customized_RNNs = {}
    for name, sub_module in top_module.named_modules():
      if isinstance(sub_module, rnn_builder.QuantLstmLayer) or isinstance(sub_module, rnn_builder.QuantGruLayer):
        customized_RNNs[name] = sub_module
      elif isinstance(sub_module, torch.nn.LSTM) or isinstance(sub_module, torch.nn.GRU):
        standard_RNNs[name] = sub_module

    return standard_RNNs, customized_RNNs
  
  def _convert_modules_info_to_layers(self, module_graph_map):
    self._layers_info = defaultdict(dict)

    for name, info in self._modules_info.items():
      if isinstance(info["QLSTM"], rnn_builder.StackedLstm) or isinstance(info["QLSTM"], rnn_builder.StackedGru):
        for sub_name, sub_m in info["QLSTM"].named_children():
          if isinstance(sub_m, rnn_builder.Lstm) or isinstance(sub_m, rnn_builder.Gru):
            lstm_name = ".".join([name, sub_name])
            for layer_name, layer_module in sub_m.named_children():
              if isinstance(layer_module, rnn_builder.QuantLstmLayer) or isinstance(layer_module, rnn_builder.QuantGruLayer):
                layer_name = ".".join([lstm_name, layer_name])
                for cell_name, cell_module in layer_module.named_children():
                  full_cell_name = ".".join([layer_name, cell_name])
                  self._layers_info[full_cell_name]["cell_module"] = cell_module
                  self._layers_info[full_cell_name]["graph"] = module_graph_map[
                      id(cell_module)]
                  self._layers_info[full_cell_name][
                      "layer_module"] = layer_module
      else:
        self._layers_info[name]["cell_module"] = info["QLSTM"]
        self._layers_info[name]["graph"] = list(
            info["layers_graph"][0].values())[0]
        self._layers_info[name]["layer_module"] = info["layer_module"]
  
  @staticmethod
  def _insert_QuantLstm_in_top_module(top_module, module_name, info):
    module = top_module
    for sub_module_name in module_name.split(".")[:-1]:
      module = getattr(module, sub_module_name)
    if module_name.split(".")[-1]:
      setattr(module, module_name.split(".")[-1], info["QLSTM"])
    else:
      top_module = info["QLSTM"]
    return top_module

  @staticmethod
  def _build_stack_lstm_module(info):
    builder = rnn_builder.DynamicRnnBuilder()
    info["QLSTM"] = builder(
        rnn_type=info["mode"],
        input_sizes=info["input_size"],
        hidden_sizes=info["hidden_size"],
        memory_sizes=info["memory_size"],
        layers=info["layers_module"],
        stack_mode=info["stack_mode"],
        batch_first=info["batch_first"])

  # def _hook_quant_module_with_quantizer(self, quantizer):
  #    for _, info in self._modules_info.items():
  #     for layer in info["layers_module"]:
  #       for direction, quant_module in layer.items():
  #         connect_module_with_quantizer(quant_module, quantizer)
         
  def _rebuild_layer_module(self):
    module_graph_map = {}
    for name, info in self._modules_info.items():
      layers_module = []
      for l_num, layer_graph in enumerate(info["layers_graph"]):
        lstm_cell_pair = {}
        for lstm_direction, graph in layer_graph.items():
          export_file = os.path.join(
              self._export_folder, f"{graph.name}{TorchSymbol.SCRIPT_SUFFIX}")
          quant_module = recreate_nndct_module(graph, True, export_file)
          connect_module_with_graph(quant_module, graph)
          lstm_cell_pair[lstm_direction] = quant_module
          module_graph_map[id(quant_module)] = graph
        layers_module.append(lstm_cell_pair)
      self._modules_info[name]["layers_module"] = layers_module
    return module_graph_map

  def _merge_subgraphs(self):
    subgraphs = []
    for info in self._modules_info.values():
      for layer_graph in info["layers_graph"]:
        for _, graph in layer_graph.items():
          subgraphs.append(graph)
    reorder_multi_subgraph_nodes(subgraphs)
    top_graph = merge_multi_subgraphs(subgraphs)
    return top_graph

  def _get_customized_LSTM_graph(self, graph_name, lstm_cell, input_size,
                                 hidden_size, memory_size):
    name_list = graph_name.replace(".", "_").split("_")
    name_gen = (w.capitalize() for w in name_list)
    graph_name = "".join(name_gen)
    input = torch.randn(1, input_size)
    h, c = torch.zeros(1, hidden_size), torch.zeros(1, memory_size)
    cell_graph = parse_module(lstm_cell.cpu(), (input, h, c), graph_name=graph_name)
    return cell_graph

  def _get_standard_RNN_graph(self, graph_name, lstm_module):
    name_list = graph_name.replace(".", "_").split("_")
    name_gen = (w.capitalize() for w in name_list)
    graph_name = "".join(name_gen)
    inputs = torch.randn(1, 2, lstm_module.input_size)
    lstm_nndct_graph = parse_module(lstm_module.cpu(), inputs, enable_opt=False, graph_name=graph_name)
    lstm_node = None
    for node in lstm_nndct_graph.nodes:
      if node.op.type in [NNDCT_OP.BASIC_LSTM, NNDCT_OP.BASIC_GRU]:
        lstm_node = node
    transform = NodeTransformer()
    assert lstm_node
    layers = transform(lstm_node)
    return layers

  def dump_xmodel(self, output_dir="quantize_result", deploy_check=False):
    """
    `dump xmodel for LSTM cell`
    """
    quantizer = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANTIZER)
    if quantizer and quantizer.quant_mode > 1:
      compiler = CompilerFactory.get_compiler("xmodel")
      xmodel_dir = os.path.join(output_dir, "xmodel")
      create_work_dir(xmodel_dir)
      for info in self._modules_info.values():
        for l_num, layer_graph in enumerate(info["layers_graph"]):
          for lstm_direction, graph in layer_graph.items():
            try:
              compiler.do_compile(
                  graph,
                  quant_config_info=quantizer.quant_config,
                  output_file_name=os.path.join(xmodel_dir, graph.name),
                  graph_attr_kwargs={"direction": lstm_direction})
            except Exception as e:
              print(
                  f"[NNDCT_ERROR]:failed convert nndct graph to xmodel({str(e)})."
              )

      if deploy_check:
        print("[NNDCT_NOTE]: Dumping checking data...")
        checker = DeployChecker(
            output_dir_name=output_dir, data_format="txt")     
        
        # get timestep output
        for name, info in self._layers_info.items():
          cell = info["cell_module"]
          layer = info["layer_module"]
          graph = info["graph"]
          if layer.input is None:
            warnings.warn(
                f"[NNDCT_WARNING]: Provide inputs for '{name}' when do deploy checking",
                RuntimeWarning)
            continue
          register_output_hook(cell, record_once=False)
          set_outputs_recorder_status(cell, True)
          layer(layer.input, layer.initial_state, layer.batch_lengths)

          for timestep in range(layer.input.size()[1]):
            enable_dump_weight = True if timestep == 0 else False
            update_nndct_blob_data(cell, graph, timestep)
            checker.update_dump_folder(f"{graph.name}/frame_{timestep}")
            checker.dump_nodes_output(
                graph,
                quantizer.quant_config,
                round_method=quantizer.quant_opt['round_method'],
                enable_dump_weight=enable_dump_weight,
                select_batch=True)
          
          set_outputs_recorder_status(cell, False)

        print("[NNDCT_NOTE]: Finsh dumping data.")



class RNNQuantProcessor(TorchQuantProcessor):
    
  def _check_args(self, module):
    if isinstance(module, list):
      for mod in module:
        self._check_args(mod)
    else:
      if not isinstance(module, torch.nn.Module):
        raise TypeError(f"{module.__class__.__name__} is not subclass of 'torch.nn.Module'.")
    
  def __init__(self,
               quant_mode: str,
               module: Union[torch.nn.Module, List[torch.nn.Module]],
               input_args: Union[torch.Tensor, Sequence[Any]] = None,
               state_dict_file: Optional[str] = None,
               output_dir: str = "quantize_result",
               bitwidth_w: int = 8,
               bitwidth_a: int = 8,
               device: torch.device = torch.device("cuda"),
               lstm_app: bool = True,
               quant_config_file: Optional[str] = None):
    self._export_folder = output_dir
    # Check arguments type
    self._check_args(module)
    
    # Check device available
    if device.type == "cuda":
      #if not (torch.cuda.is_available() and "CUDA_HOME" in os.environ):
      if not (torch.cuda.is_available() and "CUDA_HOME" or "ROCM_HOME" in os.environ):
        device = torch.device("cpu")
        NndctScreenLogger().warning2user(QWarning.CUDA_UNAVAILABLE, f"CUDA is not available, change device to CPU")
    
    # Transform torch module to quantized module format
    nndct_utils.create_work_dir(output_dir)
    
    # turn off weights equalization and bias correction
    if (hasattr(NndctOption.nndct_param_corr, '_value')):
      option_util.set_option_value("nndct_param_corr", NndctOption.nndct_param_corr._value)
    else:
      option_util.set_option_value("nndct_param_corr", False)
      
    if (hasattr(NndctOption.nndct_equalization, '_value')):
      option_util.set_option_value("nndct_equalization", NndctOption.nndct_equalization._value)
    else:
      option_util.set_option_value("nndct_equalization", False)
    
    option_util.set_option_value("nndct_cv_app", False)
    
    # Parse the quant config file
    QConfiger = RNNTorchQConfig()
    #if quant_config_file:
    QConfiger.parse_config_file(quant_config_file,
                                bit_width_w = bitwidth_w, 
                                bit_width_a = bitwidth_a)

    qconfig = QConfiger.qconfig
    quantizer, qmode = self._init_quant_env(quant_mode, 
                                            output_dir,
                                            qconfig,
                                            is_lstm=True)

    GLOBAL_MAP.set_map(NNDCT_KEYS.QUANTIZER, quantizer)
    GLOBAL_MAP.set_map(NNDCT_KEYS.QUANT_MODE, qmode)
    GLOBAL_MAP.set_map(NNDCT_KEYS.QUANT_DEVICE, device)
    GLOBAL_MAP.set_map(NNDCT_KEYS.QUANT_CONFIG, qconfig)
   
    if isinstance(module, list):
      quantize_models = []
      multi_graph = []
      self._input_tensors_name = []
      self._return_tensors_name = []
      for submod, example_input in zip(module, input_args):
        if has_lstm(submod):
          submod = submod.to(device)
          target_module = convert_lstm(submod, device)
          _, example_input = to_device(None, example_input, device)
          script_module = torch.jit.trace(target_module.eval(), example_input)
        else:
          submod, example_input = to_device(submod, example_input, device)
          script_module = torch.jit.trace(submod.eval(), example_input)
        quant_module, graph = prepare_quantizable_module(
            module=script_module,
            input_args=example_input,
            export_folder=output_dir,
            state_dict_file=state_dict_file,
            quant_mode=qmode,
            device=device)
        quant_module.from_script(True)
        multi_graph.append(graph)
        quantize_models.append(quant_module.to(device))
        if GLOBAL_MAP.get_ele(NNDCT_KEYS.TORCH_SCRIPT_MODEL):
          quantizer.add_script(GLOBAL_MAP.get_ele(NNDCT_KEYS.TORCH_SCRIPT_MODEL))
        
        if qmode > 1:
          register_output_hook(quant_module, record_once=True)
          set_outputs_recorder_status(quant_module, True)
        if isinstance(example_input, torch.Tensor):
          quant_module._input_tensors_name = graph.get_input_tensors([example_input])
          self._input_tensors_name.append(quant_module._input_tensors_name)
        else:
          quant_module._input_tensors_name = graph.get_input_tensors(example_input)
          self._input_tensors_name.append(quant_module._input_tensors_name)
        #quant_module._graph = graph
        self._return_tensors_name.append(graph.get_return_tensors())
        
      graph = merge_multi_graphs_to_single_graph(multi_graph)   
      quantizer.quant_model = quantize_models
    else:
      if has_lstm(module):
        module = module.to(device)
        target_module = convert_lstm(module, device)
      else:
        target_module = module.to(device)
      _, example_input = to_device(None, input_args, device)
      script_module = torch.jit.trace(target_module.eval(), example_input)
      quant_module, graph = prepare_quantizable_module(
          module=script_module,
          input_args=input_args,
          export_folder=output_dir,
          state_dict_file=state_dict_file,
          quant_mode=qmode,
          device=device)
      quant_module.from_script(True)
      quantizer.quant_model = quant_module.to(device)
      if GLOBAL_MAP.get_ele(NNDCT_KEYS.TORCH_SCRIPT_MODEL):
        quantizer.add_script(GLOBAL_MAP.get_ele(NNDCT_KEYS.TORCH_SCRIPT_MODEL))
      if qmode > 1:
        register_output_hook(quant_module, record_once=True)
        set_outputs_recorder_status(quant_module, True)
      if isinstance(input_args, torch.Tensor):
        quant_module._input_tensors_name = graph.get_input_tensors([input_args])
        self._input_tensors_name = quant_module._input_tensors_name
      else:
        quant_module._input_tensors_name = graph.get_input_tensors(input_args)
        self._input_tensors_name = quant_module._input_tensors_name
      self._return_tensors_name = graph.get_return_tensors()

    quantizer.setup(graph, rnn_front_end=True, lstm=True)
    self.quantizer = quantizer
    self._example_inputs = input_args

    if NndctOption.nndct_calib_before_finetune.value is True:
      self.quantizer.export_float_param()
