

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
import warnings
from collections import defaultdict
from typing import Optional

import torch

import nndct_shared.utils as nndct_utils
import pytorch_nndct.nn.modules.rnn_builder as rnn_builder
from nndct_shared.base import GLOBAL_MAP, NNDCT_KEYS, NNDCT_OP
from nndct_shared.utils import create_work_dir
from nndct_shared.compile import CompilerFactory, DeployChecker
from nndct_shared.nndct_graph import (merge_multi_subgraphs,
                                      reorder_multi_subgraph_nodes)
from pytorch_nndct.parse import NodeTransformer
from pytorch_nndct.quantization import TORCHQuantizer
from pytorch_nndct.utils import TorchSymbol

from .utils import (connect_module_with_graph, connect_module_with_quantizer,
                    parse_module, recreate_nndct_module,
                    set_outputs_recorder_status, update_nndct_blob_data)


class LSTMQuantizer(object):
  r"""
  `class used to quantize LSTM modules.`
  Args:
      export_folder (str, optional): work directory. Defaults to 'QLSTM'.
      quant_mode (Optional[int], optional): turn on post training quantization flow 1: calibration, 2:quantization.
      Defaults to None.
  
  """

  def __init__(self,
               output_dir: str = 'QLSTM',
               quant_mode: Optional[int] = None,
               bitwidth_w: Optional[int] = 8,
               bitwidth_a: Optional[int] = 8) -> None:
    self._export_folder = output_dir
    self._quant_mode = quant_mode
    self._bit_w = bitwidth_w
    self._bit_a = bitwidth_a

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

  def quantize_modules(self, top_module: torch.nn.Module) -> torch.nn.Module:
    """
    `prepare quantizable LSTM sub modules.`
    
    Args:
        top_module (torch.nn.Module): Top Module in which LSTM need to do quantization
    
    Raises:
        RuntimeError: The top module should have one LSTM at least.
    
    Returns:
        torch.nn.Module: Top Module in which LSTM sub modules are transformed to quantizible module
    """

    standard_RNNs, customized_RNNs = self._analyse_module(top_module)

    if len(standard_RNNs) == 0 and len(customized_RNNs) == 0:
      raise RuntimeError(
          f"The top module '{top_module._get_name()}' should have one LSTM module at least."
      )

    nndct_utils.create_work_dir(self._export_folder)

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
    for name, module in standard_RNNs.items():
      layers_graph = self._get_standard_RNN_graph(
          graph_name=name, lstm_module=module)
      self._modules_info[name]["layers_graph"] = layers_graph
      self._modules_info[name]["input_size"] = [module.input_size
                                                ] * module.num_layers
      self._modules_info[name]["hidden_size"] = [module.hidden_size
                                                 ] * module.num_layers
      self._modules_info[name]["memory_size"] = [module.hidden_size
                                                 ] * module.num_layers
      self._modules_info[name][
          "stack_mode"] = "bidirectional" if module.bidirectional else "unidirectional"
      self._modules_info[name][
          "batch_first"] = True if module.batch_first is True else False

      if module.mode == 'LSTM':
        self._modules_info[name]["mode"] = "LSTM"
      elif module.mode == "GRU": 
        self._modules_info[name]["mode"] = "GRU"
    # merge multi graphs into a graph
    top_graph = self._merge_subgraphs()
    
    # turn on quantizer
    if self._quant_mode:
      quantizer = TORCHQuantizer(self._quant_mode, self._export_folder,
                                 self._bit_w, self._bit_a)
      GLOBAL_MAP.set_map(NNDCT_KEYS.QUANTIZER, quantizer)
      GLOBAL_MAP.set_map(NNDCT_KEYS.QUANT_MODE, self._quant_mode)
      quantizer.setup(top_graph, lstm=True)
    
    # write and reload quantizable cell module
    module_graph_map = self._rebuild_layer_module()
    
    # hook quantizer and module
    if self._quant_mode is not None:
      self._hook_quant_module_with_quantizer(quantizer)
    
    # replace float module with quantizale module
    for name, info in self._modules_info.items():
      if info["stack_mode"] is not None:
        self._build_stack_lstm_module(info)
      else:
        info["QLSTM"] = list(info["layers_module"][0].values())[0]
      top_module = self._insert_QuantLstm_in_top_module(top_module, name, info)

    # move modules info into layers info
    self._convert_modules_info_to_layers(module_graph_map)

    return top_module

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

  def _hook_quant_module_with_quantizer(self, quantizer):
     for _, info in self._modules_info.items():
      for layer in info["layers_module"]:
        for direction, quant_module in layer.items():
          connect_module_with_quantizer(quant_module, quantizer)
         
  def _rebuild_layer_module(self):
    module_graph_map = {}
    for name, info in self._modules_info.items():
      layers_module = []
      for l_num, layer_graph in enumerate(info["layers_graph"]):
        lstm_cell_pair = {}
        for lstm_direction, graph in layer_graph.items():
          export_file = os.path.join(
              self._export_folder, f"{graph.name}{TorchSymbol.SCRIPT_SUFFIX}")
          quant_module = recreate_nndct_module(graph, export_file)
          connect_module_with_graph(quant_module, graph, record_once=False)
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

  def export_quant_config(self):
    """
    `export bitwidth and fixpoint info of blobs and parameters under work dir`
    """
    quantizer = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANTIZER)
    if quantizer and quantizer.quant_mode == 1:
      quantizer.export_quant_config()

  def dump_xmodel(self, deploy_check=False):
    """
    `dump xmodel for LSTM cell`
    """
    quantizer = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANTIZER)
    if quantizer and quantizer.quant_mode > 1:
      compiler = CompilerFactory.get_compiler("xmodel")
      xmodel_dir = os.path.join(self._export_folder, "xmodel")
      create_work_dir(xmodel_dir)
      for info in self._modules_info.values():
        for l_num, layer_graph in enumerate(info["layers_graph"]):
          for lstm_direction, graph in layer_graph.items():
            try:
              compiler.do_compile(
                  nndct_graph=graph,
                  quant_config_info=quantizer.quant_config,
                  output_file_name=os.path.join(xmodel_dir, graph.name),
                  graph_attr_kwargs={"direction": lstm_direction})
            except Exception as e:
              print(
                  f"[NNDCT_ERROR]:failed convert nndct graph to xmodel({str(e)})."
              )

            else:
              print("[NNDCT_NOTE]:Successfully convert nndct graph to xmodel!")

      if deploy_check:
        print("[NNDCT_NOTE]: Dumping checking data...")
        checker = DeployChecker(
            output_dir_name=self._export_folder, data_format="txt")     
        
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
                enable_dump_weight=enable_dump_weight)
          
          set_outputs_recorder_status(cell, False)

        print("[NNDCT_NOTE]: Finsh dumping data.")
