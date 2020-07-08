

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
from typing import Any, Optional, Sequence, Tuple, Union, NoReturn

import torch

import nndct_shared.utils as nndct_utils
from nndct_shared.base import FrameworkType
from nndct_shared.nndct_graph import Graph
from nndct_shared.optimization import NndctOptimizer
from pytorch_nndct.export import TorchScriptWriter
from pytorch_nndct.parse import TorchParser
from pytorch_nndct.utils import TorchSymbol
from pytorch_nndct.quantization import TORCHQuantizer
from nndct_shared.utils import NndctScreenLogger
from .ModuleHooker import ModuleHooker


def _reload_quantizable_module(module_file_name: str,
                               module_name: str) -> torch.nn.Module:

  py_module_name = "_".join(["nndct", module_name])
  spec = importlib.util.spec_from_file_location(py_module_name,
                                                module_file_name)
  py_module = importlib.util.module_from_spec(spec)
  sys.modules[py_module_name] = py_module
  spec.loader.exec_module(py_module)
  return py_module.__dict__[module_name]()


def recreate_nndct_module(graph: Graph, export_file: str) -> torch.nn.Module:

  exporter = TorchScriptWriter()
  exporter.write(graph, file_path=export_file)
  nndct_quant_module = _reload_quantizable_module(export_file, graph.name)
  return nndct_quant_module


def parse_module(module: torch.nn.Module,
                 input_args: Union[torch.Tensor, Sequence[Any]],
                 enable_opt: bool = True,
                 graph_name: Optional[str] = None) -> Graph:

  parser = TorchParser()
  graph = parser(module._get_name() if graph_name is None else graph_name,
                 module, input_args)
  if enable_opt:
    optimizer = NndctOptimizer(use_quant=True, model_type=FrameworkType.TORCH)
    graph = optimizer.optimize(graph, commands=['FuseBnToConv'])
  return graph


def connect_module_with_graph(module: torch.nn.Module,
                              graph: Graph,
                              record_once: bool = True,
                              recover_param: bool = True) -> NoReturn:

  ModuleHooker.register_output_hook(module, record_once)

  ModuleHooker.hook_module_with_node(module, graph)

  if recover_param:
    ModuleHooker.update_parameters(module, graph, graph2module=True)


def connect_module_with_quantizer(module: torch.nn.Module,
                                  quantizer: TORCHQuantizer) -> NoReturn:
  ModuleHooker.hook_module_with_quantizer(module, quantizer)


def update_nndct_parameters(module: torch.nn.Module, graph: Graph) -> NoReturn:
  ModuleHooker.update_parameters(module, graph, graph2module=False)


def update_nndct_blob_data(module: torch.nn.Module,
                           graph: Graph,
                           time_step: Optional[int] = None) -> NoReturn:
  ModuleHooker.update_blobs_once(module, graph, time_step)


def set_outputs_recorder_status(module, turn_on) -> NoReturn:
  ModuleHooker.clear_record_outputs(module)
  ModuleHooker.turn_on_record_outputs(
      module) if turn_on else ModuleHooker.turn_off_record_outputs(module)


def prepare_quantizable_module(
    module: torch.nn.Module,
    input_args: Union[torch.Tensor, Sequence[Any]],
    export_folder: str,
    state_dict_file: Optional[str] = None,
    quant_mode: int = 1) -> Tuple[torch.nn.Module, Graph]:

  nndct_utils.create_work_dir(export_folder)

  if isinstance(state_dict_file, str):
    state_dict = torch.load(state_dict_file)
    module.load_state_dict(state_dict)

  export_file = os.path.join(export_folder,
                             module._get_name() + TorchSymbol.SCRIPT_SUFFIX)
  
  # parse origin module to graph
  NndctScreenLogger().info(f"=>Parsing {module._get_name()}...")
  graph = parse_module(module, input_args)
  NndctScreenLogger().info(f"=>Quantizable module is generated.({export_file})")
  # recreate quantizable module from graph
  quant_module = recreate_nndct_module(graph, export_file)

  # hook module with graph
  connect_module_with_graph(quant_module, graph)

  return quant_module, graph
