

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
from typing import Any, Optional, Sequence, Union

import torch

import nndct_shared.utils as nndct_utils
from nndct_shared.base import GLOBAL_MAP, NNDCT_KEYS
from nndct_shared.compile import CompilerFactory, DeployChecker
from nndct_shared.utils import AddXopError, NndctOption, NndctScreenLogger
from nndct_shared.quantization import DefaultQstrategy
from pytorch_nndct.quantization import TORCHQuantizer
from .adaquant import AdvancedQuantProcessor

from .utils import (connect_module_with_graph,
                    disconnect_modeule_with_graph, prepare_quantizable_module,
                    register_output_hook, set_outputs_recorder_status,
                    to_device, update_nndct_blob_data, update_nndct_parameters,
                    get_deploy_graph_list)

class TorchQuantProcessor():

  def _check_args(self, module, input_args):
    if not isinstance(module, torch.nn.Module):
      raise TypeError(f"type of 'module' should be 'torch.nn.Module'.")

    if not isinstance(input_args, (tuple, list, torch.Tensor)):
      raise TypeError(f"type of input_args should be tuple/list/torch.Tensor.")
    
  def _init_quant_env(self, quant_mode, output_dir, quant_strategy):
    if isinstance(quant_mode, int):
      NndctScreenLogger().warning(f"quant_mode will not support integer value in future version. It supports string values 'calib' and 'test'.")
      qmode = quant_mode
    elif isinstance(quant_mode, str):
      if quant_mode == 'calib':
        qmode = 1
      elif quant_mode == 'test':
        qmode = 2
      else:
        NndctScreenLogger().error(f"quant_mode supported values are 'calib' and 'test'. Change it to 'calib' as calibration mode")
        qmode = 1
    else:
      NndctScreenLogger().error(f"quant_mode supported values are string 'calib' and 'test'. Change it to 'calib' as calibration mode")
      qmode = 1

    if NndctOption.nndct_quant_mode.value > 0:
      qmode = NndctOption.nndct_quant_mode.value
    
    if qmode == 1:
      NndctScreenLogger().info(f"Quantization calibration process start up...")
    elif qmode == 2:
      NndctScreenLogger().info(f"Quantization test process start up...")
      
    quantizer = TORCHQuantizer.create_from_strategy(qmode, 
                                                    output_dir, 
                                                    quant_strategy)
    return quantizer, qmode
  
  def __init__(self,
               quant_mode: str,
               module: torch.nn.Module,
               input_args: Union[torch.Tensor, Sequence[Any]] = None,
               state_dict_file: Optional[str] = None,
               output_dir: str = "quantize_result",
               bitwidth_w: int = 8,
               bitwidth_a: int = 8,
               mix_bit: bool = False,
               device: torch.device = torch.device("cuda"),
               lstm_app: bool = False):
    # Check arguments type
    self._check_args(module, input_args)
    
    # Check device available
    if device.type == "cuda":
      if not (torch.cuda.is_available() and "CUDA_HOME" in os.environ):
        device = torch.device("cpu")
        NndctScreenLogger().warning(f"CUDA is not available, change device to CPU")
    
    # Transform torch module to quantized module format
    nndct_utils.create_work_dir(output_dir)
    
    # Create a quantizer object, which can control all quantization flow,
    quant_strategy = DefaultQstrategy(bits_weight=bitwidth_w,
                                      bits_bias=bitwidth_a,
                                      bits_activation=bitwidth_a,
                                      mix_bit=mix_bit)
    quantizer, qmode = self._init_quant_env(quant_mode, 
                                            output_dir,
                                            quant_strategy)
    GLOBAL_MAP.set_map(NNDCT_KEYS.QUANTIZER, quantizer)
    GLOBAL_MAP.set_map(NNDCT_KEYS.QUANT_MODE, qmode)
    GLOBAL_MAP.set_map(NNDCT_KEYS.QUANT_DEVICE, device)
    
    # Prepare quantizable module
    quant_module, graph = prepare_quantizable_module(
        module=module,
        input_args=input_args,
        export_folder=output_dir,
        state_dict_file=state_dict_file,
        quant_mode=qmode,
        device=device)
    
    # enable record outputs of per layer
    if qmode > 1:
      register_output_hook(quant_module, record_once=True)
      set_outputs_recorder_status(quant_module, True)
        
    # intialize quantizer 
    quantizer.setup(graph, False, lstm_app)

    # hook module with quantizer
    # connect_module_with_quantizer(quant_module, quantizer)
    quantizer.quant_model = quant_module

    self.quantizer = quantizer
    self.adaquant = None

  def advanced_quant_setup(self):
    if (self.adaquant is None):
      self.adaquant = AdvancedQuantProcessor(self.quantizer)
  
  def quant_model(self):
    return self.quantizer.quant_model
  
  # function needs forwarding iteration control
  def finetune(self, run_fn, run_args):
    self.advanced_quant_setup()
    if self.adaquant is not None:
      self.adaquant.finetune(run_fn, run_args)
  
  # function needs forwarding iteration control
  def quantize(self, run_fn, run_args):
    self.advanced_quant_setup()
    if self.adaquant is not None:
      self.adaquant.quantize(run_fn, run_args)

  # quantization steps of quantized tensors
  def export_quant_config(self):
    self.quantizer.export_quant_config()

  # export xmodel file to be compiled for deployment
  def export_xmodel(self, output_dir="quantize_result", deploy_check=False):
    dump_xmodel(output_dir, deploy_check)
    

def dump_xmodel(output_dir="quantize_result", deploy_check=False):
  r"""converts module to xmodel for deployment
  compilation only works when quantm model = 2.
  The xmodel and some checking data will be generated under work dir.

  Args:
    deploy_check(bool): if true, can dump blobs and parameters of model for deployment verification

  Returns:
    None
  """
  quantizer = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANTIZER)
  if quantizer and quantizer.quant_mode > 1:
    nndct_utils.create_work_dir(output_dir)
    
    # compile to xmodel
    
    compiler = CompilerFactory.get_compiler("xmodel")
      
    NndctScreenLogger().info("=>Converting to xmodel ...")
    deploy_graphs = get_deploy_graph_list(quantizer.quant_model, quantizer.Nndctgraph)
    depoly_infos = compiler.get_deloy_graph_infos(quantizer, deploy_graphs)
      
    for depoly_info in depoly_infos:
      try:
        compiler.do_compile(
            depoly_info.dev_graph,
            quant_config_info=depoly_info.quant_info,
            output_file_name=os.path.join(output_dir, depoly_info.dev_graph.name))

      except AddXopError as e:
        NndctScreenLogger().error(f"Failed convert graph '{depoly_info.dev_graph.name}' to xmodel({str(e)}).")
       
      # dump data for accuracy check
      if deploy_check:
        NndctScreenLogger().info(f"=>Dumping '{depoly_info.dev_graph.name}'' checking data...")
        checker = DeployChecker(output_dir_name=output_dir)
        checker.update_dump_folder(f"{depoly_info.dev_graph.name}")
        checker.dump_nodes_output(
            depoly_info.dev_graph,
            depoly_info.quant_info,
            round_method=quantizer.quant_opt['round_method'], select_batch=False)
        
        NndctScreenLogger().info(f"=>Finsh dumping data.({checker.dump_folder})")
        
    set_outputs_recorder_status(quantizer.quant_model, False)
