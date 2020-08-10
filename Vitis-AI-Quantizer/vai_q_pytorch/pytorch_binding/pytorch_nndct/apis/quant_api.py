

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
from typing import Any, Optional, Sequence, Union

import torch

import nndct_shared.utils as nndct_utils
from nndct_shared.base import GLOBAL_MAP, NNDCT_KEYS
from nndct_shared.compile import CompilerFactory, DeployChecker
from nndct_shared.utils import AddXopError, NndctScreenLogger, NndctOption
from pytorch_nndct.quantization import TORCHQuantizer

from .utils import (connect_module_with_quantizer, prepare_quantizable_module,
                    set_outputs_recorder_status, update_nndct_blob_data,
                    update_nndct_parameters)


def torch_quantizer(quant_mode: int,
                    module: torch.nn.Module,
                    input_args: Union[torch.Tensor, Sequence[Any]],
                    state_dict_file: Optional[str] = None,
                    output_dir: str = "quantize_result",
                    bitwidth_w: int = 8,
                    bitwidth_a: int = 8) -> TORCHQuantizer:

  def _check_args():
    nonlocal module
    if not isinstance(module, torch.nn.Module):
      raise TypeError(f"type of 'module' should be 'torch.nn.Module'.")

    if not isinstance(input_args, (tuple, list, torch.Tensor)):
      raise TypeError(f"type of input_args should be tuple/list/torch.Tensor.")
    
    device = None
    if isinstance(input_args, torch.Tensor):
      device = input_args.device
    else:
      for inp in input_args:
        if isinstance(inp, torch.Tensor):
          device = inp.device
          break
        
    if device:
      module = module.to(device)

  def _init_quant_env():
    nonlocal quant_mode 
    if NndctOption.nndct_quant_mode.value > 0:
      quant_mode = NndctOption.nndct_quant_mode.value
    
    if quant_mode == 1:
      NndctScreenLogger().info(f"Quantization calibration process start up...")
    elif quant_mode == 2:
      NndctScreenLogger().info(f"Quantization test process start up...")
      
    quantizer = TORCHQuantizer(quant_mode, output_dir, bitwidth_w, bitwidth_a)
    GLOBAL_MAP.set_map(NNDCT_KEYS.QUANTIZER, quantizer)
    GLOBAL_MAP.set_map(NNDCT_KEYS.QUANT_MODE, quant_mode)
    return quantizer, quant_mode
  
  # Check arguments type
  _check_args()
  
  # Transform torch module to quantized module format
  nndct_utils.create_work_dir(output_dir)
  
  # Create a quantizer object, which can control all quantization flow,
  quantizer, quant_mode = _init_quant_env()
  
  quant_module, graph = prepare_quantizable_module(
      module=module,
      input_args=input_args,
      export_folder=output_dir,
      state_dict_file=state_dict_file,
      quant_mode=quant_mode)
  
  # enable record outputs of per layer
  if quant_mode > 1:
    set_outputs_recorder_status(quant_module, True)
      
  # intialize quantizer 
  quantizer.setup(graph)

  # hook module with quantizer
  connect_module_with_quantizer(quant_module, quantizer)
  
  quantizer.quant_model = quant_module

  return quantizer


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
    try:
      compiler = CompilerFactory.get_compiler("xmodel")
      NndctScreenLogger().info("=>Converting to xmodel ...")
      compiler.do_compile(
          nndct_graph=quantizer.Nndctgraph,
          quant_config_info=quantizer.quant_config,
          output_file_name=os.path.join(output_dir, quantizer.Nndctgraph.name))

    except AddXopError as e:
      NndctScreenLogger().error(f"Failed convert nndct graph to xmodel({str(e)}).")
    else:
      NndctScreenLogger().info(f"=>Successfully convert to xmodel.({compiler.xmodel_file})")
      
    # dump data for accuracy checkvim 
    if deploy_check:
      NndctScreenLogger().info("=>Dumping checking data...")
      update_nndct_blob_data(quantizer.quant_model, quantizer.Nndctgraph)
      checker = DeployChecker(output_dir_name=output_dir)
      checker.dump_nodes_output(
          quantizer.Nndctgraph,
          quantizer.quant_config,
          round_method=quantizer.quant_opt['round_method'])
      
      set_outputs_recorder_status(quantizer.quant_model, False)
      NndctScreenLogger().info(f"=>Finsh dumping data.({checker.dump_folder})")
