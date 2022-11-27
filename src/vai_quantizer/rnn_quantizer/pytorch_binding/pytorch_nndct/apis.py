

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
from typing import Any, Optional, Sequence, Union, List, Dict
import torch
from nndct_shared.utils import NndctScreenLogger, NndctOption
from .qproc import TorchQuantProcessor
from .qproc import base as qp
from .qproc import LSTMTorchQuantProcessor, RNNQuantProcessor
from .quantization import QatProcessor

# API class
class torch_quantizer():
  def __init__(self,
               quant_mode: str, # ['calib', 'test']
               module: torch.nn.Module,
               input_args: Union[torch.Tensor, Sequence[Any]] = None,
               state_dict_file: Optional[str] = None,
               output_dir: str = "quantize_result",
               bitwidth: int = 8,
               mix_bit: bool = False,
               device: torch.device = torch.device("cuda"),
               lstm: bool = False,
               app_deploy: str = "CV",
               qat_proc: bool = False,
               custom_quant_ops: List[str] = None):
    
    if app_deploy == "CV": lstm_app = False
    elif app_deploy == "NLP": lstm_app = True
    self._qat_proc = False
    if qat_proc:
      self.processor = QatProcessor(model = module,
                                    inputs = input_args,
                                    bitwidth = bitwidth,
                                    mix_bit = mix_bit,
                                    device = device)
      self._qat_proc = True
    elif lstm:
      if NndctOption.nndct_jit_script_mode.value is True:
        self.processor = RNNQuantProcessor(quant_mode = quant_mode,
                                               module = module,
                                               input_args = input_args,
                                               state_dict_file = state_dict_file,
                                               output_dir = output_dir,
                                               bitwidth_w = bitwidth,
                                               # lstm IP only support 16 bit activation
                                               bitwidth_a = 16,
                                               device = device,
                                               lstm_app = lstm_app)
      else:
        self.processor = LSTMTorchQuantProcessor(quant_mode = quant_mode,
                                                module = module,
                                                input_args = input_args,
                                                state_dict_file = state_dict_file,
                                                output_dir = output_dir,
                                                bitwidth_w = bitwidth,
                                                # lstm IP only support 16 bit activation
                                                bitwidth_a = 16,
                                                device = device,
                                                lstm_app = lstm_app)
    else:
      self.processor = TorchQuantProcessor(quant_mode = quant_mode,
                                           module = module,
                                           input_args = input_args,
                                           state_dict_file = state_dict_file,
                                           output_dir = output_dir,
                                           bitwidth_w = bitwidth,
                                           bitwidth_a = bitwidth,
                                           device = device,
                                           lstm_app = lstm_app,
                                           custom_quant_ops = custom_quant_ops)
  # Finetune parameters,
  # After finetuning, run original forwarding code for calibration
  # After calibration, run original forwarding code to test quantized model accuracy
  def fast_finetune(self, run_fn, run_args):
    self.processor.finetune(run_fn, run_args)

  # load finetuned parameters
  def load_ft_param(self):
    self.processor.advanced_quant_setup()

  # full procedures mode including finetune, calibration and test accuracy
  def quantize(self, un_fn, run_args):
    self.processor.quantize(run_fn, run_args)

  # export quantization steps information for tensors to be quantized
  def export_quant_config(self):
    self.processor.export_quant_config()

  # export xmodel for compilation
  def export_xmodel(self, output_dir="quantize_result", deploy_check=False):
    self.processor.export_xmodel(output_dir, deploy_check)

  # deploy the trained model for quant aware training process
  def deploy(self, trained_model, output_dir, mix_bit=False):
    if not self._qat_proc:
      NndctScreenLogger().warning(f"Only quant aware training process has deploy function.")
      return
    self.processor.convert_to_deployable(trained_model, output_dir, mix_bit=mix_bit)

  @property
  def quant_model(self):
    NndctScreenLogger().info(f"=>Get module with quantization.")
    return self.processor.quant_model()

  @property
  def deploy_model(self):
    if not self._qat_proc:
      NndctScreenLogger().warning(f"Only quant aware training process has deployable model.")
      return
    NndctScreenLogger().info(f"=>Get deployable module.")
    return self.processor.deploy_model()

# for vitis-ai 1.2 backward compatible
def dump_xmodel(output_dir="quantize_result", deploy_check=False, app_deploy="CV"):
  if app_deploy == "CV": lstm_app = False
  elif app_deploy == "NLP": lstm_app = True
  NndctScreenLogger().warning(f"The function dump_xmodel() will retire in future version. Use torch_quantizer.export_xmodel() reversely.")
  qp.dump_xmodel(output_dir, deploy_check, lstm_app)
