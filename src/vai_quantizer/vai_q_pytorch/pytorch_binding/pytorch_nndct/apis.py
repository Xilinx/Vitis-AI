

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
from typing import Any, Optional, Sequence, Union, List, Dict, Tuple
import torch
from nndct_shared.utils import NndctScreenLogger, NndctOption, QError, QWarning, QNote
from .qproc import TorchQuantProcessor
from .qproc import base as qp
from .qproc import LSTMTorchQuantProcessor, RNNQuantProcessor
from .qproc import vaiq_system_info
from .quantization import QatProcessor

# API class
class torch_quantizer():
  def __init__(self,
               quant_mode: str, # ['calib', 'test']
               module: Union[torch.nn.Module, List[torch.nn.Module]],
               input_args: Union[torch.Tensor, Sequence[Any]] = None,
               state_dict_file: Optional[str] = None,
               output_dir: str = "quantize_result",
               bitwidth: int = None,
               mix_bit: bool = False,
               device: torch.device = torch.device("cuda"),
               lstm: bool = False,
               app_deploy: str = "CV",
               qat_proc: bool = False,
               custom_quant_ops: List[str] = None,
               quant_config_file: Optional[str] = None,
               target: Optional[str] = None):

    vaiq_system_info(device)

    if NndctOption.nndct_target.value:
      target = NndctOption.nndct_target.value
    
    if NndctOption.nndct_inspect_test.value and target:
      from pytorch_nndct.apis import Inspector
      inspector = Inspector(target)
      inspector.inspect(module, input_args, device, output_dir)


    if bitwidth is None and quant_config_file is None:
      bitwidth = 8
    
    if app_deploy == "CV": lstm_app = False
    elif app_deploy == "NLP": lstm_app = True
    self._qat_proc = False
    if qat_proc:
      if bitwidth is None:
        bitwidth = 8
      self.processor = QatProcessor(model = module,
                                    inputs = input_args,
                                    bitwidth = bitwidth,
                                    mix_bit = mix_bit,
                                    device = device)
      self._qat_proc = True
    elif lstm:
      self.processor = RNNQuantProcessor(quant_mode = quant_mode,
                                               module = module,
                                               input_args = input_args,
                                               state_dict_file = state_dict_file,
                                               output_dir = output_dir,
                                               bitwidth_w = bitwidth,
                                               # lstm IP only support 16 bit activation
                                               bitwidth_a = 16,
                                               device = device,
                                               lstm_app = lstm_app,
                                               quant_config_file = quant_config_file)
                                      
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
                                           custom_quant_ops = custom_quant_ops,
                                           quant_config_file = quant_config_file,
                                           target = target)
  # Finetune parameters,
  # After finetuning, run original forwarding code for calibration
  # After calibration, run original forwarding code to test quantized model accuracy
  def fast_finetune(self, run_fn, run_args):
    self.processor.finetune(run_fn, run_args)

  # load finetuned parameters
  def load_ft_param(self):
    #self.processor.advanced_quant_setup()
    self.processor.quantizer.load_param()

  # calibration can be called in the same process with test and deploy
  def quantize(self, run_fn, run_args, ft_run_args=None):
    self.processor.quantize(run_fn, run_args, ft_run_args)

  # test can be called in the same process with calibration and deploy
  def test(self, run_fn, run_args):
    self.processor.test(run_fn, run_args)

  # deploy can be called in the same process with calibration and test
  def deploy(self, run_fn, run_args, fmt='xmodel'):
    self.processor.deploy(run_fn, run_args, fmt)

  # export quantization steps information for tensors to be quantized
  def export_quant_config(self):
    self.processor.export_quant_config()

  # export xmodel for compilation
  def export_xmodel(self, output_dir="quantize_result", deploy_check=False, dynamic_batch=False):
    self.processor.export_xmodel(output_dir, deploy_check, dynamic_batch)

  def export_onnx_model(self, output_dir="quantize_result", verbose=False, dynamic_batch=False, opset_version=None):
    self.processor.export_onnx_model(output_dir, verbose, dynamic_batch, opset_version)
     
  def export_traced_torch_script(self, output_dir="quantize_result", verbose=False):
    NndctScreenLogger().warning(
    '"export_traced_torch_script" is deprecated and will be removed in the future. '
    'Use "export_torch_script" instead.')
    self.processor.export_traced_torch_script(output_dir, verbose)
  
  def export_torch_script(self, output_dir="quantize_result", verbose=False):
    return self.processor.export_torch_script(output_dir, verbose)
     
  @property
  def quant_model(self):
    NndctScreenLogger().info(f"=>Get module with quantization.")
    return self.processor.quant_model()

  @property
  def deploy_model(self):
    if not self._qat_proc:
      NndctScreenLogger().warning2user(QWarning.DEPLOY_MODEL, f"Only quant aware training process has deployable model.")
      return
    NndctScreenLogger().info(f"=>Get deployable module.")
    return self.processor.deploy_model()

# for vitis-ai 1.2 backward compatible
def dump_xmodel(output_dir="quantize_result", deploy_check=False, app_deploy="CV"):
  if app_deploy == "CV": lstm_app = False
  elif app_deploy == "NLP": lstm_app = True
  NndctScreenLogger().warning(f"The function dump_xmodel() will retire in future version. Use torch_quantizer.export_xmodel() reversely.")
  qp.dump_xmodel(output_dir, deploy_check, lstm_app)


class Inspector(object):
  def __init__(self, name_or_fingerprint: str):
    """The inspector is design to diagnoise neural network(NN) model under different architecure of DPU. 
        It's very useful to find which type of device will be assigned to the operator in NN model.
        It can provide hardware constraints messages for user to optimize NN model for deployment.
    
    """
    
    if NndctOption.nndct_use_old_inspector.value is True:
      from pytorch_nndct.hardware import InspectorImpl
    else:
      from pytorch_nndct.hardware_v3 import InspectorImpl
      NndctScreenLogger().info("Inspector is on.")

    in_type = "name"
    if name_or_fingerprint.startswith("0x"):
      in_type = "fingerprint"
      
    self._inspector_impl = None
    if in_type == "name":
      self._inspector_impl = InspectorImpl.create_by_DPU_arch_name(name_or_fingerprint)
    else:
      self._inspector_impl = InspectorImpl.create_by_DPU_fingerprint(name_or_fingerprint)

  def inspect(self, module: torch.nn.Module, 
              input_args: Union[torch.Tensor, Tuple[Any]], 
              device: torch.device = torch.device("cuda"), 
              output_dir: str = "quantize_result", 
              verbose_level: int = 1, 
              image_format: Optional[str] = None):
    NndctScreenLogger().info(f"=>Start to inspect model...")
    self._inspector_impl.inspect(module, input_args, device, output_dir, verbose_level)
    if image_format is not None:
      available_format = ["svg", "png"]
      NndctScreenLogger().check2user(QError.INSPECTOR_OUTPUT_FORMAT, f"Only support dump svg or png format.", image_format in available_format)
      self._inspector_impl.export_dot_image_v2(output_dir, image_format)
    NndctScreenLogger().info(f"=>Finish inspecting.")
