
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
from typing import Any, List, Optional, Sequence, Union

import re
import torch

from pytorch_nndct.version import __version__, Ctorch_version
import nndct_shared.utils as nndct_utils
from nndct_shared.base import GLOBAL_MAP, NNDCT_KEYS, NNDCT_OP
from nndct_shared.compile import CompilerFactory, DeployChecker
from nndct_shared.utils import (AddXopError, NndctOption, NndctScreenLogger,
                                option_util, QError, QWarning)
from nndct_shared.nndct_graph import Graph
#from nndct_shared.quantization import DefaultQstrategy, QStrategyFactory
from pytorch_nndct.quantization import (FakeQuantizer, TorchQConfig,
                                        TORCHQuantizer)
from pytorch_nndct.utils.jit_utils import get_torch_version
from pytorch_nndct.utils.onnx_utils import get_opset_version
from pytorch_nndct.utils.module_util import visualize_tensors, to_device, get_module_name
from nndct_shared.compile import get_xmodel_and_dump_infos
from .adaquant import AdvancedQuantProcessor
from .utils import (connect_module_with_graph, disconnect_modeule_with_graph,
                    get_deploy_graph_list, insert_fix_neuron_in_script_model,
                    opt_script_model_for_quant, prepare_quantizable_module,
                    register_output_hook, set_outputs_recorder_status,
                    update_nndct_blob_data, quant_model_inferenced,
                    insert_mul_after_avgpool, register_input_checker,
                    remove_quant_dequant_stub)
from .ModuleHooker import ModuleHooker


class TorchQuantProcessor():

  def _check_args(self, module, input_args):
    if not isinstance(module, torch.nn.Module):
      raise TypeError(f"Type of 'module' should be 'torch.nn.Module'.")

    if not isinstance(input_args, (tuple, list, torch.Tensor)):
      raise TypeError(f"Type of input_args should be tuple/list/torch.Tensor.")
    
  def _init_quant_env(self, quant_mode, output_dir, quant_strategy_info, is_lstm=False):
    if isinstance(quant_mode, int):
      NndctScreenLogger().warning(f"quant_mode will not support integer value in future version. It supports string values 'calib' and 'test'.")
      qmode = quant_mode
    elif isinstance(quant_mode, str):
      if quant_mode == 'calib':
        qmode = 1
      elif quant_mode == 'test':
        qmode = 2
      else:
        NndctScreenLogger().warning(f"quant_mode supported values are 'calib' and 'test'. Change it to 'calib' as calibration mode.")
        qmode = 1
    else:
      NndctScreenLogger().warning(f"quant_mode supported values are string 'calib' and 'test'. Change it to 'calib' as calibration mode.")
      qmode = 1

    if NndctOption.nndct_quant_mode.value > 0:
      qmode = NndctOption.nndct_quant_mode.value
    
    if qmode == 1:
      NndctScreenLogger().info(f"Quantization calibration process start up...")
    elif qmode == 2:
      NndctScreenLogger().info(f"Quantization test process start up...")
    
    
    target_device = quant_strategy_info['target_device']
    if target_device == 'DPU':
      quantizer = TORCHQuantizer.create_from_strategy(qmode,
                                                      output_dir,
                                                      quant_strategy_info,
                                                      is_lstm=is_lstm)
    elif target_device in ['CPU', 'GPU']:
      quantizer = FakeQuantizer.create_from_strategy(qmode, 
                                                     output_dir, 
                                                     quant_strategy_info,
                                                     is_lstm=is_lstm)
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
               lstm_app: bool = False,
               custom_quant_ops: Optional[List[str]] = None,
               quant_config_file: Optional[str] = None,
               target: Optional[str] = None):
    # Check arguments type
    self._check_args(module, input_args)
    if isinstance(module, torch.nn.DataParallel):
        module = module.module
        NndctScreenLogger().warning2user(QWarning.DATA_PARALLEL, f"Data parallel is not supported. The wrapper 'torch.nn.DataParallel' has been removed in quantizer.")

    # Check device available
    if device.type == "cuda":
      if not (torch.cuda.is_available() and ("CUDA_HOME" or "ROCM_HOME" in os.environ)):
        device = torch.device("cpu")
        NndctScreenLogger().warning2user(QWarning.CUDA_UNAVAILABLE, f"CUDA (HIP) is not available, change device to CPU.")
    
    # Transform torch module to quantized module format
    nndct_utils.create_work_dir(output_dir)
   
    # Parse the quant config file
    QConfiger = TorchQConfig()
    #if quant_config_file:
    QConfiger.parse_config_file(quant_config_file, 
                                bit_width_w = bitwidth_w, 
                                bit_width_a = bitwidth_a, 
                                mix_bit = mix_bit)
    qconfig = QConfiger.qconfig
    #bitwidth_w = qconfig['weights']['bit_width']
    #bitwidth_b = qconfig['bias']['bit_width']
    #bitwidth_a = qconfig['activation']['bit_width']
    #mix_bit = qconfig['mix_bit'] 

    # Create a quantizer object, which can control all quantization flow,
    #qstrategy_factory = QstrategyFactory()
    #quant_strategy = qstrategy_factory.create_qstrategy(qconfig) 
    #quant_strategy = DefaultQstrategy(bits_weight=bitwidth_w,
    #                                  bits_bias=bitwidth_a,
    #                                  bits_activation=bitwidth_a,
    #                                  mix_bit=mix_bit)
    quantizer, qmode = self._init_quant_env(quant_mode, 
                                            output_dir,
                                            qconfig)

    GLOBAL_MAP.set_map(NNDCT_KEYS.QUANTIZER, quantizer)
    GLOBAL_MAP.set_map(NNDCT_KEYS.QUANT_MODE, qmode)
    GLOBAL_MAP.set_map(NNDCT_KEYS.QUANT_DEVICE, device)
    GLOBAL_MAP.set_map(NNDCT_KEYS.QUANT_CONFIG, qconfig)
    if lstm_app: option_util.set_option_value("nndct_cv_app", False)
    else: option_util.set_option_value("nndct_cv_app", True)
      
    # Prepare quantizable module
   
    quant_module, graph = prepare_quantizable_module(
        module=module,
        input_args=input_args,
        export_folder=output_dir,
        state_dict_file=state_dict_file,
        quant_mode=qmode,
        device=device,
        connect_qm_with_graph=False)
       
    if len(graph.all_blocks()) > 1:
      quant_module.from_script(True)
    else:
      quant_module.from_script(False)

    # enable record outputs of per layer
    if qmode > 1:
      register_output_hook(quant_module, record_once=True)
      set_outputs_recorder_status(quant_module, True)
        
    # intialize quantizer 
    if target is None or quant_config_file:
      if target is not None and quant_config_file is not None:
        NndctScreenLogger().warning2user(QWarning.HW_AWARE_QUANT, "The hardware-aware quantization is turned off by quant_config_file.")
      quantizer.setup(graph, False, lstm_app, custom_quant_ops=custom_quant_ops)
    else:
      tmp_module = copy.deepcopy(quant_module)
      connect_module_with_graph(tmp_module, graph)
      tmp_module, input_args = to_device(tmp_module, input_args, device)
      quant_off_stat = NndctOption.nndct_quant_off.value
      param_corr_stat = NndctOption.nndct_param_corr.value
      nndct_utils.set_option_value("nndct_quant_off", True)
      nndct_utils.set_option_value("nndct_param_corr", False)
      register_output_hook(tmp_module, record_once=True)
      set_outputs_recorder_status(tmp_module, True)
      tmp_module.eval()
      if isinstance(input_args, tuple):
        _ = tmp_module(*input_args)
      else:
        _ = tmp_module(input_args)

      nndct_utils.set_option_value("nndct_quant_off", quant_off_stat)
      nndct_utils.set_option_value("nndct_param_corr", param_corr_stat)
      _, dev_graph = get_deploy_graph_list(tmp_module, graph, need_partition=False)
      quantizer.setup_for_target(target, graph, dev_graph)
      dev_graph.clean_tensors_data()
      
    # connect module with graph
    connect_module_with_graph(quant_module, graph)
    # hook module with quantizer
    quantizer.quant_model = quant_module
    if GLOBAL_MAP.get_ele(NNDCT_KEYS.TORCH_SCRIPT_MODEL):
      quantizer.add_script(GLOBAL_MAP.get_ele(NNDCT_KEYS.TORCH_SCRIPT_MODEL))
    self._example_inputs = input_args
    if isinstance(input_args, torch.Tensor):
      quant_module._input_tensors_name = graph.get_input_tensors([input_args])
      self._input_tensors_name = quant_module._input_tensors_name
    else:
      quant_module._input_tensors_name = graph.get_input_tensors(input_args)
      self._input_tensors_name = quant_module._input_tensors_name
    #quant_module._graph = graph
    self._return_tensors_name = graph.get_return_tensors()
    
    self._lstm_app = lstm_app
    self.quantizer = quantizer
    #self.adaquant = None
    
    # dump blob dist 
    if NndctOption.nndct_visualize.value is True:
      visualize_tensors(quantizer.quant_model)
      
    if NndctOption.nndct_calib_before_finetune.value is True:
      self.quantizer.export_float_param()

  def advanced_quant_setup(self, module, graph, example_inputs):
    return AdvancedQuantProcessor(module, graph, self.quantizer, example_inputs)
    # if (self.adaquant is None):
    #   self.adaquant = AdvancedQuantProcessor(self.quantizer)
  
  def quant_model(self):
    return self.quantizer.quant_model
   
  
  # function needs forwarding iteration control
  def finetune(self, run_fn, run_args):
    grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    if self.quantizer.quant_mode == 1 and NndctOption.nndct_calib_before_finetune.value is True:
      self.quantizer.load_float_param()
    
    if isinstance(self.quantizer.quant_model, (list, tuple)) and \
      len(self.quantizer.quant_model)>0:
      multi_graph = self.quantizer.graph
      index = 0
      for module in self.quantizer.quant_model:
        module.requires_grad_(requires_grad=True)
        module_name = module._get_name()
        module_graph = Graph(graph_name=module_name)
        for node in multi_graph.nodes:
          if node.name == module_name:
            module_block = node.blocks[0]
            module_graph.set_top_block(module_block)
            for node in module_block.nodes:
              module_graph.add_node(node)
            break
        example_input = self._example_inputs[index]
        adaquant = self.advanced_quant_setup(module, module_graph, example_input)
        index = index+1
        if NndctOption.nndct_ft_mode.value == 0:
          adaquant.finetune(run_fn, run_args)
        else:
          adaquant.finetune_v2(run_fn, run_args)
        module.requires_grad_(requires_grad=grad_enabled)
    else:
      self.quantizer.quant_model.requires_grad_(requires_grad=True)
      adaquant = self.advanced_quant_setup(self.quantizer.quant_model, self.quantizer.graph, self._example_inputs)
      if NndctOption.nndct_ft_mode.value == 0:
        adaquant.finetune(run_fn, run_args)
      else:
        adaquant.finetune_v2(run_fn, run_args)
      self.quantizer.quant_model.requires_grad_(requires_grad=grad_enabled)
    
    NndctScreenLogger().info(f"=>Export fast finetuned parameters ...")
    # export finetuned parameters
    self.quantizer.export_param()
    
    torch.set_grad_enabled(grad_enabled)
    
    # self.advanced_quant_setup()
    # if self.adaquant is not None:
    #   if NndctOption.nndct_ft_mode.value == 0:
    #     self.adaquant.finetune(run_fn, run_args)
    #   else:
    #     self.adaquant.finetune_v2(run_fn, run_args)
    # torch.set_grad_enabled(grad_enabled)

  # full control quantization 
  def quantize(self, run_fn, run_args, ft_run_args):
    NndctScreenLogger().info(f'Model quantization calibration begin:')
    # calibration
    self.quantizer.quant_mode = 1
    if ft_run_args is not None:
      self.finetune(run_fn, ft_run_args)
      self.quantizer.fast_finetuned = True
    run_fn(*run_args)
    self.quantizer.export_quant_config()
    NndctScreenLogger().info(f'Model quantization calibration end.')

  def test(self, run_fn, run_args):
    NndctScreenLogger().info(f'Quantized model test begin:')
    # test and print log message
    self.quantizer.quant_mode = 2
    if self.quantizer.fast_finetuned:
      # self.advanced_quant_setup()
      self.quantizer.load_param()
      
    log_str = run_fn(*run_args)
    NndctScreenLogger().info(f'Quantized model evaluation returns metric:\n {log_str}')
    NndctScreenLogger().info(f'Quantized model end.')

  def deploy(self, run_fn, run_args, fmt='xmodel', dynamic_batch=False):
    NndctScreenLogger().info(f'Quantized model depoyment begin:')
    # export quantized model
    # how to handle batch size must be 1
    # check function input
    if fmt not in ['xmodel', 'onnx', 'torch_script']:
      NndctScreenLogger().error(f"Parameter deploy only can be set 'xmodel', 'onnx' and 'torch_script'.")

    # set quantizer status and run simple evaluation
    self.quantizer.quant_mode = 2
    register_output_hook(self.quantizer.quant_model, record_once=True)
    set_outputs_recorder_status(self.quantizer.quant_model, True)
    if self.quantizer.fast_finetuned:
      #self.advanced_quant_setup()
      self.quantizer.load_param()
    run_fn(*run_args)

    # export quantized model
    if fmt == 'xmodel':
      self.export_xmodel(self.quantizer.output_dir, deploy_check=False)
    elif fmt == 'onnx':
      self.export_onnx_model(self.quantizer.output_dir, verbose=True, dynamic_batch=dynamic_batch)
    elif fmt == 'torch_script':
      self.export_traced_torch_script(self.quantizer.output_dir, verbose=True)
    NndctScreenLogger().info(f'Quantized model depoyment end.')

  # quantization steps of quantized tensors
  def export_quant_config(self):
    self.quantizer.export_quant_config()

  # export xmodel file to be compiled for deployment
  def export_xmodel(self, output_dir, deploy_check=False, dynamic_batch=False):
    if quant_model_inferenced(self.quantizer.quant_model) is False:
      NndctScreenLogger().error2user(QError.NO_FORWARD, f"torch_quantizer.quant_model FORWARD function must be called before exporting quantization result.\n    Please refer to example code at https://github.com/Xilinx/Vitis-AI/blob/master/src/Vitis-AI-Quantizer/vai_q_pytorch/example/resnet18_quant.py.")
      return
    dump_xmodel(output_dir, deploy_check, self._lstm_app)
    dump_all_fmt = os.getenv('NNDCT_DUMP_ALL_FORMAT')
    if dump_all_fmt is not None:
      self.export_onnx_model(output_dir, dynamic_batch=dynamic_batch)
      self.export_traced_torch_script(output_dir)
  
  def export_onnx_model(self, output_dir, verbose=False, dynamic_batch=False, opset_version=None):
    if quant_model_inferenced(self.quantizer.quant_model) is False:
      NndctScreenLogger().error2user(QError.NO_FORWARD, f"torch_quantizer.quant_model FORWARD function must be called before exporting quantization result.\n    Please refer to example code at https://github.com/Xilinx/Vitis-AI/blob/master/src/Vitis-AI-Quantizer/vai_q_pytorch/example/resnet18_quant.py.")
      return
    if self.quantizer.is_lstm:
      self.export_onnx_model_for_lstm(output_dir, verbose, dynamic_batch, opset_version)
    else:
      self.export_onnx_runable_model(output_dir, verbose, dynamic_batch, opset_version)

  def export_onnx_model_for_lstm(self, output_dir, verbose=False, dynamic_batch=False, opset_version=None):
    from torch.onnx import register_custom_op_symbolic
    from torch.onnx.symbolic_helper import parse_args

    @parse_args("v", "v", "v", "v", "v", "v", "v", "v")
    def symbolic_fix_neuron(g, input, valmin, valmax, valamp, zero_point, method, device_id, inplace):
      return g.op("vai::fix_neuron", input, valmax, valamp, method, device_id, inplace).setType(input.type())
    
    register_custom_op_symbolic("vai::fix_neuron", symbolic_fix_neuron, 9)
    opset_version = get_opset_version() if opset_version is None else opset_version
    device = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_DEVICE)

    script_models = self.quantizer.scripts if len(self.quantizer.scripts) > 0 else None
    self.quantizer.reset_status_for_exporting()
    if script_models is None:
      output_file = os.path.join(output_dir, f"{self.quantizer.quant_model._get_name()}_int.onnx")
      model, input_args = to_device(self.quantizer.quant_model, self._example_inputs, device)
      try:
        torch.onnx.export(self.quantizer.quant_model, input_args, output_file, 
                          verbose=verbose, 
                          input_names=self._input_tensors_name,
                          opset_version=opset_version,
                          custom_opsets={'vai' : 2})
      except Exception as e:
        NndctScreenLogger().error2user(QError.EXPORT_ONNX, f"The {get_module_name(self.quantizer.quant_model)} can not be exported for onnx. The PyTorch internal failed reason is:\n{str(e)}.")
      else:
        NndctScreenLogger().info(f"{get_module_name(self.quantizer.quant_model)}_int.onnx is generated.({output_file})")
      # check onnxruntime
      if NndctOption.nndct_check_onnx.value or os.getenv('NNDCT_CHECK_ONNX'): 
        if output_file is not None and os.path.exists(output_file):
          check_onnxruntime(output_file, input_args)
        else:
          NndctScreenLogger().warning(f"Exported onnx model is tested failed: the file {output_file} does not exist!")
    else:
      if len(script_models) == 1:
        inputs = [self._example_inputs]
        inputs_name = [self._input_tensors_name]
        outputs_name = [self._return_tensors_name]
      else:
        inputs = self._example_inputs
        inputs_name = self._input_tensors_name
        outputs_name = self._return_tensors_name
      
      for script, inputs, input_name, output_name in zip(script_models, inputs, inputs_name, outputs_name):
        q_model = self.convert_script_to_qscript(script, verbose=verbose)
        _, inputs = to_device(None, inputs, device)
        output_file =  os.path.join(output_dir, f"{get_module_name(q_model)}_int.onnx")
        try:
          torch.onnx.export(q_model, inputs, output_file, 
                            verbose=verbose, 
                            input_names=input_name,
                            opset_version=opset_version, 
                            custom_opsets={'vai' : 2})
        except Exception as e:
          NndctScreenLogger().error2user(QError.EXPORT_ONNX, f"The {get_module_name(q_model)} can not be exported for onnx. The PyTorch internal failed reason is:\n{str(e)}.")
        NndctScreenLogger().info(f"{get_module_name(q_model)}_int.onnx is generated.({output_file})")
        # check onnxruntime
        if NndctOption.nndct_check_onnx.value or os.getenv('NNDCT_CHECK_ONNX'): 
          if output_file is not None and os.path.exists(output_file):
            check_onnxruntime(output_file, inputs)
          else:
            NndctScreenLogger().warning(f"Exported onnx model check failed: the file {output_file} does not exist!")

  def export_onnx_runable_model(self, output_dir, verbose=False, dynamic_batch=False, opset_version=None):
    from torch.onnx import register_custom_op_symbolic
    from torch.onnx.symbolic_helper import parse_args
    import sys
    if get_torch_version() < 170:
      NndctScreenLogger().error2user(QError.TORCH_VERSION, f'Only supprt exporting onnx model with pytorch 1.7 and later version.')
      return
    
    if self.quantizer.contain_channel_quantize():
      if int(torch_version[0]) == 1 and int(torch_version[1]) < 10:
        NndctScreenLogger().error2user(QError.TORCH_VERSION, f'Only supprt exporting per_channel quantization onnx model with pytorch 1.10 and later version.')
        return
    
    @parse_args("v", "i", "i", "f", "i", "i", "i", "i")
    def symbolic_fix_neuron(g, input, valmin, valmax, valamp, zero_point, method, device_id, inplace):
      #print(f'{valmax} {valamp} {method} {device_id}')
      if valamp < sys.float_info.min:
        scale = torch.tensor(sys.float_info.max).float()  # Avoid exportor generating double type
      else:
        scale = torch.tensor(1.0 / valamp).float()  # Avoid exportor generating double type
      zero_point = torch.tensor(0, dtype=torch.int8)  # ONNX requires zero_point to be tensor
      if not isinstance(input, torch._C.Value) or not isinstance(valmin, int) or not isinstance(valmax, int) \
              or not isinstance(valamp, float) or zero_point.dtype != torch.int8 or not isinstance(method, int) \
              or not isinstance(device_id, int) or not isinstance(inplace, int) or valamp <= 0.:
        NndctScreenLogger().error2user(QError.FIX_INPUT_TYPE, f'Data type or value illegal fix neuron in when exporting onnx model.')
      return g.op("DequantizeLinear", g.op("QuantizeLinear", input, scale, zero_point), scale, zero_point)
      # an alternative method for checking onnx
      # if NndctOption.nndct_check_onnx.value or os.getenv('NNDCT_CHECK_ONNX'): # y = ((round(clip(x/scale + zero_point, min, max))) - zero_point)*scale
      #   return g.op("Mul", g.op("Sub", g.op("Round", g.op("Clip", g.op("Add", g.op("Div", input, scale), zero_point), torch.tensor(-128.0), torch.tensor(127.0))), zero_point), scale)
      # else:
      #   return g.op("DequantizeLinear", g.op("QuantizeLinear", input, scale, zero_point), scale, zero_point)
    
    def check_onnxruntime(onnx_path, input):
      import onnxruntime as ort
      import onnx
      import numpy as np
      
      # quantizer result
      option_util.set_option_value("nndct_use_torch_quantizer", True)
      self.quantizer.reset_status_for_exporting()
      quant_out = self.quantizer.quant_model(input)
      
      # load onnx model
      onnx_model = onnx.load(onnx_path)
      
      # Check that the model is well formed
      onnx.checker.check_model(onnx_model)
      
      # run exported model
      ort_sess = ort.InferenceSession(onnx_path)
      input_args = {}
      if isinstance(input, (tuple, list)):
        for i in range(len(input)):
          input_name = ort_sess.get_inputs()[i].name
          input_data = input[i][0].cpu().numpy() if isinstance(input[i], (tuple, list)) else input[i].cpu().numpy()
          input_args[input_name] = input_data
      else:
        input_args[ort_sess.get_inputs()[0].name] = input.cpu().numpy()
      ort_out = ort_sess.run(None, input_args)
       
      # compare onnxruntime with quantizer
      rtol = 1e-4
      atol = 1e-5
      if isinstance(quant_out, (tuple, list)):
        for i in range(len(quant_out)):
          np.testing.assert_allclose(quant_out[i].detach().cpu().numpy(), ort_out[i], rtol=rtol, atol=atol)
        NndctScreenLogger().info(f"Exported onnx model is tested successfully!")
      elif isinstance(quant_out, torch.Tensor):
        np.testing.assert_allclose(quant_out.detach().cpu().numpy(), ort_out[0], rtol=rtol, atol=atol)
        NndctScreenLogger().info(f"Exported onnx model is tested successfully!")
      else:
        NndctScreenLogger().warning(f"Exported onnx model is tested failed: quant output is not tensor, tuple or list!")

    register_custom_op_symbolic("vai::fix_neuron", symbolic_fix_neuron, 9)
    opset_version = get_opset_version() if opset_version is None else opset_version
    device = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_DEVICE)
    script_models = self.quantizer.scripts if len(self.quantizer.scripts) > 0 else None
    self.quantizer.reset_status_for_exporting()
    if dynamic_batch: # dynamic batch N in [N, C, L, H, W]
      dynamic_axes = {}
      for i in range(len(self._input_tensors_name)):
        dynamic_axes[self._input_tensors_name[i]] = [0]
    else:
      dynamic_axes = None

    if script_models is None:
      
      output_file = os.path.join(output_dir, f"{self.quantizer.quant_model._get_name()}_int.onnx")
      model, input_args = to_device(self.quantizer.quant_model, self._example_inputs, device)
      try:
        torch.onnx.export(self.quantizer.quant_model.eval(), input_args, output_file, 
                          input_names=self._input_tensors_name,
                          verbose=verbose, 
                          opset_version=opset_version,
                          dynamic_axes=dynamic_axes)
      except Exception as e:
         NndctScreenLogger().error2user(QError.EXPORT_ONNX, f"The {get_module_name(self.quantizer.quant_model)} can not be exported for onnx. The PyTorch internal failed reason is:\n{str(e)}")
      else:
        NndctScreenLogger().info(f"{get_module_name(self.quantizer.quant_model)}_int.onnx is generated.({output_file})")
      # check onnxruntime
      if NndctOption.nndct_check_onnx.value or os.getenv('NNDCT_CHECK_ONNX'):
        if output_file is not None and os.path.exists(output_file):
          check_onnxruntime(output_file, input_args)
        else:
          NndctScreenLogger().warning(f"Exported onnx model is tested failed: the file {output_file} does not exist!")
    else:
      if len(script_models) == 1:
        inputs = [self._example_inputs]
        inputs_name = [self._input_tensors_name]
        outputs_name = [self._return_tensors_name]
      else:
        inputs = self._example_inputs
        inputs_name = self._input_tensors_name
        outputs_name = self._return_tensors_name
      
      for script, inputs, input_name, output_name in zip(script_models, inputs, inputs_name, outputs_name):
        q_model = self.convert_script_to_qscript(script, verbose=verbose)
        _, inputs = to_device(None, inputs, device)
        output_file =  os.path.join(output_dir, f"{get_module_name(q_model)}_int.onnx")
        try:
          torch.onnx.export(q_model, inputs, output_file, 
                            input_names=input_name,
                            verbose=verbose, 
                            opset_version=opset_version,
                            dynamic_axes=dynamic_axes)
        except Exception as e:
          NndctScreenLogger().error2user(QError.EXPORT_ONNX, f"The {get_module_name(q_model)} can not be exported for onnx. The PyTorch internal failed reason is:\n{str(e)}.")
        else:
          NndctScreenLogger().info(f"{get_module_name(q_model)}_int.onnx is generated.({output_file}).")
        # check onnxruntime
        if NndctOption.nndct_check_onnx.value or os.getenv('NNDCT_CHECK_ONNX'): 
          if output_file is not None and os.path.exists(output_file):
            check_onnxruntime(output_file, inputs)
          else:
              NndctScreenLogger().warning(f"Exported onnx model check failed: the file {output_file} does not exist!")

  def export_traced_torch_script(self, output_dir, verbose=False):
    if quant_model_inferenced(self.quantizer.quant_model) is False:
      NndctScreenLogger().error2user(QError.NO_FORWARD, f"torch_quantizer.quant_model FORWARD function must be called before exporting quantization result.\n    Please refer to example code at https://github.com/Xilinx/Vitis-AI/blob/master/src/Vitis-AI-Quantizer/vai_q_pytorch/example/resnet18_quant.py.")
      return
    nndct_utils.create_work_dir(output_dir)
    torch_version = torch.__version__.split('.')
    if int(torch_version[0]) == 1 and int(torch_version[1]) < 7:
      NndctScreenLogger().error2user(QError.TORCH_VERSION, f'Only supprt exporting torch script with pytorch 1.7 and later version.')
      return
    self.quantizer.reset_status_for_exporting()
    device = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_DEVICE)
    force_cpu = os.getenv('NNDCT_FORCE_CPU_DUMP')
    if force_cpu is not None:
      device = torch.device('cpu')
      GLOBAL_MAP.set_map(NNDCT_KEYS.QUANT_DEVICE, device)
    model, input_args = to_device(self.quantizer.quant_model, self._example_inputs, device)
    script_module = torch.jit.trace(model, input_args, check_trace=False)
    output_file = os.path.join(output_dir, f"{self.quantizer.quant_model._get_name()}_int.pt")
    if verbose is True:
      print(script_module.inlined_graph)
    torch.jit.save(script_module, output_file)
    NndctScreenLogger().info(f"{self.quantizer.quant_model._get_name()}_int.pt is generated.({output_file})")
    return script_module

  def convert_script_to_qscript(self, script_model, verbose=False):
    script_model = opt_script_model_for_quant(script_model)
    if isinstance(self.quantizer.quant_model, list):
      for qmod in self.quantizer.quant_model:
        ModuleHooker.update_blobs_once(qmod, update_shape_only=True)
    else:
      ModuleHooker.update_blobs_once(self.quantizer.quant_model, update_shape_only=True)
      
    quantized_script_model = insert_fix_neuron_in_script_model(script_model, self.quantizer)
    quantized_script_model = insert_mul_after_avgpool(quantized_script_model, self.quantizer)
    quantized_script_model = remove_quant_dequant_stub(quantized_script_model)
    if verbose:
      print(quantized_script_model.graph)
    return quantized_script_model

  def export_torch_script(self, output_dir, verbose=False): 
    if quant_model_inferenced(self.quantizer.quant_model) is False:
      NndctScreenLogger().error2user(QError.NO_FORWARD, f"torch_quantizer.quant_model FORWARD function must be called before exporting quantization result.\n    Please refer to example code at https://github.com/Xilinx/Vitis-AI/blob/master/src/Vitis-AI-Quantizer/vai_q_pytorch/example/resnet18_quant.py.")
      return
    if isinstance(self.quantizer.quant_model, torch.nn.Module) and len(self.quantizer.Nndctgraph.all_blocks()) == 1 \
        and NndctOption.nndct_export_jit.value is False:
      return self.export_traced_torch_script(output_dir=output_dir, verbose=verbose)
   

    if get_torch_version() <= 1100:
      NndctScreenLogger().error2user(QError.TORCH_VERSION, f'Only supprt exporting torch script with pytorch 1.10 and later version.')
      return
    nndct_utils.create_work_dir(output_dir)
    script_models = self.quantizer.scripts if len(self.quantizer.scripts) > 0 else None
    NndctScreenLogger().check2user(QError.NO_SCRIPT_MODEL, 'Quantizer does not find any script model.', script_models is not None)
    quantized_scripts = []
    for script_model in script_models:
      quantized_script_model = self.convert_script_to_qscript(script_model, verbose)
      output_file = os.path.join(output_dir, f"{get_module_name(script_model)}_int.pt")
      torch.jit.save(quantized_script_model, output_file)
      NndctScreenLogger().info(f"{get_module_name(script_model)}_int.pt is generate.({output_file})")
      quantized_scripts.append(quantized_script_model)
    if len(quantized_scripts) == 1:
      return quantized_scripts[0]
    else:
      return quantized_scripts
      
def dump_xmodel(output_dir="quantize_result", deploy_check=False, lstm_app=False):
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
    deploy_graphs, _ = get_deploy_graph_list(quantizer.quant_model, quantizer.Nndctgraph)
    xmodel_depoly_infos, dump_deploy_infos = get_xmodel_and_dump_infos(quantizer, deploy_graphs)
    if not lstm_app:
      for node in xmodel_depoly_infos[0].dev_graph.nodes:
        error_out = False
        if node.op.type not in [NNDCT_OP.INPUT, NNDCT_OP.QUANT_STUB]:
          continue
        for i, tensor in enumerate(node.out_tensors):
          if tensor.shape and tensor.shape[0] != 1:
            NndctScreenLogger().error2user(QError.XMODEL_BATCHSIZE, f"Batch size must be 1 when exporting xmodel.")
            error_out = True
            break
        if error_out:
          break
      
    for depoly_info in dump_deploy_infos:
      # dump data for accuracy check
      if deploy_check:
        # sync data
        NndctScreenLogger().info(f"=>Dumping '{depoly_info.dev_graph.name}'' checking data...")
        # connect_module_with_graph(quantizer.quant_model, depoly_info.dev_graph, recover_param=False)
        # update_nndct_blob_data(quantizer.quant_model, depoly_info.dev_graph)
        # connect_module_with_graph(quantizer.quant_model, quantizer.Nndctgraph, recover_param=False)  
        last_only = os.getenv('NNDCT_ONLY_DUMP_LAST')
        if lstm_app:
          checker = DeployChecker(output_dir_name=output_dir, data_format='txt')
          checker.update_dump_folder(f"{depoly_info.dev_graph.name}/frame_0")
          select_batch = True
        else:
          checker = DeployChecker(output_dir_name=output_dir, last_only=(last_only is not None))
          checker.update_dump_folder(f"{depoly_info.dev_graph.name}")
          select_batch = False
        checker.quant_data_for_dump(quantizer, depoly_info)
        checker.dump_nodes_output(
            depoly_info.dev_graph,
            depoly_info.quant_info,
            round_method=quantizer.quant_opt['round_method'], select_batch=select_batch)
        
        NndctScreenLogger().info(f"=>Finsh dumping data.({checker.dump_folder})")
    
    if quantizer.quant_strategy_info['target_device'] == "DPU":
      for depoly_info in xmodel_depoly_infos:
        try:
          valid, msg = compiler.verify_nndct_graph(depoly_info.dev_graph)
          if not valid:
            NndctScreenLogger().warning2user(QWarning.CONVERT_XMODEL, f"""Convert '{depoly_info.dev_graph.name}' to xmodel failed with following reasons:\n{msg}""")
            continue
          xgraph = compiler.do_compile(
              depoly_info.dev_graph,
              quant_config_info=depoly_info.quant_info,
              output_file_name=os.path.join(output_dir, depoly_info.dev_graph.name))

        except AddXopError as e:
          NndctScreenLogger().error2user(QError.EXPORT_XMODEL, f"Failed convert graph '{depoly_info.dev_graph.name}' to xmodel.")
          raise e
        
        compiler.verify_xmodel(depoly_info.dev_graph, xgraph)
    else:
      NndctScreenLogger().warning2user(QWarning.XMODEL_DEVICE, f"Not support to dump xmodel when target device is not DPU.")
    

def vaiq_system_info(device):
  # Check device available
  if device.type == "cuda":
    if not (torch.cuda.is_available() and ("CUDA_HOME" or "ROCM_HOME" in os.environ)):
      device = torch.device("cpu")
      NndctScreenLogger().warning2user(QWarning.CUDA_UNAVAILABLE, f"CUDA (HIP) is not available, change device to CPU")
  
  NndctScreenLogger().check2user(QError.TORCH_VERSION, f"Installed pytorch version is {torch.__version__}, \
not consistent with pytorch version when compiling quantizer ({Ctorch_version})", 
      torch.__version__ in __version__)
  import platform as pf
  import sys
  long_ver = sys.version
  py_ver = '3.x'
  gcc_ver = 'x.x'
  result = re.findall(r'\d+.\d+.\d+', long_ver)
  if len(result) > 0:
    py_ver = result[0]
  result = re.findall(r'GCC \d+.\d+.\d+', long_ver)
  if len(result) > 0:
    gcc_ver = result[0]
  NndctScreenLogger().info(f'OS and CPU information:\n\
               system --- {pf.system()}\n\
                 node --- {pf.node()}\n\
              release --- {pf.release()}\n\
              version --- {pf.version()}\n\
              machine --- {pf.machine()}\n\
            processor --- {pf.processor()}')
  NndctScreenLogger().info(f'Tools version information:\n\
                  GCC --- {gcc_ver}\n\
               python --- {py_ver}\n\
              pytorch --- {torch.__version__}\n\
        vai_q_pytorch --- {__version__}')
  if device == torch.device('cuda'):
    NndctScreenLogger().info(f'GPU information:\n\
          device name --- {torch.cuda.get_device_name()}\n\
     device available --- {torch.cuda.is_available()}\n\
         device count --- {torch.cuda.device_count()}\n\
       current device --- {torch.cuda.current_device()}')

