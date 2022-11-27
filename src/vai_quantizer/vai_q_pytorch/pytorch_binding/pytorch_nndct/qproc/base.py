
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
from typing import Any, Optional, Sequence, Union, List

import torch

import nndct_shared.utils as nndct_utils
from nndct_shared.base import GLOBAL_MAP, NNDCT_KEYS, NNDCT_OP
from nndct_shared.compile import CompilerFactory, DeployChecker
from nndct_shared.utils import AddXopError, NndctOption, NndctScreenLogger, option_util
from pytorch_nndct.utils.module_util import visualize_tensors
#from nndct_shared.quantization import DefaultQstrategy, QStrategyFactory
from pytorch_nndct.quantization import TORCHQuantizer, FakeQuantizer
from pytorch_nndct.quantization import TorchQConfig
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
               quant_config_file: Optional[str] = None):
    # Check arguments type
    self._check_args(module, input_args)
    
    # Check device available
    if device.type == "cuda":
      #if not (torch.cuda.is_available() and "CUDA_HOME" in os.environ):
      if not (torch.cuda.is_available() and ("CUDA_HOME" or "ROCM_HOME" in os.environ)):
        device = torch.device("cpu")
        #NndctScreenLogger().warning(f"CUDA is not available, change device to CPU")
        NndctScreenLogger().warning(f"CUDA (HIP) is not available, change device to CPU")
    
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
        device=device)
    
    # enable record outputs of per layer
    if qmode > 1:
      register_output_hook(quant_module, record_once=True)
      set_outputs_recorder_status(quant_module, True)
        
    # intialize quantizer 
    quantizer.setup(graph, False, lstm_app, custom_quant_ops=custom_quant_ops)
    #if qmode > 1:
    #  quantizer.features_check()

    # hook module with quantizer
    # connect_module_with_quantizer(quant_module, quantizer)
    quantizer.quant_model = quant_module
    self._example_inputs = input_args

    self._lstm_app = lstm_app
    self.quantizer = quantizer
    self.adaquant = None
    
    # dump blob dist 
    if NndctOption.nndct_visualize.value is True:
      visualize_tensors(quantizer.quant_model)

  def advanced_quant_setup(self):
    if (self.adaquant is None):
      self.adaquant = AdvancedQuantProcessor(self.quantizer)
  
  def quant_model(self):
    return self.quantizer.quant_model
  
  # function needs forwarding iteration control
  def finetune(self, run_fn, run_args):
    self.advanced_quant_setup()
    grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    if self.adaquant is not None:
      if NndctOption.nndct_ft_mode.value == 0:
        self.adaquant.finetune(run_fn, run_args)
      else:
        self.adaquant.finetune_v2(run_fn, run_args)
    torch.set_grad_enabled(grad_enabled)

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
      self.advanced_quant_setup()
    log_str = run_fn(*run_args)
    NndctScreenLogger().info(f'Quantized model evaluation returns metric:\n {log_str}')
    NndctScreenLogger().info(f'Quantized model end.')

  def deploy(self, run_fn, run_args, fmt='xmodel'):
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
      self.advanced_quant_setup()
    run_fn(*run_args)

    # export quantized model
    if fmt == 'xmodel':
      self.export_xmodel(self.quantizer.output_dir, deploy_check=False)
    elif fmt == 'onnx':
      self.export_onnx_model(self.quantizer.output_dir, verbose=True)
    elif fmt == 'torch_script':
      self.export_traced_torch_script(self.quantizer.output_dir, verbose=True)
    NndctScreenLogger().info(f'Quantized model depoyment end.')

  # quantization steps of quantized tensors
  def export_quant_config(self):
    self.quantizer.export_quant_config()

  # export xmodel file to be compiled for deployment
  def export_xmodel(self, output_dir, deploy_check=False):
    dump_xmodel(output_dir, deploy_check, self._lstm_app)
    dump_all_fmt = os.getenv('NNDCT_DUMP_ALL_FORMAT')
    if dump_all_fmt is not None:
      self.export_onnx_model(output_dir)
      self.export_traced_torch_script(output_dir)
  
  def export_onnx_model(self, output_dir, verbose=False):
    from torch.onnx import register_custom_op_symbolic
    from torch.onnx.symbolic_helper import parse_args
    import sys
    torch_version = torch.__version__.split('.')
    if int(torch_version[0]) == 1 and int(torch_version[1]) < 7:
      NndctScreenLogger().error(f'Only supprt exporting onnx model with pytorch 1.7 and later version')
      return
    
    if self.quantizer.contain_channel_quantize():
      if int(torch_version[0]) == 1 and int(torch_version[1]) < 10:
        NndctScreenLogger().error(f'Only supprt exporting per_channel quantization onnx model with pytorch 1.10 and later version')
        return
    
    @parse_args("v", "i", "i", "f", "i", "i", "i", "i")
    def symbolic_fix_neuron(g, input, valmin, valmax, valamp, zero_point, method, device_id, inplace):
      #print(f'{valmax} {valamp} {method} {device_id}')
      if valamp < sys.float_info.min:
        scale = torch.tensor(sys.float_info.max).float()  # Avoid exportor generating double type
      else:
        scale = torch.tensor(1.0 / valamp).float()  # Avoid exportor generating double type
      zero_point = torch.tensor(0, dtype=torch.int8)  # ONNX requires zero_point to be tensor
      return g.op("DequantizeLinear", g.op("QuantizeLinear", input, scale, zero_point), scale, zero_point)
    
    nndct_utils.create_work_dir(output_dir)
    register_custom_op_symbolic("vai::fix_neuron", symbolic_fix_neuron, 9)
    output_file = os.path.join(output_dir, f"{self.quantizer.quant_model._get_name()}_int.onnx")
    opset_version = torch.onnx.symbolic_helper._onnx_stable_opsets[-1]
    device = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_DEVICE)
    self.quantizer.reset_status_for_exporting()
    model, input_args = to_device(self.quantizer.quant_model, self._example_inputs, device)
    torch.onnx.export(self.quantizer.quant_model, input_args, output_file, 
                      verbose=verbose, 
                      opset_version=opset_version)

  def export_traced_torch_script(self, output_dir, verbose=False):
    torch_version = torch.__version__.split('.')
    if int(torch_version[0]) == 1 and int(torch_version[1]) < 7:
      NndctScreenLogger().error(f'Only supprt exporting torch script with pytorch 1.7 and later version')
      return
    nndct_utils.create_work_dir(output_dir)
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
    deploy_graphs = get_deploy_graph_list(quantizer.quant_model, quantizer.Nndctgraph)
    #depoly_infos = compiler.get_deloy_graph_infos(quantizer, deploy_graphs)
    xmodel_depoly_infos, dump_deploy_infos = compiler.get_xmodel_and_dump_infos(quantizer, deploy_graphs)
    if not lstm_app:
      for node in xmodel_depoly_infos[0].dev_graph.nodes:
        error_out = False
        if node.op.type not in [NNDCT_OP.INPUT, NNDCT_OP.QUANT_STUB]:
          continue
        for i, tensor in enumerate(node.out_tensors):
          if tensor.shape and tensor.shape[0] != 1:
            NndctScreenLogger().error(f"Batch size must be 1 when exporting xmodel.")
            error_out = True
            break
        if error_out:
          break
      
    for depoly_info in dump_deploy_infos:
      # dump data for accuracy check
      if deploy_check:
        NndctScreenLogger().info(f"=>Dumping '{depoly_info.dev_graph.name}'' checking data...")
        last_only = os.getenv('NNDCT_ONLY_DUMP_LAST')
        if lstm_app:
          checker = DeployChecker(output_dir_name=output_dir, data_format='txt')
          checker.update_dump_folder(f"{depoly_info.dev_graph.name}/frame_0")
          select_batch = True
        else:
          checker = DeployChecker(output_dir_name=output_dir, last_only=(last_only is not None))
          checker.update_dump_folder(f"{depoly_info.dev_graph.name}")
          select_batch = False
        checker.dump_nodes_output(
            depoly_info.dev_graph,
            depoly_info.quant_info,
            round_method=quantizer.quant_opt['round_method'], select_batch=select_batch)
        
        NndctScreenLogger().info(f"=>Finsh dumping data.({checker.dump_folder})")
    
    if quantizer.quant_strategy_info['target_device'] == "DPU":
      for depoly_info in xmodel_depoly_infos:
        try:
          xgraph = compiler.do_compile(
              depoly_info.dev_graph,
              quant_config_info=depoly_info.quant_info,
              output_file_name=os.path.join(output_dir, depoly_info.dev_graph.name))

        except AddXopError as e:
          NndctScreenLogger().error(f"Failed convert graph '{depoly_info.dev_graph.name}' to xmodel.")
          raise e
        
        compiler.verify_xmodel(depoly_info.dev_graph, xgraph)
    else:
      NndctScreenLogger().warning(f"Not support to dump xmodel when target device is not DPU")
    
    set_outputs_recorder_status(quantizer.quant_model, False)
