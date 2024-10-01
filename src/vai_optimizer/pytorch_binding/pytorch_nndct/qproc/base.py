
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
from nndct_shared.nndct_graph import Graph, Node, Block, convert_graph_to_block_node
from nndct_shared.nndct_graph import operator_definition as base_op
#from nndct_shared.quantization import DefaultQstrategy, QStrategyFactory
from pytorch_nndct.quantization import (FakeQuantizer, TorchQConfig,
                                        TORCHQuantizer)
from pytorch_nndct.utils.torch_utils import CmpFlag, compare_torch_version
from pytorch_nndct.utils.onnx_utils import get_opset_version
from pytorch_nndct.utils.module_util import visualize_tensors, to_device, get_module_name
from nndct_shared.compile import get_xmodel_and_dump_infos
from nndct_shared.quantization.fix_pos_adjust import FixPosInserter
from .adaquant import AdvancedQuantProcessor
from .utils import (connect_module_with_graph, disconnect_modeule_with_graph,
                    get_deploy_graph_list, insert_fix_neuron_in_script_model,
                    opt_script_model_for_quant, prepare_quantizable_module,
                    register_output_hook, set_outputs_recorder_status,
                    update_nndct_blob_data, quant_model_inferenced,
                    insert_mul_after_avgpool,
                    remove_quant_dequant_stub)
from .ModuleHooker import ModuleHooker
from .onnx import export_onnx_model_for_lstm, export_onnx_runable_model
from pytorch_nndct.parse.rich_in_out_helper import RecoveryModel, StandardInputData, flatten_to_tuple

class TorchQuantProcessor():

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
    if target_device in ['DPU','FLEXML']:
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
               input_data : StandardInputData,
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
    if not isinstance(input_data, StandardInputData):
      if not isinstance(input_data, tuple):
        input_data = (input_data,)
      input_data = StandardInputData(input_data, {}, device)
    self.device = device
    module.to(device)
    from pytorch_nndct.utils.jit_utils import set_training
    with torch.no_grad() ,set_training(module, False):
      tmp_args = copy.deepcopy(input_data.args)
      tmp_kwargs = copy.deepcopy(input_data.kwargs)
      _, self.module_output_schema = flatten_to_tuple(module(*tmp_args, **tmp_kwargs))
    # Check arguments type
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
        input_args=input_data,
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
      if tmp_module is not None:
        tmp_module = tmp_module.to(device)

      quant_off_stat = NndctOption.nndct_quant_off.value
      param_corr_stat = NndctOption.nndct_param_corr.value
      nndct_utils.set_option_value("nndct_quant_off", True)
      nndct_utils.set_option_value("nndct_param_corr", False)
      register_output_hook(tmp_module, record_once=True)
      set_outputs_recorder_status(tmp_module, True)
      tmp_module.eval()
      _ = tmp_module(*input_data.args, **input_data.kwargs)


      nndct_utils.set_option_value("nndct_quant_off", quant_off_stat)
      nndct_utils.set_option_value("nndct_param_corr", param_corr_stat)
      _, dev_graph = get_deploy_graph_list(tmp_module, graph, need_partition=False)
      quantizer.setup(graph, False, lstm_app, custom_quant_ops=custom_quant_ops, target=target)
      quantizer.configer.assign_device_info(dev_graph)
      quantizer.configer.filter_quant_config_by_device_info()
      dev_graph.clean_tensors_data()

    # connect module with graph
    connect_module_with_graph(quant_module, graph)
    # hook module with quantizer
    quantizer.quant_model = quant_module
    if GLOBAL_MAP.get_ele(NNDCT_KEYS.TORCH_SCRIPT_MODEL):
      quantizer.add_script(GLOBAL_MAP.get_ele(NNDCT_KEYS.TORCH_SCRIPT_MODEL))

    flatten_input_for_innermodel, _ = input_data.make_flatten_data()
    self._example_inputs = flatten_input_for_innermodel
    if isinstance(flatten_input_for_innermodel, torch.Tensor):
      quant_module._input_tensors_name = graph.get_input_tensors([flatten_input_for_innermodel])
      self._input_tensors_name = quant_module._input_tensors_name
    else:
      quant_module._input_tensors_name = graph.get_input_tensors(flatten_input_for_innermodel)
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
    return RecoveryModel(self.quantizer.quant_model, self.module_output_schema, self.device)
   
  
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
    else:
      adaquant = self.advanced_quant_setup(self.quantizer.quant_model, self.quantizer.graph, self._example_inputs)
      if NndctOption.nndct_ft_mode.value == 0:
        adaquant.finetune(run_fn, run_args)
      else:
        adaquant.finetune_v2(run_fn, run_args)

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
      self.export_onnx_model(self.quantizer.output_dir, verbose=True, dynamic_batch=dynamic_batch, opset_version=None)
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
      self.export_onnx_model(output_dir, dynamic_batch=dynamic_batch, opset_version=None)
      self.export_traced_torch_script(output_dir)

  def export_onnx_model(self, output_dir, verbose=False, dynamic_batch=False, opset_version=None, native_onnx=True, dump_layers=False, check_model=False, opt_graph=False):
    fixposinserter = FixPosInserter(self.quantizer)
    fixposinserter(self.quantizer.Nndctgraph)

    if quant_model_inferenced(self.quantizer.quant_model) is False:
      NndctScreenLogger().error2user(QError.NO_FORWARD, f"torch_quantizer.quant_model FORWARD function must be called before exporting quantization result.\n    Please refer to example code at https://github.com/Xilinx/Vitis-AI/blob/master/src/Vitis-AI-Quantizer/vai_q_pytorch/example/resnet18_quant.py.")
      return
    if self.quantizer.is_lstm:
      export_onnx_model_for_lstm(self.quantizer,
                                 self._example_inputs,
                                 self._input_tensors_name,
                                 self._return_tensors_name,
                                 self.convert_script_to_qscript,
                                 output_dir,
                                 verbose,
                                 dynamic_batch,
                                 opset_version,
                                 native_onnx,
                                 dump_layers,
                                 check_model,
                                 opt_graph)
    else:
      export_onnx_runable_model(self.quantizer,
                                self._example_inputs,
                                self._input_tensors_name,
                                self._return_tensors_name,
                                self.convert_script_to_qscript,
                                output_dir,
                                verbose,
                                dynamic_batch,
                                opset_version,
                                native_onnx,
                                dump_layers,
                                check_model,
                                opt_graph)

  def export_traced_torch_script(self, output_dir, verbose=False):
    fixposinserter = FixPosInserter(self.quantizer)
    fixposinserter(self.quantizer.Nndctgraph)

    if quant_model_inferenced(self.quantizer.quant_model) is False:
      NndctScreenLogger().error2user(QError.NO_FORWARD, f"torch_quantizer.quant_model FORWARD function must be called before exporting quantization result.\n    Please refer to example code at https://github.com/Xilinx/Vitis-AI/blob/master/src/Vitis-AI-Quantizer/vai_q_pytorch/example/resnet18_quant.py.")
      return
    nndct_utils.create_work_dir(output_dir)
    if compare_torch_version(CmpFlag.LESS, "1.7.0"):
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


    if compare_torch_version(CmpFlag.LESS_EQUAL, "1.10.0"):
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

  fixposinserter = FixPosInserter(quantizer)
  fixposinserter(quantizer.Nndctgraph)

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

        NndctScreenLogger().info(f"=>Finish dumping data.({checker.dump_folder})")

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



partial_graph = False
def _check_args(module):
  if not isinstance(module, torch.nn.Module):
    raise TypeError(f"Type of 'module' should be 'torch.nn.Module'.")


def _init_quant_env(quant_mode, output_dir, quant_strategy_info, is_lstm=False):
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


class DynamoQuantProcessor(object):
  
  def __init__(self,
              quant_mode: str,
              module: torch.nn.Module,
              input_args: Union[torch.Tensor, Sequence[Any]] = None,
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
    import torch._dynamo
    from .dynamo import aot_module_quantize 
    _check_args(module)
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
    quantizer, qmode = _init_quant_env(quant_mode, 
                                            output_dir,
                                            qconfig)

    GLOBAL_MAP.set_map(NNDCT_KEYS.QUANTIZER, quantizer)
    GLOBAL_MAP.set_map(NNDCT_KEYS.QUANT_MODE, qmode)
    GLOBAL_MAP.set_map(NNDCT_KEYS.QUANT_DEVICE, device)
    GLOBAL_MAP.set_map(NNDCT_KEYS.QUANT_CONFIG, qconfig)
    if lstm_app:
      option_util.set_option_value("nndct_cv_app", False)
    else: 
      option_util.set_option_value("nndct_cv_app", True)

    quantizable_fn = aot_module_quantize(quantizer, qmode , device=device)
   
    self._dynamo_opt_model = torch._dynamo.optimize(backend=quantizable_fn, nopython=partial_graph)(module)
  
    # if isinstance(input_args, tuple):
    #   explanation, out_guards, graphs, ops_per_graph, _, verbose = torch._dynamo.explain(module, *input_args)
    # else:
    #   explanation, out_guards, graphs, ops_per_graph, _, verbose = torch._dynamo.explain(module, input_args)
    self._lstm_app = lstm_app
    self.quantizer = quantizer
    self._example_inputs = input_args

    
    # dump blob dist 
    if NndctOption.nndct_visualize.value is True:
      visualize_tensors(quantizer.quant_model)
      
    if NndctOption.nndct_calib_before_finetune.value is True:
      self.quantizer.export_float_param()
   
  def quant_model(self):
    return self._dynamo_opt_model

  def export_quant_config(self):
    self.quantizer.export_quant_config()

  def export_torch_script(self, output_dir, verbose=False):
    raise NotImplementedError("The dynamo mode doesn't support whole graph script exporting.")
  
  def export_onnx_model(self, output_dir, verbose=False, dynamic_batch=False, opset_version=None):
    raise NotImplementedError("The dynamo mode doesn't support whole graph onnx exporting.")
  
  def export_xmodel(self, output_dir, deploy_check=False):
    raise NotImplementedError("The dynamo mode doesn't support whole graph XIR exporting.")
   





