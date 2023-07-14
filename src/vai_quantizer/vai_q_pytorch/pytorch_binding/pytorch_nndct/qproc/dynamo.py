
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
import functools
import itertools
import types

import torch
from torch.utils._python_dispatch import _disable_current_modes

from nndct_shared.base import GLOBAL_MAP, NNDCT_KEYS, NNDCT_OP

from nndct_shared.nndct_graph import (Block, Graph, Node,
                                      convert_graph_to_block_node)
from nndct_shared.nndct_graph import operator_definition as base_op
import nndct_shared.utils as nndct_utils

from .ModuleHooker import ModuleHooker
from .utils import (prepare_quantizable_module, register_output_hook,
                    set_outputs_recorder_status)

from pytorch_nndct.quantization.torch_qconfig import TorchQConfig
from pytorch_nndct.quantization.fakequantizer import FakeQuantizer
from pytorch_nndct.quantization.torchquantizer import TORCHQuantizer
from nndct_shared.utils import (NndctOption, NndctScreenLogger,
                                option_util, QError, QWarning)

from torch._dynamo.backends.common import fake_tensor_unsupported


def build_root_graph():
  root_graph = Graph("root")
  op = base_op.CustomOp(NNDCT_OP.PLACEHOLDER)
  input_node = Node(name="input_placeholder", op=op, in_quant_part=False)
  input_node.owning_graph = root_graph
  op = base_op.CustomOp(NNDCT_OP.PLACEHOLDER)
  return_node = Node(name="return_placeholder", op=op, in_quant_part=False)
  return_node.owning_graph = root_graph
  
  top_block = Block(root_graph, None, input_node, return_node)
  root_graph.set_top_block(top_block)
  return root_graph


def init_graph_counter():
  counter = itertools.count(0)
  GLOBAL_MAP.set_map(NNDCT_KEYS.GRAPH_COUNTER, counter)

def get_graph_id():
  return next(GLOBAL_MAP.get_ele(NNDCT_KEYS.GRAPH_COUNTER))

def convert_gm_to_quantizable_module(gm, example_inputs, *, quantizer, root_graph, quant_models, graph_id, device):
  def _get_name(self):
      return f"GraphModule_{graph_id}"
  if len(example_inputs) == 0:
    return gm
  gm._get_name = types.MethodType(_get_name, gm)

  quant_module, graph = prepare_quantizable_module(module=gm, 
                                                  input_args=tuple(example_inputs), 
                                                  export_folder=quantizer.output_dir,
                                                  device=device,
                                                  connect_qm_with_graph=True)
  if len(graph.all_blocks()) > 1:
    quant_module.from_script(True)
  else:
    quant_module.from_script(False)
  
  # if quant_mode > 1 and jit_compile is False:
  #   register_output_hook(quant_module, record_once=True)
  #   set_outputs_recorder_status(quant_module, True)

  block_node = convert_graph_to_block_node(root_graph, graph)
  if not block_node.in_node_list():
    root_graph.append_node(block_node)
  quant_info, history_quant_info = copy.deepcopy(quantizer.quant_config), copy.deepcopy(quantizer.config_history)
  try:
    quantizer.setup(root_graph, False, False, custom_quant_ops=None, dynamo=True)
  except RuntimeError:
    quant_module.eval()
    ModuleHooker.hook_module_with_quantizer(quant_module, None)
  else:
    update_quant_config(quantizer, quant_info, history_quant_info)
    quant_module.eval()
    quant_models.append(quant_module)
    ModuleHooker.hook_module_with_quantizer(quant_module, quantizer)
    quantizer.quant_model = quant_models
  return quant_module

def _gen_traced_quantized_script(gm, example_inputs, quantizer, root_graph, quant_model_lst, graph_id, device):
  quantizable_model = convert_gm_to_quantizable_module(gm, 
                                                      example_inputs, 
                                                      quantizer=quantizer, 
                                                      root_graph=root_graph, 
                                                      quant_models=quant_model_lst, 
                                                      graph_id=graph_id, 
                                                      device=device)
  _ = quantizable_model(*example_inputs)
  quantizer.reset_status_for_exporting()
  return torch.jit.trace(quantizable_model, tuple(example_inputs), check_trace=False) 


def aot_module_quantize(quantizer, quant_mode, *, device):

  root_graph = build_root_graph()
  quant_model_lst = []
  init_graph_counter()

  @fake_tensor_unsupported
  def quantizable_forward(gm, example_inputs):
    # TODO: Try to avoid to generate a new graph without any modification
    graph_id = get_graph_id()

    quant_module = convert_gm_to_quantizable_module(gm, 
                                      example_inputs, 
                                      quantizer=quantizer, 
                                      root_graph=root_graph, 
                                      quant_models=quant_model_lst, 
                                      graph_id=graph_id, 
                                      device=device)
    @torch.no_grad()
    def run(*args):
      result = quant_module.forward(*args)
      # There is a unpack sequence instruction after compiled fn in bytecode, so we need to construct a tuple for results of compiled fn
      if not isinstance(result, tuple):
        return (result, )
      else:
        return result
    return run
  return quantizable_forward


def update_quant_config(quantizer, quant_info, history_quant_info):
  for quant_type in quantizer.quant_config:
    if quant_type not in quant_info:
      continue
    if isinstance(quantizer.quant_config[quant_type], dict):
      for item in quantizer.quant_config[quant_type]:
        if item in quant_info[quant_type] and quant_info[quant_type][item]:
          quantizer.quant_config[quant_type][item] = copy.deepcopy(quant_info[quant_type][item])
    else:
      quantizer.quant_config[quant_type] = copy.deepcopy(quant_info[quant_type])
  
  for quant_type in quantizer.config_history:
    for item in quantizer.config_history[quant_type]:
      if item in history_quant_info[quant_type] and history_quant_info[quant_type][item]:
        quantizer.config_history[quant_type][item] = copy.deepcopy(history_quant_info[quant_type][item]) 


# WEGO API
def init_wego_dynamo_env(output_dir, device=torch.device("cpu"), quant_config_file=None):
    torch._dynamo.reset()
    nndct_utils.create_work_dir(output_dir)
   
    # Parse the quant config file
    QConfiger = TorchQConfig()
    #if quant_config_file:
    QConfiger.parse_config_file(quant_config_file, 
                                bit_width_w=8, 
                                bit_width_a=8, 
                                mix_bit=False)
    qconfig = QConfiger.qconfig
    if NndctOption.nndct_dump_quant_config.value is True:
      config_dump_file = '/'.join([output_dir, 'effective_config.json'])
      QConfiger.dump_quant_config(config_dump_file)
    quantizer, qmode = init_quant_env("test", output_dir, qconfig)
    # device = torch.device("")   
    GLOBAL_MAP.set_map(NNDCT_KEYS.QUANT_DEVICE, device)
    root_graph = build_root_graph()
    init_graph_counter()
    quant_model_lst = []
    _wego_traced_script_fn = functools.partial(_gen_traced_quantized_script, 
                                              quantizer=quantizer, 
                                              root_graph=root_graph, 
                                              quant_model_lst=quant_model_lst, 
                                              device=device)
    GLOBAL_MAP.set_map(NNDCT_KEYS.WEGO_DYNAMO_SCRIPTER, _wego_traced_script_fn)


# WEGO API
@torch.no_grad()
def get_traced_quantized_script(gm, example_inputs, graph_id):
  scripter_fn = GLOBAL_MAP.get_ele(NNDCT_KEYS.WEGO_DYNAMO_SCRIPTER)
  script_model = scripter_fn(gm, example_inputs, graph_id=graph_id)
  return script_model


def init_quant_env(quant_mode, output_dir, quant_strategy_info, is_lstm=False):
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
  GLOBAL_MAP.set_map(NNDCT_KEYS.QUANTIZER, quantizer)
  GLOBAL_MAP.set_map(NNDCT_KEYS.QUANT_MODE, qmode)
  GLOBAL_MAP.set_map(NNDCT_KEYS.QUANT_CONFIG, quant_strategy_info)
  return quantizer, qmode
