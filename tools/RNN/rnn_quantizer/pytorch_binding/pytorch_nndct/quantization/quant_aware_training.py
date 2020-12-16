

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
import torch
import types

from nndct_shared.base import GLOBAL_MAP, NNDCT_KEYS
from nndct_shared.utils import option_util
from nndct_shared.utils import NndctScreenLogger
from pytorch_nndct import parse
from pytorch_nndct.nn.qat.modules import conv_fused
from pytorch_nndct.nn.qat.modules import quantizer as quantizer_mod
from pytorch_nndct.nn.qat.modules.quantizer import TQTQuantizer
from pytorch_nndct.parse.utils import get_short_name
from pytorch_nndct.qproc import base as qproc
from pytorch_nndct.quantization import module_transform
from pytorch_nndct.quantization.module_transform import ModuleTransformer
from pytorch_nndct.utils import module_util as mod_util

# TODO(yuwang): Move to utils to reduce repeatness.
class InputSpec(object):

  def __init__(self, shape, dtype):
    self.shape = shape
    self.dtype = dtype

def dummy_inputs(input_specs):
  inputs = []
  for spec in input_specs:
    inputs.append(torch.rand(1, *spec.shape).type(spec.dtype))
  return inputs

class TopoNode(object):

  def __init__(self, name, qconfig=None, module=None, inputs=None, op=None):
    self.name = name
    self.qconfig = qconfig
    self.module = module
    self.inputs = inputs
    self.op = op

  def __str__(self):
    return '{}({}) <- {}'.format(
        self.name,
        self.module._get_name() if self.module else None,
        ', '.join([str(inp) for inp in self.inputs]))

class ModelTopology(object):

  def __init__(self):
    self.nodes = []
    self._node_by_name = {}

    self.inputs = []
    self.outputs = []

  def add_node(self, node):
    self._node_by_name[node.name] = len(self.nodes)
    self.nodes.append(node)

  def node(self, name):
    return self.nodes[self._node_by_name[name]]

  def __str__(self):
    strs = []
    for node in self.nodes:
      strs.append(str(node))
    return '\n'.join(strs)

class QConfig(object):

  def __init__(self, input=None, output=None, weight=None, bias=None):
    # Currently only single input/output supported.
    # If xir and TorchQuantizer supports multiple inputs/outputs in the future,
    # then we can also expand this QuantConfig to support it.
    self.input = input
    self.output = output
    self.weight = weight
    self.bias = bias

  def __repr__(self):
    return 'QConfig(input={}, output={}, weight={}, bias={})'.format(
        self.input, self.output, self.weight, self.bias)


def _topo_node_name(node):
  module_name = mod_util.module_name_from_node(node)
  node_name = node if isinstance(node, str) else node.name
  # Use node name for non-module node so that
  # we can have a complete topology.
  return module_name if module_name else node_name

def quantize_input(module, index, quantizer):
  """Insert a quantizer for quantizing the input of the module.

    The input module is modified inplace with added quantizer module
    and forward_pre_hooks.

    Args:
      module: Input module that we want to quantize.
      index: The index of module's inputs to be quantized.
      quantizer: Module of quantizer to be added.
    """

  quantizer_name_template = 'input%d_quantizer'

  def _quantizer_forward_pre_hook(self, input):
    """Forward hook that calls quantizer on the input"""
    quantized_input = []
    for i, inp in enumerate(input):
      quantizer_name = quantizer_name_template % i
      if hasattr(self, quantizer_name):
        quantized_input.append(getattr(self, quantizer_name)(inp))
      else:
        quantized_input.append(inp)
    return tuple(quantized_input)

  #TODO(yuwang): Support torch.nn.Sequential
  module.add_module(quantizer_name_template % index, quantizer)
  # Register quantizer as the last entry in the hook list.
  # All forward pre hooks are preserved and will be executed before the quantizer.
  quantizer_flag_name = '_input_quantizer_hook_registered'
  if not hasattr(module, quantizer_flag_name):
    setattr(module, quantizer_flag_name, True)
    handle = module.register_forward_pre_hook(_quantizer_forward_pre_hook)
    module._forward_pre_hooks.move_to_end(handle.id, last=True)

# TODO(yuwang): Maybe support multiple outputs like quantize_input ?
def quantize_output(module, quantizer):
  """Insert a quantizer for quantizing the output of the module.

    The input module is modified inplace with added quantizer module
    and forward_hooks

    Args:
      module: Input module that we want to quantize.
      quantizer: Module of quantizer to be added.
    """

  def _quantizer_forward_hook(self, input, output):
    """Forward hook that calls quantizer on the output"""
    return self.output_quantizer(output)

  #TODO(yuwang): Support torch.nn.Sequential
  module.add_module('output_quantizer', quantizer)
  # Register quantizer as the first entry in the hook list.
  # All post forward hooks are preserved and will be executed after the quantizer.
  handle = module.register_forward_hook(_quantizer_forward_hook)
  module._forward_hooks.move_to_end(handle.id, last=False)

def insert_quantizer(model, node_to_qconfig):
  for node, qconfig in node_to_qconfig.items():
    if not qconfig.input and not qconfig.output:
      continue

    module_name = mod_util.module_name_from_node(node)
    if not module_name:
      raise ValueError(('Can not find module for node "{}"'
          'Only module object can be quantized,'
          'please re-implement this operation as a module.').format(node))

    module = mod_util.get_module(model, module_name)
    if qconfig.input:
      # Reserved support for multiple inputs, currently will always be 0.
      quantize_input(module, 0, qconfig.input)

    if qconfig.output:
      quantize_output(module, qconfig.output)

def _quant_info_key(group_name, key):
  return '.'.join([group_name, key])

def enable_quant(model):
  model.apply(quantizer_mod.enable_quant)

def disable_quant(model):
  model.apply(quantizer_mod.disable_quant)

def enable_warmup(model):
  model.apply(quantizer_mod.enable_warmup)

def disable_warmup(model):
  model.apply(quantizer_mod.disable_warmup)

def freeze_bn(model):
  model.apply(conv_fused.freeze_bn)

def dump_xmodel(output_dir='quantize_result', deploy_check=False):
  qproc.dump_xmodel(output_dir, deploy_check)

class QatScheduler(object):

  #def __init__(self, model, input_specs):
  def __init__(self, model, input_args, base_bit=8, mix_bit=False):
    self._model = model
    self._qinfo_to_quantizer = None

    # Original module name to transformed module name.
    # We can use it to convert the transformed model's state_dict keys
    # so that the original float model can load it.
    self._module_map = None
    '''
    if not isinstance(input_specs, (tuple, list)):
      input_specs = [input_specs]
    self._input_specs = input_specs
    '''
    self._input_args = input_args

    # turn off options optimization for following quantization
    option_util.set_option_value("nndct_quant_opt", 0)
    option_util.set_option_value("nndct_param_corr", False)
    option_util.set_option_value("nndct_equalization", False)

    #inputs = dummy_inputs(self._input_specs)
    inputs = self._input_args

    parser = parse.TorchParser()
    #graph = parser(self._model._get_name(), self._model, *inputs)
    graph = parser(self._model._get_name(), self._model, inputs)

    qprocessor = qproc.TorchQuantProcessor(
        'calib',
        self._model,
        inputs,
        mix_bit=mix_bit,
        device=torch.device('cpu'))
    quantizer = qprocessor.quantizer

    # Use hard-coded value to fill in fp_pos and export quant config,
    # so that we can initialize a new TorchQuantProcessor in 'test' mode later.
    for _, group in quantizer.quant_config.items():
      for key in group:
        group[key][-1] = 4
    quantizer.export_quant_config()

    # Use quantizer's graph to build param_to_node as the quant_info is
    # generated from the quantizer's graph.
    # For example, the param 'ResNet::conv.bias' only exist in the quantizer's
    # graph because it comes from the fused conv + bias.
    param_to_node = {}
    for node in quantizer.Nndctgraph.nodes:
      for name, tensor in node.op.params.items():
        param_to_node[tensor.name] = node.name

    # Create quantizer modules and build qconfig for each node.
    node_to_qconfig = {}
    qinfo_to_quantizer = {}

    def get_num_bits(quant_info):
      return quant_info[0] if quant_info[0] == 8 else base_bit

    group_name = 'param'
    group = quantizer.quant_config[group_name]
    for param_name, info in group.items():
      # layer1.0.conv1.weight
      state_dict_key = get_short_name(param_name)
      node_name = param_to_node[param_name]
      qconfig = node_to_qconfig.get(node_name, QConfig())
      attr_name = state_dict_key.split('.')[-1]
      tqt_quantizer = TQTQuantizer(get_num_bits(info), tensor_type='param')
      setattr(qconfig, attr_name, tqt_quantizer)
      node_to_qconfig[node_name] = qconfig
      qinfo_to_quantizer[_quant_info_key(group_name, param_name)] = tqt_quantizer

    for group_name in ['input', 'output']:
      group = quantizer.quant_config[group_name]
      for node_name, info in group.items():
        qconfig = node_to_qconfig.get(node_name, QConfig())
        tqt_quantizer = TQTQuantizer(get_num_bits(info), tensor_type='blob')
        setattr(qconfig, group_name, tqt_quantizer)
        node_to_qconfig[node_name] = qconfig
        qinfo_to_quantizer[_quant_info_key(group_name, node_name)] = tqt_quantizer
    self._qinfo_to_quantizer = qinfo_to_quantizer

    model_topo = ModelTopology()
    for node in graph.nodes:
      name = _topo_node_name(node)
      inputs = []
      for input_name in node.in_nodes:
        inputs.append(_topo_node_name(input_name))
      qconfig = node_to_qconfig.get(node.name, None)
      model_topo.add_node(TopoNode(name, qconfig, None, inputs, node.op))

    # TODO(yuwang): Output all transformed modules.
    transforms = [
        module_transform.FuseAndQuantizeConv2dBatchNorm(),
        module_transform.QuantizeLinear(),
        module_transform.ReplaceAdaptiveAvgPool2d(),
    ]

    transformer = ModuleTransformer(self._model, model_topo, transforms)
    model, self._module_map = transformer.transform()

    model.enable_quant = types.MethodType(enable_quant, model)
    model.disable_quant = types.MethodType(disable_quant, model)
    model.enable_warmup = types.MethodType(enable_warmup, model)
    model.disable_warmup = types.MethodType(disable_warmup, model)
    model.freeze_bn = types.MethodType(freeze_bn, model)
    insert_quantizer(model, node_to_qconfig)

    quantizer.quant_model = model
    self.quant_quantizer = quantizer

  def quant_model(self):
    return self.quant_quantizer.quant_model

  def _fill_in_quant_info(self, quantizer, qinfo_to_quantizer):

    def update_quant_info(orig_quant_info, quant_info):
      """Use quant_info to update orig_quant_info."""
      for index in range(len(orig_quant_info)):
        orig_quant_info[index] = quant_info[index]

    quant_info = quantizer.quant_config
    for name, group in quant_info.items():
      for key in group:
        tqt_quantizer = qinfo_to_quantizer[_quant_info_key(name, key)]
        update_quant_info(group[key], tqt_quantizer.quant_info())

  def convert_to_deployable(self, trained_model, mix_bit=False):
    if not self._qinfo_to_quantizer or not self._module_map:
      raise RuntimeError('Must call "trainable_model" first.')

    # Copy trained parameters from transformed model to original float model.
    orig_state_dict = self._model.state_dict()
    trained_state_dict = trained_model.state_dict()
    state_dict = {}
    for key in orig_state_dict.keys():
      module_name, weight_name, = key.rsplit('.', 1)
      if module_name in self._module_map:
        trained_module_name = self._module_map[module_name]
        trained_key = '.'.join([trained_module_name, weight_name])
      else:
        trained_key = key
      state_dict[key] = trained_state_dict[trained_key]
    model = copy.deepcopy(self._model)
    model.load_state_dict(state_dict)
    model.eval()
    '''
    inputs = dummy_inputs(self._input_specs)
    qprocessor = qproc.TorchQuantProcessor(
        'test',
        model,
        [inp.cuda() for inp in inputs],
        mix_bit=mix_bit,
        device=torch.device('cuda'))
    '''
    inputs = self._input_args
    qprocessor = qproc.TorchQuantProcessor(
        'test',
        model,
        inputs,
        mix_bit=mix_bit,
        device=torch.device('cuda'))

    quantizer = qprocessor.quantizer
    self._fill_in_quant_info(quantizer, self._qinfo_to_quantizer)
    quantizer.export_quant_config()

    quant_model = quantizer.quant_model
    quant_model.dump_xmodel = dump_xmodel
    self.deploy_quantizer = quantizer
    GLOBAL_MAP.set_map(NNDCT_KEYS.QUANTIZER, quantizer)
    NndctScreenLogger().info(f"=>Deployable model is generated.")

  def deploy_model(self):
    return self.deploy_quantizer.quant_model

  def finetune(self, run_fn, run_args):
    NndctScreenLogger().warning(f"Quant aware training process has not finetune function.")

  # full procedures mode including finetune, calibration and test accuracy
  def quantize(self, un_fn, run_args):
    NndctScreenLogger().warning(f"Quant aware training process has not quantize function.")

  # export quantization steps information for tensors to be quantized
  def export_quant_config(self):
    self.deploy_quantizer.export_quant_config()

  # export xmodel for compilation
  def export_xmodel(self, output_dir, deploy_check):
    self.dump_xmodel(output_dir, deploy_check)

  # dump xmodel for compilation
  def dump_xmodel(self, output_dir = "quantize_result", deploy_check=False):
    qproc.dump_xmodel(output_dir, deploy_check)
