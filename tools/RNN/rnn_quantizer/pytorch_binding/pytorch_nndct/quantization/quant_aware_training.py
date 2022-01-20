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

import copy
import os
import torch
import types

from nndct_shared.base import NNDCT_OP as OpTypes
from nndct_shared.utils import io as io_util
from nndct_shared.utils import option_util

from pytorch_nndct import parse
from pytorch_nndct import utils as py_utils
from pytorch_nndct.nn.modules import functional
from pytorch_nndct.nn.qat.modules import conv_fused
from pytorch_nndct.nn.qat.modules import quantizer as quantizer_mod
from pytorch_nndct.nn.qat.modules.quantizer import TQTQuantizer
from pytorch_nndct.parse.parse_utils import get_short_name
from pytorch_nndct.qproc import base as qproc
from pytorch_nndct.qproc.ModuleHooker import ModuleHooker
from pytorch_nndct.quantization import module_transform
from pytorch_nndct.quantization.module_transform import ModuleTransformer
from pytorch_nndct.utils import logging
from pytorch_nndct.utils import module_util as mod_util

_QUANT_INFO_FILE_NAME = 'quant_info.json'
_DEPLOYABLE_MODEL_NAME = 'deployable.pth'

class TopoNode(object):

  def __init__(self,
               node,
               name,
               qconfig=None,
               module=None,
               inputs=None,
               op=None):
    self.graph_node = node
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

  def _forward_pre_hook(self, input):
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
    handle = module.register_forward_pre_hook(_forward_pre_hook)
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

  def _forward_hook(self, input, output):
    """Forward hook that calls quantizer on the output"""
    return self.output_quantizer(output)

  def _forward_hook_max(self, input, output):
    """Forward hook that calls quantizer on the output for functional.Max."""
    quantized_values = self.output_quantizer(output[0])
    # (values, indices)
    return (quantized_values, output[1])

  quantizer_flag_name = '_output_quantizer_hook_registered'
  if hasattr(module, quantizer_flag_name):
    raise RuntimeError('Insert multiple quantizers to a module: {}'.format(
        type(module)))

  #TODO(yuwang): Support torch.nn.Sequential
  module.add_module('output_quantizer', quantizer)
  setattr(module, quantizer_flag_name, True)
  # Register quantizer as the first entry in the hook list.
  # All post forward hooks are preserved and will be executed after the quantizer.
  if isinstance(module, functional.Max):
    handle = module.register_forward_hook(_forward_hook_max)
  else:
    handle = module.register_forward_hook(_forward_hook)
  module._forward_hooks.move_to_end(handle.id, last=False)

def enable_quant(model):
  model.apply(quantizer_mod.enable_quant)
  logging.info('Enable quantization: quantized operations will be performed.')

def disable_quant(model):
  model.apply(quantizer_mod.disable_quant)
  logging.info(
      'Disable quantization: floating point operations will be performed.')

def enable_warmup(model):
  model.apply(quantizer_mod.enable_warmup)
  logging.info('Initialize quantizer.')

def disable_warmup(model):
  model.apply(quantizer_mod.disable_warmup)

def freeze_quant(model):
  model.apply(quantizer_mod.freeze_quant)
  logging.info('Scale of quantizer has been frozen.')

def freeze_bn_stats(model):
  model.apply(conv_fused.freeze_bn_stats)
  logging.info('Running statistics of batch normlization has been frozen.')

def fuse_conv_bn(model):
  model.apply(conv_fused.fuse_conv_bn)
  model.conv_bn_fused = True
  logging.info('Merge batchnorm to conv.')

def quantizer_parameters(model):
  return [
      param for name, param in model.named_parameters()
      if 'log_threshold' in name
  ]

def non_quantizer_parameters(model):
  return [
      param for name, param in model.named_parameters()
      if 'log_threshold' not in name
  ]

class QatProcessor(object):

  def __init__(self,
               model,
               inputs,
               bitwidth,
               mix_bit=False,
               device=torch.device("cuda")):

    if isinstance(model, torch.nn.DataParallel):
      raise ValueError('DataParallel object is not allowed.')

    # turn off options optimization for following quantization
    option_util.set_option_value("nndct_quant_opt", 0)
    option_util.set_option_value("nndct_param_corr", False)
    option_util.set_option_value("nndct_equalization", False)

    self._model = model
    self._inputs = inputs
    self._bitwidth = bitwidth
    self._mix_bit = mix_bit
    self._device = device

    # Original module name to transformed module name.
    # We can use it to convert the transformed model's state_dict keys
    # so that the original float model can load it.
    self._module_map = None

    self._trainable_model = None
    self._tmp_qat_dir = '.qat'

    qprocessor = qproc.TorchQuantProcessor(
        'calib',
        model,
        inputs,
        output_dir=self._tmp_qat_dir,
        bitwidth_w=self._bitwidth,
        bitwidth_a=self._bitwidth,
        mix_bit=mix_bit,
        device=device)
    quantizer = qprocessor.quantizer

    # Use hard-coded value to fill in fp_pos and export quant config,
    # so that we can initialize a new TorchQuantProcessor in 'test' mode later.
    quant_config = quantizer.quant_config
    for _, group in quant_config.items():
      for key in group:
        group[key][-1] = 4
    quantizer.export_quant_config(adjust_pos=False)

    # Use quantizer's graph to build param_to_node as the quant_info is
    # generated from the quantizer's graph.
    # For example, the param 'ResNet::conv.bias' only exist in the quantizer's
    # graph because it comes from the fused conv + bias.
    self._tensor_to_node = {}
    graph = quantizer.Nndctgraph
    for node in graph.nodes:
      for name, tensor in node.op.params.items():
        self._tensor_to_node[tensor.name] = (node.name, name)

    parser = parse.TorchParser()
    self._graph = parser(self._model._get_name(), self._model, self._inputs)
    self._torch_quantizer = quantizer

    def get_bitwidth(quant_info):
      return quant_info[0] if quant_info[0] == 8 else bitwidth

    # Create quantizer for each item in quant config.
    self._node_to_qconfig = {}
    self._quant_config = copy.deepcopy(quant_config)
    for name, group in self._quant_config.items():
      for key, qinfo in group.items():
        if name == 'param':
          node, param = self._tensor_to_node[key]
          attr = ModuleHooker._parameter_map[param]
          tensor_type = 'weight'
        else:
          node, attr = key, name
          tensor_type = 'act'

        tqt_quantizer = TQTQuantizer(get_bitwidth(qinfo), tensor_type)
        qconfig = self._node_to_qconfig.get(node, QConfig())
        mod_util.setattr_if_has(qconfig, attr, tqt_quantizer)
        self._node_to_qconfig[node] = qconfig
        self._quant_config[name][key] = (node, attr)
        logging.vlog(2, '[{}][{}] = ({}, {})'.format(name, key, node, attr))

  def trainable_model(self, calib_dir='', allow_reused_module=False):
    if calib_dir:
      self._torch_quantizer.init_quant_config(
          os.path.join(calib_dir, _QUANT_INFO_FILE_NAME))
      self._import_quant_info(self._torch_quantizer)

    self._pre_check(allow_reused_module)

    model_topo = self._build_model_topo()

    model, model_topo, self._module_map = self._transform_module(
        self._model, model_topo)

    self._insert_quantizer(model_topo, allow_reused_module)

    model.enable_quant = types.MethodType(enable_quant, model)
    model.disable_quant = types.MethodType(disable_quant, model)
    model.enable_warmup = types.MethodType(enable_warmup, model)
    model.disable_warmup = types.MethodType(disable_warmup, model)
    model.freeze_quant = types.MethodType(freeze_quant, model)
    model.freeze_bn_stats = types.MethodType(freeze_bn_stats, model)
    model.fuse_conv_bn = types.MethodType(fuse_conv_bn, model)
    model.quantizer_parameters = types.MethodType(quantizer_parameters, model)
    model.non_quantizer_parameters = types.MethodType(non_quantizer_parameters,
                                                      model)

    if self._device is not None:
      model = model.to(self._device)
    return model

  def _pre_check(self, allow_reused_module):
    # If two or more nodes point to a same module, then we will let them
    # use the same qconfig.
    module_to_qconfig = {}
    for node in self._graph.nodes:
      module_name = mod_util.module_name_from_node(node)
      if not module_name or node.name not in self._node_to_qconfig:
        continue

      if module_name in module_to_qconfig:
        if allow_reused_module:
          self._node_to_qconfig[node.name] = module_to_qconfig[module_name]
          logging.warn(
              ('Reused module ({}) may lead to poor result of QAT, '
               'make sure this is what you expect.').format(module_name))
        else:
          raise ValueError(
              ('Quantized module "{}" has been called multiple '
               'times in forward pass. If you want to share quantized '
               'parameters in multiple calls, call trainable_model with '
               '"allow_reused_module=True"').format(module_name))
      module_to_qconfig[module_name] = self._node_to_qconfig[node.name]

    # Make sure all quantizable operations are instance of torch.nn.Module.
    replacement_map = {
        OpTypes.ADD: ('torch.add/+', functional.Add),
        OpTypes.CONCAT: ('torch.cat', functional.Cat),
        OpTypes.MAX: ('torch.max', functional.Max),
        OpTypes.PAD: ('torch.nn.functional.pad', functional.Pad),
        OpTypes.RELU: ('torch.nn.functional.relu', torch.nn.ReLU),
        OpTypes.SUM: ('torch.sum', functional.Sum),
    }

    for name, group in self._quant_config.items():
      for key in group:
        node_name, _ = self._quant_config[name][key]

        module = mod_util.get_module_by_node(self._model, node_name)
        node = self._graph.node(node_name)

        module_cls = type(module) if module else None
        if node.op.type in replacement_map:
          op, target_cls = replacement_map[node.op.type]
          if module_cls != target_cls:
            raise ValueError(
                ('Quantized operation({}) must be instance '
                 'of "torch.nn.Module", please replace {} with {}').format(
                     node.name, op, target_cls))

        # A quantized op must be implemented as a module.
        if not module:
          if node.op.type == OpTypes.INPUT:
            raise ValueError(
                ('Found float input, make sure your forward pass is wrapped'
                'with with QuantStub/DeQuantStub pair')
            )
          else:
            raise ValueError(
                ('Can not quantize node "{}({})" as it is not a '
                 'torch.nn.Module object, please re-implement this operation '
                 'as a module.').format(node.name, node.op.type))

        torch_op_type = py_utils.get_torch_op_type(node.op.type)
        torch_op_attr = py_utils.get_torch_op_attr(torch_op_type)
        if not torch_op_attr.op_name.startswith('torch'):
          logging.vlog(1,
                       'Non-torch op found: {}'.format(torch_op_attr.op_name))
          continue

        # Check if we get the correct module.
        op_type_name = torch_op_attr.op_name.split('.')[-1]
        logging.vlog(
            1, '{}({}): {} vs. {}'.format(node.name, node.op.type,
                                          module_cls.__name__,
                                          torch_op_attr.op_name))
        if not module_cls.__module__.startswith(
            'pytorch_nndct') and module_cls.__name__ != op_type_name:
          raise ValueError(('{} is a quantized operation, please re-implement '
                            'your op as a nn.Module (Node: {})').format(
                                torch_op_attr.op_name, node_name))

  def _build_model_topo(self):
    model_topo = ModelTopology()
    for node in self._graph.nodes:
      name = _topo_node_name(node)
      inputs = []
      for input_name in node.in_nodes:
        inputs.append(_topo_node_name(input_name))
      qconfig = self._node_to_qconfig.get(node.name, None)
      model_topo.add_node(TopoNode(node, name, qconfig, None, inputs, node.op))
    return model_topo

  def _transform_module(self, model, model_topo):
    bn_cls = (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
    for module in model.modules():
      if isinstance(module, bn_cls) and type(module) not in bn_cls:
        bn_cls_names = ['torch.nn.' + cls.__name__ for cls in bn_cls]
        raise RuntimeError('BatchNorm class must be: {}, but got {}'.format(
            bn_cls_names, type(module)))

    transforms = [
        module_transform.FuseAndQuantizeConv2dBatchNorm(),
        module_transform.FuseAndQuantizeConv3dBatchNorm(),
        module_transform.QuantizeConvNd(),
        module_transform.QuantizeLinear(),
        module_transform.ReplacePooling2d(),
        module_transform.ReplaceLeakyReLU(),
    ]
    transformer = ModuleTransformer(model, model_topo, transforms)
    return transformer.transform()

  def _insert_quantizer(self, model_topo, allow_reused_module):
    """Insert quantizer for quantizing input/output of a module.
      The quantization of weight/bias is handled by quantized module itself.
    """

    quantized_modules = set()
    for node in model_topo.nodes:
      qconfig = node.qconfig
      if not qconfig:
        continue

      # Check if there are parameterized modules that have not been
      # transformed to the corresponding quantized version.
      if qconfig.weight or qconfig.bias:
        if not hasattr(node.module, 'is_quantized'):
          raise NotImplementedError(
              ('The quantization of {} not implemented '
               'yet. (Node name: {})').format(type(node.module), node.name))

      if node.name in quantized_modules:
        continue

      logging.vlog(
          3,
          'Inserting quantizer for node {}: {}'.format(node.graph_node.name,
                                                       qconfig))
      quantized_modules.add(node.name)
      if qconfig.input:
        # Reserved support for multiple inputs, currently will always be 0.
        quantize_input(node.module, 0, qconfig.input)

      if qconfig.output:
        quantize_output(node.module, qconfig.output)

  def _quantizer_by_quant_config(self, tensor_type, key):
    node, attr = self._quant_config[tensor_type][key]
    qconfig = self._node_to_qconfig[node]
    return getattr(qconfig, attr)

  def _fill_in_quant_config(self, quantizer):
    for tensor_type, group in quantizer.quant_config.items():
      for key in group:
        tqt_quantizer = self._quantizer_by_quant_config(tensor_type, key)
        if tqt_quantizer.warmup_enabled[0] == 1:
          raise RuntimeError(
              'Attempt to export quant info of a untrained quantizer: {}'
              .format(key))
        group[key] = tqt_quantizer.export_quant_info()

  def _import_quant_info(self, quantizer):
    for tensor_type, group in quantizer.quant_config.items():
      for key in group:
        tqt_quantizer = self._quantizer_by_quant_config(tensor_type, key)
        tqt_quantizer.import_quant_info(group[key])

  def to_deployable(self, trained_model, output_dir):
    model = self._to_deployable(trained_model, output_dir)
    torch.save(model.state_dict(),
               os.path.join(output_dir, _DEPLOYABLE_MODEL_NAME))
    return self._qprocessor.quant_model()

  def convert_to_deployable(self, trained_model, output_dir):
    logging.warn((
        '"convert_to_deployable" is deprecated and will be removed in the future. '
        'Use "to_deployable" instead.'))
    return self._to_deployable(trained_model, output_dir)

  def _to_deployable(self, trained_model, output_dir):
    if not self._quant_config or self._module_map is None:
      raise RuntimeError('Must call "trainable_model" first.')

    if hasattr(trained_model, 'conv_bn_fused') and getattr(
        trained_model, 'conv_bn_fused'):
      raise RuntimeError(
          'Not allowed to convert a fused model to a deployable model.')

    # Copy trained parameters from transformed model to original float model.
    orig_state_dict = self._model.state_dict()
    trained_state_dict = trained_model.state_dict()
    state_dict = {}
    for key in orig_state_dict:
      if '.' in key:
        module_name, weight_name = key.rsplit('.', 1)
      else:
        # Such as 'global_step'.
        module_name, weight_name = None, key
      if module_name in self._module_map:
        # Currently only for bn.
        # conv1.0.0.bn.weight -> conv1.0.1.weight
        trained_module_name = self._module_map[module_name]
        trained_key = '.'.join([trained_module_name, weight_name])
      else:
        trained_key = key
      state_dict[key] = trained_state_dict[trained_key]
      logging.vlog(3, 'state dict of {} is from {}'.format(key, trained_key))
    model = copy.deepcopy(self._model)
    model.load_state_dict(state_dict)
    model.eval()

    qprocessor = qproc.TorchQuantProcessor(
        'test',
        model,
        self._inputs,
        output_dir=self._tmp_qat_dir,
        bitwidth_w=self._bitwidth,
        bitwidth_a=self._bitwidth,
        mix_bit=self._mix_bit,
        device=self._device)

    quantizer = qprocessor.quantizer
    self._fill_in_quant_config(quantizer)

    sub_dir = os.path.join(output_dir, 'test')
    io_util.create_work_dir(sub_dir)
    # Must set adjust_pos=False first, because quantizer will modify its
    # quant info inplace when adjust_pos=True.
    # Export original (not adjusted yet) quant info for testing deployable
    # model and the accuracy should be the same with the trainable model.
    quantizer.export_quant_config(
        os.path.join(sub_dir, _QUANT_INFO_FILE_NAME), adjust_pos=False)
    quantizer.export_quant_config(
        os.path.join(output_dir, _QUANT_INFO_FILE_NAME), adjust_pos=True)

    self._qprocessor = qprocessor
    return model

  def quant_model(self):
    if not self._trainable_model:
      self._trainable_model = self.trainable_model()
    return self._trainable_model

  def deployable_model(self, src_dir, used_for_xmodel=False):
    if used_for_xmodel:
      device = torch.device('cpu')
      inputs = self._inputs.to(device)
    else:
      device = self._device
      inputs = self._inputs

    model = copy.deepcopy(self._model)
    model.load_state_dict(
        torch.load(os.path.join(src_dir, _DEPLOYABLE_MODEL_NAME)))
    qprocessor = qproc.TorchQuantProcessor(
        'test',
        model,
        inputs,
        output_dir=self._tmp_qat_dir,
        bitwidth_w=self._bitwidth,
        bitwidth_a=self._bitwidth,
        mix_bit=self._mix_bit,
        device=device)
    self._qprocessor = qprocessor
    if used_for_xmodel:
      logging.info(
          'Forward the deployable model with data of batch_size=1 in cpu mode to dump xmodel.'
      )
    return qprocessor.quant_model()

  def finetune(self, run_fn, run_args):
    raise NotImplementedError(
        'Quant aware training process has not finetune function.')

  # full procedures mode including finetune, calibration and test accuracy
  def quantize(self, un_fn, run_args):
    raise NotImplementedError(
        'Quant aware training process has not quantize function.')

  # export xmodel for compilation
  def export_xmodel(self, output_dir, deploy_check=False):
    if not hasattr(self, '_qprocessor'):
      raise RuntimeError('Must call "deployable_model" first.')

    if next(self._qprocessor.quant_model().parameters()).device != torch.device(
        'cpu'):
      raise ValueError((
          'Xmodel can only be exported in cpu mode,'
          'use deployable_model(src_dir, used_for_xmodel=True) to get a cpu model.'
      ))

    self._qprocessor.export_xmodel(output_dir, deploy_check)
