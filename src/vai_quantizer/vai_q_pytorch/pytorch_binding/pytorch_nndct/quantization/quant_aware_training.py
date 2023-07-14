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
from nndct_shared.optimization.optimizer import QuantOptimizer
from nndct_shared.utils import NndctOption
from nndct_shared.utils import NndctScreenLogger
from nndct_shared.utils import io as io_util
from nndct_shared.utils import option_util
from nndct_shared.utils.msg_code import QError
from nndct_shared.utils.msg_code import QWarning

from pytorch_nndct import parse
from pytorch_nndct import utils as py_utils
from pytorch_nndct.nn.modules import functional
from pytorch_nndct.nn.quantization.modules import conv_fused
from pytorch_nndct.nn.quantization.modules import tqt as tqt_mod
from pytorch_nndct.qproc import base as qproc
from pytorch_nndct.qproc.ModuleHooker import ModuleHooker
from pytorch_nndct.quantization import model_topo as model_topo_mod
from pytorch_nndct.quantization import module_transform
from pytorch_nndct.quantization import config as config_mod
from pytorch_nndct.quantization import transforms as transforms_mod
from pytorch_nndct.utils import logging
from pytorch_nndct.utils import module_util as mod_util

_QUANT_INFO_FILE_NAME = 'quant_info.json'
_DEPLOYABLE_MODEL_NAME = 'deployable.pth'

class TensorTypes(object):
  PARAM = 'param'
  INPUT = 'input'
  OUTPUT = 'output'

_valid_tensor_types = [TensorTypes.PARAM, TensorTypes.INPUT, TensorTypes.OUTPUT]

class QatConfigItem(object):

  def __init__(self,
               tensor_type: str,
               key: str,
               index: int,
               weight_name=None,
               node=None,
               quantizer=None):
    self.tensor_type = tensor_type
    self.key = key
    self.index = index
    self.weight_name = weight_name
    self.node = node
    self.quantizer = quantizer

  def __str__(self):
    node_name = self.node.name if self.node is not None else None
    return (f'{self.__class__.__name__}(tensor_type={self.tensor_type}, '
            f'key={self.key}, index={self.index}, node={node_name}, '
            f'quantizer={self.quantizer})')

class QatConfig(object):

  def __init__(self, quantizer):
    self._items = {}

    for item in quantizer.quant_config:
      if item not in _valid_tensor_types:
        continue
      group = quantizer.quant_config[item]
      for key in group:
        for index in range(quantizer.get_quant_len(key, item)):
          self._items[self._config_key(item, key, index)] = QatConfigItem(
              item, key, index)

  def _config_key(self, tensor_type, key, index):
    return f'{tensor_type}_{key}_{index}'

  def get(self, tensor_type, key, index):
    return self._items[self._config_key(tensor_type, key, index)]

  @property
  def values(self):
    return self._items.values()

def logging_warn(code, message):
  NndctScreenLogger().warning2user(code, message)

def logging_error(code, message):
  NndctScreenLogger().check2user(code, message, False)

def enable_quant(model):
  model.apply(tqt_mod.enable_quant)
  logging.info('Enable quantization: quantized operations will be performed.')

def disable_quant(model):
  model.apply(tqt_mod.disable_quant)
  logging.info(
      'Disable quantization: floating point operations will be performed.')

def enable_warmup(model):
  model.apply(tqt_mod.enable_warmup)
  logging.info('Initialize quantizer.')

def disable_warmup(model):
  model.apply(tqt_mod.disable_warmup)

def freeze_quant(model):
  model.apply(tqt_mod.freeze_quant)
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

  def __init__(self, model, inputs, bitwidth=8, mix_bit=False, device=None):

    # turn off options optimization for following quantization
    option_util.set_option_value('nndct_quant_opt', 0)
    option_util.set_option_value('nndct_param_corr', False)
    option_util.set_option_value('nndct_equalization', False)

    if isinstance(model, torch.nn.DataParallel):
      logging_error(QError.DATA_PARALLEL_NOT_ALLOWED,
                    'torch.nn.DataParallel object is not allowed.')

    self._model = model
    self._inputs = tuple(inputs) if isinstance(inputs,
                                               (tuple, list)) else (inputs,)
    self._bitwidth = bitwidth
    self._mix_bit = mix_bit

    if device is not None:
      logging_warn(
          QWarning.DEPRECATED_ARGUMENT,
          ('The argument "device" is no longer used and will be ignored.'))
    self._device = next(model.parameters()).device

    # Original module name to transformed module name.
    # We can use it to convert the transformed model's state_dict keys
    # so that the original float model can load it.
    self._module_map = None

    self._trainable_model = None
    self._tmp_qat_dir = '.vai_qat'

    qprocessor = qproc.TorchQuantProcessor(
        'calib',
        model,
        inputs,
        output_dir=self._tmp_qat_dir,
        bitwidth_w=self._bitwidth,
        bitwidth_a=self._bitwidth,
        mix_bit=mix_bit,
        device=self._device)
    quantizer = qprocessor.quantizer
    self._torch_quantizer = quantizer

    # Have to export quant config first so that we can create a new
    # TorchQuantProcessor in 'test' mode later.
    quantizer.export_quant_config(adjust_pos=False, inference_check=False)

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

    quant_optimizer = QuantOptimizer()
    if NndctOption.nndct_partition_mode.value > 0:
      quant_optimizer._tag_quant_nodes_v2(self._raph)
    else:
      quant_optimizer._tag_quant_nodes(self._graph)

    self._qat_config = QatConfig(quantizer)
    self._node_to_spec = self._create_quantizers(self._qat_config)

  def _create_quantizers(self, qat_config):
    """Create quantizer for each item in quant config."""

    def input_index_to_node(node, tensor_name):
      for index, tensor in enumerate(node.in_tensors):
        if tensor_name == tensor.name:
          return index
      return None

    def select_rounding_mode(node, tensor_type):
      rounding_mode = 3 if tensor_type == TensorTypes.PARAM else 2
      if tensor_type != TensorTypes.PARAM and \
          NndctOption.nndct_ip_v70_bert_qat.value and \
          node.op.type in [OpTypes.LAYER_NORM, OpTypes.SOFTMAX]:
        rounding_mode = 4
      return rounding_mode

    node_to_spec = {}
    for config in self._qat_config.values:
      tensor_type = config.tensor_type
      if tensor_type == TensorTypes.PARAM:
        node_name, param = self._tensor_to_node[config.key]
        node = self._graph.node(node_name)
        if node.has_bound_params():
          weight_name = ModuleHooker._parameter_map[param]
        else:
          # If a tensor isn't bounded to a node, we treat it as input tensor.
          tensor_index = input_index_to_node(node, config.key)
          if not tensor_index:
            raise RuntimeError(
                f'Can not get input index: node={node.name}, tensor={config.key}'
            )
          tensor_type = TensorTypes.INPUT
      else:
        node_name, tensor_index = config.key, config.index

      # See TorchQuantizer::do_quantize() in quantization/torchquantizer.py
      node = self._graph.node(node_name)
      quantizer = tqt_mod.TQTQuantizer(self._bitwidth, tensor_type,
                                       select_rounding_mode(node, tensor_type))
      config.node, config.quantizer = node, quantizer

      spec = node_to_spec.get(node_name, config_mod.LayerRuntimeSpec())
      if tensor_type == TensorTypes.PARAM:
        spec.set_weight_quantizer(weight_name, quantizer)
        config.weight_name = weight_name
      elif tensor_type == TensorTypes.INPUT:
        spec.set_input_quantizer(tensor_index, quantizer)
      else:
        spec.set_output_quantizer(tensor_index, quantizer)
      node_to_spec[node_name] = spec
      logging.vlog(2, f'Create quantizer for config: {config}')
    return node_to_spec

  def trainable_model(self, calib_dir='', allow_reused_module=False):
    if calib_dir:
      self._torch_quantizer.load_quant_config(
          config=os.path.join(calib_dir, _QUANT_INFO_FILE_NAME))
      self._import_quant_info(self._torch_quantizer)

    self._check_reused_module(allow_reused_module)

    self._check_op_quantizable()

    model_topo = model_topo_mod.build_model_topo(self._graph,
                                                 self._node_to_spec)

    excluded_nodes = []
    for node in model_topo.nodes:
      if not node.in_quant_part:
        excluded_nodes.append(node.name)
    model, model_topo, self._module_map = self._transform_module(
        self._model, model_topo, excluded_nodes)

    self._check_module_quantized(model)

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

    return model.to(self._device)

  def _check_reused_module(self, allow_reused_module):
    # If two or more nodes point to a same module, then we will let them
    # use the same spec.
    module_to_spec = {}
    for node in self._graph.nodes:
      module_name = mod_util.module_name_from_node(node)
      if not module_name or node.name not in self._node_to_spec:
        continue

      if module_name in module_to_spec:
        if allow_reused_module:
          self._node_to_spec[node.name] = module_to_spec[module_name]
          logging_warn(
              QWarning.REUSED_MODULE,
              ('Reused module ({}) may lead to low accuracy of QAT, '
               'make sure this is what you expect.').format(module_name))
        else:
          logging_error(
              QError.REUSED_MODULE,
              ('Quantized module "{}" has been called multiple '
               'times in forward pass. If you want to share quantized '
               'parameters in multiple calls, call trainable_model with '
               '"allow_reused_module=True"').format(module_name))
      module_to_spec[module_name] = self._node_to_spec[node.name]

  def _check_op_quantizable(self):
    # Make sure all quantizable operations are instance of torch.nn.Module.
    replacement_map = {
        OpTypes.ADD: ('torch.add/+', functional.Add),
        OpTypes.CONCAT: ('torch.cat', functional.Cat),
        OpTypes.MAX: ('torch.max', functional.Max),
        OpTypes.PAD: ('torch.nn.functional.pad', functional.Pad),
        OpTypes.RELU: ('torch.nn.functional.relu', torch.nn.ReLU),
        OpTypes.SUM: ('torch.sum', functional.Sum),
        OpTypes.CLAMP: ('torch.clamp', functional.Clamp),
    }

    for config in self._qat_config.values:
      node = config.node
      module = mod_util.get_module_by_node(self._model, node.name)

      module_cls = type(module) if module else None
      if node.op.type in replacement_map:
        op, target_cls = replacement_map[node.op.type]
        if module_cls != target_cls:
          logging_error(QError.NOT_A_MODULE,
                        ('Quantized operation({}) must be instance '
                         'of "torch.nn.Module", please replace {} with {}.'
                         'The original source range is:\n{}').format(
                             node.name, op, target_cls, node.source_range))

      # A quantized op must be implemented as a module.
      if not module:
        if node.op.type == OpTypes.INPUT:
          logging_error(
              QError.INPUT_NOT_QUANTIZED,
              ('Input is not quantized. Please use QuantStub/DeQuantStub to '
               'define quantization scope.'))
        else:
          logging_error(
              QError.NOT_A_MODULE,
              ('Can not quantize node "{}({})" as it is not a '
               'torch.nn.Module object, please re-implement this operation '
               'as a module. The original source range:\n{}').format(
                   node.name, node.op.type, node.source_range))

      torch_op_type = py_utils.get_torch_op_type(node.op.type)
      torch_op_attr = py_utils.get_torch_op_attr(torch_op_type)
      if not torch_op_attr.op_name.startswith('torch'):
        logging.vlog(1, 'Non-torch op found: {}'.format(torch_op_attr.op_name))
        continue

      # Check if we get the correct module.
      op_type_name = torch_op_attr.op_name.split('.')[-1]
      logging.vlog(
          1, '{}({}): {} vs. {}'.format(node.name, node.op.type,
                                        module_cls.__name__,
                                        torch_op_attr.op_name))
      if not module_cls.__module__.startswith(
          'pytorch_nndct') and module_cls.__name__ != op_type_name:
        logging_warn(QWarning.OP_TYPE_MISMATCH,
                     ('Module class name and op type name do not match: '
                      '{} vs. {} (Node: {})'.format(module_cls.__name__,
                                                    op_type_name, node.name)))

  def _check_module_quantized(self, model):
    """Check that all parameterized modules are transformed to the
    corresponding quantized version."""

    for node, spec in self._node_to_spec.items():
      if spec.weight_quantizers:
        module = mod_util.get_module_by_node(model, node)
        if not hasattr(module, 'is_quantized'):
          logging_error(QError.UNSUPPORTED_OPS,
                        ('The quantization of {} not implemented '
                         'yet. (Node name: {})').format(type(module), node))

  def _transform_module(self, model, model_topo, excluded_nodes):
    bn_cls = (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
    for module in model.modules():
      if isinstance(module, bn_cls) and type(module) not in bn_cls:
        bn_cls_names = ['torch.nn.' + cls.__name__ for cls in bn_cls]
        logging_error(
            QError.NOT_A_MODULE,
            'BatchNorm class must be: {}, but got {}.'.format(
                bn_cls_names, type(module)))

    transforms = [
        transforms_mod.QuantizeConv2dBatchNorm(),
        transforms_mod.QuantizeConv3dBatchNorm(),
        transforms_mod.QuantizeConvNd(),
        transforms_mod.QuantizeLinear(),
        transforms_mod.ReplacePooling2d(),
        transforms_mod.ReplaceLeakyReLU(),
        transforms_mod.ReplaceLayerNorm()
    ]
    transformer = module_transform.ModuleTransformer(model, model_topo,
                                                     transforms)
    return transformer.transform(excluded_nodes)

  def _get_quantizer_by_config(self, tensor_type, key, index):
    config = self._qat_config.get(tensor_type, key, index)
    # We can't use config.quantizer directly. Reused modules lead to reused quantizer.
    # The quantizers in spec are the ones that are actually used.
    spec = self._node_to_spec[config.node.name]
    if tensor_type == TensorTypes.PARAM:
      quantizer = spec.get_weight_quantizer(config.weight_name)
    elif tensor_type == TensorTypes.INPUT:
      quantizer = spec.get_input_quantizers(index)
    else:
      quantizer = spec.get_output_quantizer(index)
    return quantizer

  def _set_fix_position(self, torch_quantizer):
    for tensor_type, group in torch_quantizer.quant_config.items():
      if tensor_type not in _valid_tensor_types:
        continue
      for key in group:
        for index in range(torch_quantizer.get_quant_len(key, tensor_type)):
          quantizer = self._get_quantizer_by_config(tensor_type, key, index)
          if quantizer.is_warmup_enabled():
            logging_warn(
                QWarning.SCALE_VALUE,
                'Exported scale values are not trained: {}.'.format(key))
          torch_quantizer.set_fix_position(key, quantizer.get_fix_position(),
                                           tensor_type, index)

  def _import_quant_info(self, torch_quantizer):
    for tensor_type, group in torch_quantizer.quant_config.items():
      if tensor_type not in _valid_tensor_types:
        continue
      for key in group:
        for index in range(torch_quantizer.get_quant_len(key, tensor_type)):
          bitwidth = torch_quantizer.get_bit_width(key, tensor_type, index)
          fix_position = torch_quantizer.get_fix_position(
              key, tensor_type, index)
          quantizer = self._get_quantizer_by_config(tensor_type, key, index)
          quantizer.import_quant_info(bitwidth, fix_position)

  def to_deployable(self, trained_model, output_dir):
    model = self._to_deployable(trained_model, output_dir)
    saved_path = os.path.join(output_dir, _DEPLOYABLE_MODEL_NAME)
    torch.save(model.state_dict(), saved_path)
    logging.info(
        f'Saving deployable model to {saved_path}, and you can get it by calling "deployable_model()"'
    )
    return self._qprocessor.quant_model()

  def convert_to_deployable(self, trained_model, output_dir):
    logging_warn(QWarning.DEPRECATED_ARGUMENT, (
        '"convert_to_deployable" is deprecated and will be removed in the future. '
        'Use "to_deployable" instead.'))
    return self._to_deployable(trained_model, output_dir)

  def _to_deployable(self, trained_model, output_dir):
    if not self._qat_config or self._module_map is None:
      logging_error(QError.QAT_PROCESS_ERROR,
                    'Must call "trainable_model" first.')

    if hasattr(trained_model, 'conv_bn_fused') and getattr(
        trained_model, 'conv_bn_fused'):
      logging_error(
          QError.QAT_DEPLOYABLE_MODEL_ERROR,
          ('The given trained model has bn fused to conv and cannot be '
           'converted to a deployable model. Make sure model.fuse_conv_bn()'
           'is not called.'))

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
    self._set_fix_position(quantizer)

    sub_dir = os.path.join(output_dir, 'test')
    io_util.create_work_dir(sub_dir)
    # Must set adjust_pos=False first, because quantizer will modify its
    # quant info inplace when adjust_pos=True.
    # Export original (not adjusted yet) quant info for testing deployable
    # model and the accuracy should be the same with the trainable model.
    quantizer.export_quant_config(
        os.path.join(sub_dir, _QUANT_INFO_FILE_NAME),
        adjust_pos=False,
        inference_check=False)
    quantizer.export_quant_config(
        os.path.join(output_dir, _QUANT_INFO_FILE_NAME),
        adjust_pos=True,
        inference_check=False)

    self._qprocessor = qprocessor
    return model

  def quant_model(self):
    if not self._trainable_model:
      self._trainable_model = self.trainable_model()
    return self._trainable_model

  def deployable_model(self, src_dir, used_for_xmodel=False):
    if used_for_xmodel:
      device = torch.device('cpu')
      inputs = tuple([inp.to(device) for inp in self._inputs])
    else:
      device = self._device
      inputs = self._inputs

    model = copy.deepcopy(self._model)
    saved_path = os.path.join(src_dir, _DEPLOYABLE_MODEL_NAME)
    logging.info(f'Loading deployable model from {saved_path}')
    model.load_state_dict(torch.load(saved_path, map_location=device))
    qprocessor = qproc.TorchQuantProcessor(
        'test',
        model,
        inputs,
        output_dir=src_dir,
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

  def dump_blob(self, model, trainable=True):
    if trainable:
      model.fuse_conv_bn()
      mod_util.enable_print_blob(model, self._graph)
    else:
      mod_util.enable_print_blob(model)

    model(*self._inputs)

  # export xmodel for compilation
  def export_xmodel(self, output_dir, deploy_check=False, dynamic_batch=False):
    if not hasattr(self, '_qprocessor'):
      logging_error(QError.QAT_PROCESS_ERROR,
                    'Must call "deployable_model" first.')

    if next(self._qprocessor.quant_model().parameters()).device != torch.device(
        'cpu'):
      logging_error(QError.XMODEL_DEVICE, (
          'Xmodel can only be exported in cpu mode, '
          'use deployable_model(src_dir, used_for_xmodel=True) to get a cpu model.'
      ))

    self._qprocessor.export_xmodel(output_dir, deploy_check, dynamic_batch)

  def export_onnx_model(self,
                        output_dir,
                        verbose=False,
                        dynamic_batch=False,
                        opset_version=None,
                        native_onnx=True,
                        dump_layers=False,
                        check_model=False,
                        opt_graph=False):
    self._qprocessor.export_onnx_model(output_dir, verbose, dynamic_batch,
                                       opset_version, native_onnx, dump_layers,
                                       check_model, opt_graph)

  def export_torch_script(self, output_dir, verbose=False):
    self._qprocessor.export_torch_script(output_dir, verbose)
