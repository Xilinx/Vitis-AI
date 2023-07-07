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
from pytorch_nndct.parse.parse_utils import get_short_name
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

def logging_warn(code, message):
  NndctScreenLogger().warning2user(code, message)

def logging_error(code, message):
  NndctScreenLogger().check2user(code, message, False)

class TensorTypes(object):
  PARAM = 'param'
  INPUT = 'input'
  OUTPUT = 'output'

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
      logging_warn(QWarning.DEPRECATED_ARGUMENT,
                   ('The argument "device" is no longer needed. '
                    'Device information is get from the model directly.'))
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

    self._qinfo_keys = [
        TensorTypes.PARAM, TensorTypes.INPUT, TensorTypes.OUTPUT
    ]

    # Use hard-coded value to fill in fp_pos and export quant config,
    # so that we can initialize a new TorchQuantProcessor in 'test' mode later.
    quant_config = quantizer.quant_config
    for key, group in quant_config.items():
      if key not in self._qinfo_keys:
        continue
      for item in group:
        quantizer.set_fix_position(item, 4, key)
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

    def get_bitwidth(quant_info):
      return quant_info[0] if quant_info[0] == 8 else bitwidth

    # Create quantizer for each item in quant config.
    self._node_to_qconfig = {}
    self._quant_config = copy.deepcopy(quant_config)
    for name, group in self._quant_config.items():
      if name not in self._qinfo_keys:
        continue
      for key in group:
        qinfo = quantizer.get_quant_config(key, False, name)
        if name == TensorTypes.PARAM:
          node, param = self._tensor_to_node[key]
          if self._graph.node(node).has_bound_params():
            attr = ModuleHooker._parameter_map[param]
          else:
            attr = param.value
          tensor_type = 'weight'
        else:
          node, attr = key, None
          tensor_type = 'act'
        
        quant_method = None
        if (not name == TensorTypes.PARAM) and NndctOption.nndct_ip_v70_bert_qat.value:
          specil_method_op = ['nndct_layer_norm', 'nndct_softmax']
          if self._graph.node(node).op.type in specil_method_op:
            print(node)
            quant_method = 4

        tqt_quantizer = tqt_mod.TQTQuantizer(get_bitwidth(qinfo), tensor_type, method = quant_method)
        qconfig = self._node_to_qconfig.get(node, config_mod.LayerRuntimeSpec())
        if name == TensorTypes.PARAM:
          qconfig.add_weight_quantizer(attr, tqt_quantizer)
        elif name == TensorTypes.INPUT:
          qconfig.add_input_quantizer(tqt_quantizer)
        else:
          qconfig.add_output_quantizer(tqt_quantizer)

        self._node_to_qconfig[node] = qconfig
        self._quant_config[name][key] = (node, attr)
        logging.vlog(2, f'[{name}][{key}] = ({node}, {attr})')

  def trainable_model(self, calib_dir='', allow_reused_module=False):
    if calib_dir:
      self._torch_quantizer.load_quant_config(
          os.path.join(calib_dir, _QUANT_INFO_FILE_NAME))
      self._import_quant_info(self._torch_quantizer)

    self._check_reused_module(allow_reused_module)

    self._check_op_quantizable()

    model_topo = model_topo_mod.build_model_topo(self._graph,
                                                 self._node_to_qconfig)

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
    # use the same qconfig.
    module_to_qconfig = {}
    for node in self._graph.nodes:
      module_name = mod_util.module_name_from_node(node)
      if not module_name or node.name not in self._node_to_qconfig:
        continue

      if module_name in module_to_qconfig:
        if allow_reused_module:
          self._node_to_qconfig[node.name] = module_to_qconfig[module_name]
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
      module_to_qconfig[module_name] = self._node_to_qconfig[node.name]

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

    for name, group in self._quant_config.items():
      if name not in self._qinfo_keys:
        continue
      for key in group:
        node_name, _ = self._quant_config[name][key]

        module = mod_util.get_module_by_node(self._model, node_name)
        node = self._graph.node(node_name)

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
          logging_error(QError.NOT_A_MODULE,
                        ('{} is a quantized operation, please re-implement '
                         'your op as a nn.Module (Node: {})').format(
                             torch_op_attr.op_name, node_name))

  def _check_module_quantized(self, model):
    """Check that all parameterized modules are transformed to the
    corresponding quantized version."""

    for node, qconfig in self._node_to_qconfig.items():
      if qconfig.weight_quantizers:
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
    ]
    if NndctOption.nndct_ip_v70_bert_qat.value:
      transforms.append(transforms_mod.ReplaceLayerNorm())
    transformer = module_transform.ModuleTransformer(model, model_topo,
                                                     transforms)
    return transformer.transform(excluded_nodes)

  def _quantizer_by_quant_config(self, tensor_type, key):
    node, attr = self._quant_config[tensor_type][key]
    qconfig = self._node_to_qconfig[node]
    if tensor_type == TensorTypes.PARAM:
      quantizer = qconfig.get_weight_quantizer(attr)
    elif tensor_type == TensorTypes.INPUT:
      quantizer = qconfig.input_quantizers[0]
    else:
      quantizer = qconfig.output_quantizers[0]
    return quantizer

  def _fill_in_quant_config(self, quantizer):
    for tensor_type, group in quantizer.quant_config.items():
      if tensor_type not in self._qinfo_keys:
        continue
      for key in group:
        tqt_quantizer = self._quantizer_by_quant_config(tensor_type, key)
        if tqt_quantizer.is_warmup_enabled():
          logging_warn(QWarning.SCALE_VALUE,
                       'Exported scale values are not trained: {}.'.format(key))
        group[key] = tqt_quantizer.export_quant_info()

  def _import_quant_info(self, quantizer):
    for tensor_type, group in quantizer.quant_config.items():
      if tensor_type not in self._qinfo_keys:
        continue
      for key in group:
        qconfig = quantizer.get_quant_config(key, False, tensor_type)
        tqt_quantizer = self._quantizer_by_quant_config(tensor_type, key)
        tqt_quantizer.import_quant_info(qconfig)

  def to_deployable(self, trained_model, output_dir):
    model = self._to_deployable(trained_model, output_dir)
    torch.save(model.state_dict(),
               os.path.join(output_dir, _DEPLOYABLE_MODEL_NAME))
    return self._qprocessor.quant_model()

  def convert_to_deployable(self, trained_model, output_dir):
    logging_warn(QWarning.DEPRECATED_ARGUMENT, (
        '"convert_to_deployable" is deprecated and will be removed in the future. '
        'Use "to_deployable" instead.'))
    return self._to_deployable(trained_model, output_dir)

  def _to_deployable(self, trained_model, output_dir):
    if not self._quant_config or self._module_map is None:
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
    self._fill_in_quant_config(quantizer)

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
    model.load_state_dict(
        torch.load(
            os.path.join(src_dir, _DEPLOYABLE_MODEL_NAME), map_location=device))
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
                        opset_version=None):
    self._qprocessor.export_onnx_model(output_dir, verbose, dynamic_batch,
                                       opset_version)

  def export_torch_script(self, output_dir, verbose=False):
    self._qprocessor.export_torch_script(output_dir, verbose)
