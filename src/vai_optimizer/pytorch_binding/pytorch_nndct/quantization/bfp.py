# Copyright 2022 Xilinx Inc.
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

import abc

from torch import nn
from torch import Tensor
from typing import Any, NoReturn, Optional, Sequence, Tuple, Union, List

from nndct_shared.base import NNDCT_OP as OpTypes
from nndct_shared.optimization.commander import OptimizeCommander
from nndct_shared.utils import registry

from pytorch_nndct import parse
from pytorch_nndct.nn import quantization as nnq
from pytorch_nndct.nn.nonlinear import mode
from pytorch_nndct.qproc.ModuleHooker import ModuleHooker
from pytorch_nndct.quantization import config as config_mod
from pytorch_nndct.quantization import model_topo as model_topo_mod
from pytorch_nndct.quantization import module_transform as mt
from pytorch_nndct.quantization import transforms as transforms_mod
from pytorch_nndct.utils import logging
from pytorch_nndct.utils import module_util as mod_util

_spec_generator = registry.Registry('Runtime Spec Generator')

class RegisterRuntimeSpec(object):
  """A decorator for registering the specification generating function for
  a op type.

  If a function for a type is registered multiple times, a KeyError will be
  raised.

  For example,
    @RegisterRuntimeSpec(OpTypes.CONV2D)
    def spec_for_conv2d():
      ...
  """

  def __init__(self, op_types):
    if not isinstance(op_types, (list, tuple)):
      op_types = [op_types]
    self._op_types = op_types

  def __call__(self, f):
    if not callable(f):
      raise TypeError("Registered function must be callable.")
    for op_type in self._op_types:
      if not isinstance(op_type, str):
        raise TypeError("op_type must be a string.")
      _spec_generator.register(f, op_type)
    return f

class QuantizeScheme(abc.ABC):

  def __init__(self, model, graph, config):
    self.model = model
    self.graph = graph
    self.config = config

  @abc.abstractmethod
  def get_transforms(self):
    raise NotImplementedError

  @abc.abstractmethod
  def get_runtime_specification(self):
    raise NotImplementedError

  def apply(self):
    node_to_spec = self.get_runtime_specification()
    for node_name, spec in node_to_spec.items():
      if len(spec.weight_quantizers) or len(spec.input_quantizers) or len(
          spec.output_quantizers):
        module = mod_util.get_module_by_node(self.model, node_name)
        node = self.graph.node(node_name)
        # A quantized op must be implemented as a module.
        if not module:
          raise ValueError(
              ('Can not quantize node "{}({})" as it is not a '
               'torch.nn.Module object, please re-implement this operation '
               'as a module. The original source range:\n{}').format(
                   node.name, node.op.type, node.source_range))

    model_topo = model_topo_mod.build_model_topo(self.graph, node_to_spec)
    transformer = mt.ModuleTransformer(self.model, model_topo,
                                       self.get_transforms())
    return transformer.transform()[0]

class BFPQuantizeScheme(QuantizeScheme):

  def __init__(self, model, graph, config):
    super(BFPQuantizeScheme, self).__init__(model, graph, config)

    prime_mode_to_quantizer = {
        None: nnq.BFPQuantizer,
        'normal': nnq.BFPPrimeQuantizer,
        'shared': nnq.BFPPrimeSharedQuantizer
    }

    if config.bfp.prime.mode not in prime_mode_to_quantizer:
      raise ValueError(
          f'BFP prime mode must be one of {prime_mode_to_quantizer.keys()}, but given {config.bfp.prime.mode}'
      )
    self.quantizer_cls = prime_mode_to_quantizer[config.bfp.prime.mode]

    self.bfp_args = self._bfp_args_from_runtime_config()

  def _bfp_args_from_runtime_config(self):
    args = {
        'bitwidth': self.config.bfp.bitwidth,
        'block_size': self.config.bfp.block_size,
        'rounding_mode': self.config.bfp.rounding_mode,
    }
    if self.config.bfp.prime.mode == 'shared':
      args['sub_block_size'] = self.config.bfp.prime.sub_block_size
      args['sub_block_shift_bits'] = self.config.bfp.prime.sub_block_shift_bits
    return args

  def get_transforms(self):
    return [
        transforms_mod.QuantizeConv2dBatchNorm(),
        transforms_mod.QuantizeConv3dBatchNorm(),
        transforms_mod.QuantizeConvNd(),
        transforms_mod.QuantizeLinear(),
        #transforms_mod.QuantizeBatchNormNd(),
        transforms_mod.ReplaceSoftmax(),
        transforms_mod.ReplaceSigmoid(),
        transforms_mod.ReplaceTanh(),
        transforms_mod.ReplaceGELU(),
        transforms_mod.ReplaceLayerNorm(),
    ]

  def get_runtime_specification(self):
    node_to_spec = {}
    for node in self.graph.nodes:
      spec = self.default_spec()
      generator = _spec_generator.get(node.op.type, None)
      if generator:
        generator(self, spec)
      node_to_spec[node.name] = spec
    return node_to_spec

  def default_spec(self):
    return config_mod.LayerRuntimeSpec(self.config)

  @RegisterRuntimeSpec(OpTypes.CONV2D)
  def conv_spec(self, spec):
    spec.add_input_quantizer(self.quantizer_cls(**self.bfp_args, axis=1))
    spec.add_weight_quantizer('weight',
                              self.quantizer_cls(**self.bfp_args, axis=1))
    spec.add_weight_quantizer('bias', nn.Identity())

  @RegisterRuntimeSpec(OpTypes.DEPTHWISE_CONV2D)
  def depthwise_conv_spec(self, spec):
    spec.add_input_quantizer(self.quantizer_cls(**self.bfp_args, axis=1))
    spec.add_weight_quantizer('weight',
                              self.quantizer_cls(**self.bfp_args, axis=1))
    spec.add_weight_quantizer('bias', nn.Identity())

  @RegisterRuntimeSpec(OpTypes.CONVTRANSPOSE2D)
  def conv_transpose_spec(self, spec):
    spec.add_input_quantizer(self.quantizer_cls(**self.bfp_args, axis=1))
    spec.add_weight_quantizer('weight',
                              self.quantizer_cls(**self.bfp_args, axis=1))
    spec.add_weight_quantizer('bias', nn.Identity())

  #@RegisterRuntimeSpec(OpTypes.BATCH_NORM)
  #def depthwise_conv_spec(self, spec):
  #  spec.add_input_quantizer(nnq.BFloat16Quantizer())
  #  spec.add_weight_quantizer('weight', nnq.BFloat16Quantizer())
  #  spec.add_weight_quantizer('bias', nnq.BFloat16Quantizer())
  #  spec.add_weight_quantizer('running_mean', nnq.BFloat16Quantizer())
  #  spec.add_weight_quantizer('running_var', nnq.BFloat16Quantizer())

  @RegisterRuntimeSpec(OpTypes.DENSE)
  def linear_spec(self, spec):
    spec.add_input_quantizer(self.quantizer_cls(**self.bfp_args, axis=-1))
    spec.add_weight_quantizer('weight',
                              self.quantizer_cls(**self.bfp_args, axis=-1))
    spec.add_weight_quantizer('bias', nn.Identity())

  @RegisterRuntimeSpec(OpTypes.ADAPTIVEAVGPOOL2D)
  def adaptive_avg_pool_spec(self, spec):
    # input: (N, C, H, W) or (C, H, W)
    spec.add_input_quantizer(nnq.BFloat16Quantizer())

  #@RegisterRuntimeSpec(OpTypes.MULTIPLY)
  #def multiply_spec(self, spec):
  #  spec.add_input_quantizer(self.quantizer_cls(**self.bfp_args))
  #  spec.add_input_quantizer(self.quantizer_cls(**self.bfp_args))

  @RegisterRuntimeSpec(OpTypes.MATMUL)
  def matmul_spec(self, spec):
    spec.add_input_quantizer(self.quantizer_cls(**self.bfp_args, axis=-1))
    spec.add_input_quantizer(self.quantizer_cls(**self.bfp_args, axis=-2))
    spec.add_output_quantizer(nnq.FP32Quantizer())

  @RegisterRuntimeSpec(OpTypes.GELU)
  def gelu_spec(self, spec):
    if mode.is_no_approx(self.config.non_linear_approx.mode):
      spec.add_input_quantizer(nnq.BFloat16Quantizer())

  @RegisterRuntimeSpec(OpTypes.SIGMOID)
  def sigmoid_spec(self, spec):
    if mode.is_no_approx(self.config.non_linear_approx.mode):
      spec.add_input_quantizer(nnq.BFloat16Quantizer())

  @RegisterRuntimeSpec(OpTypes.SOFTMAX)
  def softmax_spec(self, spec):
    if mode.is_no_approx(self.config.non_linear_approx.mode):
      spec.add_input_quantizer(nnq.BFloat16Quantizer())

  @RegisterRuntimeSpec(OpTypes.TANH)
  def tanh_spec(self, spec):
    if mode.is_no_approx(config.non_linear_approx.mode):
      spec.add_input_quantizer(nnq.BFloat16Quantizer())

  @RegisterRuntimeSpec(OpTypes.LAYER_NORM)
  def layernorm_spec(self, spec):
    if mode.is_no_approx(self.config.non_linear_approx.mode):
      spec.add_input_quantizer(nnq.BFloat16Quantizer())
      spec.add_weight_quantizer('weight', nnq.BFloat16Quantizer())

def _get_graph(model, inputs):
  return parse.TorchParser()(model._get_name(), model, inputs)

def quantize_model(model: nn.Module,
                   inputs: Tuple[Tensor, ...],
                   dtype: str = 'mx6',
                   config_file: str = None) -> nn.Module:
  if config_file:
    logging.info(f'Loading config from {config_file}')
    rt_config = config_mod.RuntimeConfig.from_yaml(config_file)
  else:
    logging.info(f'Getting config for {dtype}')
    rt_config = config_mod.get(dtype)
  logging.info('RuntimeConfig: {}'.format(rt_config))

  graph = _get_graph(model, inputs)

  #if not runtime_config.training:
  #  model, graph = equalize_weights_cross_conv_layers(model, inputs)

  scheme = BFPQuantizeScheme(model, graph, rt_config)
  return scheme.apply()

def _attach_node_to_model(model, graph):
  for node in graph.nodes:
    module = mod_util.get_module_by_node(model, node)
    if module:
      module.node = node

def _detach_node_from_model(model):

  def delete_node(module):
    if hasattr(module, 'node'):
      del module.node

  model.apply(delete_node)

def equalize_weights_cross_conv_layers(model, inputs):
  model.eval()
  graph = _get_graph(model, inputs)
  model_topo = model_topo_mod.build_model_topo(graph, {})
  transformer = mt.ModuleTransformer(model, model_topo, [
      transforms_mod.FuseConv2dBatchNorm(),
      transforms_mod.FuseConv3dBatchNorm()
  ])
  model, _, _ = transformer.transform()

  graph = _get_graph(model, inputs)
  commander = OptimizeCommander(graph)
  commander.DecoupleSharedParamsInConv()
  graph = commander.equalize_weights_cross_conv_layers()

  _attach_node_to_model(model, graph)
  ModuleHooker.update_parameters(model, graph, True)
  _detach_node_from_model(model)

  return model, graph
