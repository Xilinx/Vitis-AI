# Copyright 2023 Xilinx Inc.
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

class RegisterSpec(object):
  """A decorator for registering the specification generating function for
  a op type.

  If a function for a type is registered multiple times, a KeyError will be
  raised.

  For example,
    @RegisterSpec(OpTypes.CONV2D)
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

  def __init__(self, model, graph, runtime_config):
    self.model = model
    self.graph = graph
    self.runtime_config = runtime_config

  @abc.abstractmethod
  def get_transforms(self):
    raise NotImplementedError

  @abc.abstractmethod
  def get_runtime_specification(self):
    raise NotImplementedError

  def apply(self):
    model_topo = model_topo_mod.build_model_topo(
        self.graph, self.get_runtime_specification())

    transformer = mt.ModuleTransformer(self.model, model_topo,
                                       self.get_transforms())
    return transformer.transform()[0]

class FP8QuantizeScheme(QuantizeScheme):

  def get_transforms(self):
    gemm_layer_transforms = [
        transforms_mod.QuantizeConv2dBatchNorm(),
        transforms_mod.QuantizeConv3dBatchNorm(),
        transforms_mod.QuantizeConvNd(),
        transforms_mod.QuantizeLinear()
    ]
    activation_transforms = [
        transforms_mod.ReplaceSoftmax(),
        transforms_mod.ReplaceSigmoid(),
        transforms_mod.ReplaceTanh(),
        transforms_mod.ReplaceGELU(),
        transforms_mod.ReplaceLayerNorm(),
    ]

    if mode.is_no_approx(self.runtime_config.approx_mode):
      return gemm_layer_transforms
    else:
      return gemm_layer_transforms + activation_transforms

  def get_runtime_specification(self):
    node_to_spec = {}
    for node in self.graph.nodes:
      generator = _spec_generator.get(node.op.type, self.__class__.default_spec)
      spec = generator(self)
      if not spec or not isinstance(spec, config_mod.LayerRuntimeSpec):
        raise ValueError(
            'Expecting a LayerRuntimeSpec object, but got {}'.format(
                type(spec)))
      node_to_spec[node.name] = spec
    return node_to_spec

  def _fp_args_from_runtime_config(self):
    return {
        'exponent_bits': self.runtime_config.exp_bits,
        'bias_mode': self.runtime_config.exp_bias_mode,
        'exponent_bias': self.runtime_config.exp_bias,
        'round_mode': self.runtime_config.round_mode,
    }

  def default_spec(self):
    return config_mod.LayerRuntimeSpec(self.runtime_config)

  @RegisterSpec(OpTypes.CONV2D)
  def conv_spec(self):
    fp_args = self._fp_args_from_runtime_config()

    spec = self.default_spec()
    spec.add_input_quantizer(nnq.FP8Quantizer(**fp_args))
    spec.add_weight_quantizer('weight', nnq.FP8Quantizer(**fp_args))
    spec.add_weight_quantizer('bias', nn.Identity())
    return spec

  @RegisterSpec(OpTypes.DEPTHWISE_CONV2D)
  def depth_conv_spec(self):
    fp_args = self._fp_args_from_runtime_config()

    spec = self.default_spec()
    spec.add_input_quantizer(nnq.FP8Quantizer(**fp_args))
    spec.add_weight_quantizer('weight', nnq.FP8Quantizer(**fp_args))
    spec.add_weight_quantizer('bias', nn.Identity())
    return spec

  @RegisterSpec(OpTypes.DENSE)
  def linear_spec(self):
    fp_args = self._fp_args_from_runtime_config()

    spec = self.default_spec()
    spec.add_input_quantizer(nnq.FP8Quantizer(**fp_args))
    spec.add_weight_quantizer('weight', nnq.FP8Quantizer(**fp_args))
    spec.add_weight_quantizer('bias', nn.Identity())
    return spec

  @RegisterSpec(OpTypes.MATMUL)
  def matmul_spec(self):
    fp_args = self._fp_args_from_runtime_config()

    spec = self.default_spec()
    spec.add_input_quantizer(nnq.FP8Quantizer(**fp_args))
    spec.add_input_quantizer(nnq.FP8Quantizer(**fp_args))
    spec.add_output_quantizer(nnq.FP32Quantizer())
    return spec

  @RegisterSpec([
      OpTypes.ADAPTIVEAVGPOOL2D, OpTypes.ADD, OpTypes.SIGMOID, OpTypes.TANH,
      OpTypes.SOFTMAX, OpTypes.GELU, OpTypes.RELU, 'aten::silu'
  ])
  def input_bfloat16_spec(self):
    spec = self.default_spec()
    spec.add_input_quantizer(nnq.BFloat16Quantizer())
    return spec

def quantize_model(model,
                   inputs,
                   exp_bits=5,
                   exp_bias_mode='ieee',
                   exp_bias=None):
  parser = parse.TorchParser()
  graph = parser(model._get_name(), model, inputs)

  rt_config = config_mod.RuntimeConfig()
  rt_config.exp_bits = exp_bits
  rt_config.exp_bias_mode = exp_bias_mode
  rt_config.exp_bias = exp_bias

  scheme = FP8QuantizeScheme(model, graph, rt_config)
  return scheme.apply()
