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

import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor
from torch.nn.modules import batchnorm

from pytorch_nndct.nn.nonlinear import approx
from pytorch_nndct.nn.nonlinear import mode
from nndct_shared.utils import NndctOption

# Adopted from https://github.com/pytorch/pytorch/blob/v1.13.1/torch/nn/modules/batchnorm.py#L121
class _QuantizedBatchNorm(batchnorm._BatchNorm):

  def __init__(self,
               num_features: int,
               eps: float = 1e-5,
               momentum: float = 0.1,
               affine: bool = True,
               track_running_stats: bool = True,
               device=None,
               dtype=None,
               rt_spec=None) -> None:
    super(_QuantizedBatchNorm,
          self).__init__(num_features, eps, momentum, affine, track_running_stats)

    assert rt_spec, 'Runtime spec must be provided for quantized module'
    self.rt_spec = rt_spec

    self.running_mean_quantizer = rt_spec.maybe_get_weight_quantizer('running_mean')
    self.running_var_quantizer = rt_spec.maybe_get_weight_quantizer('running_var')

    self.weight_quantizer = rt_spec.maybe_get_weight_quantizer('weight')
    self.bias_quantizer = rt_spec.maybe_get_weight_quantizer('bias')

  @property
  def is_quantized(self):
    return True

  def forward(self, input: Tensor) -> Tensor:
    self._check_input_dim(input)

    # exponential_average_factor is set to self.momentum
    # (when it is available) only so that it gets updated
    # in ONNX graph when this node is exported to ONNX.
    if self.momentum is None:
      exponential_average_factor = 0.0
    else:
      exponential_average_factor = self.momentum

    if self.training and self.track_running_stats:
      # TODO: if statement only here to tell the jit to skip emitting this when it is None
      if self.num_batches_tracked is not None:  # type: ignore[has-type]
        self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        if self.momentum is None:  # use cumulative moving average
          exponential_average_factor = 1.0 / float(self.num_batches_tracked)
        else:  # use exponential moving average
          exponential_average_factor = self.momentum
    r"""
    Decide whether the mini-batch stats should be used for normalization rather than the buffers.
    Mini-batch stats are used in training mode, and in eval mode when buffers are None.
    """
    if self.training:
      bn_training = True
    else:
      bn_training = (self.running_mean is None) and (self.running_var is None)
    r"""
    Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
    passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
    used for normalization (i.e. in eval mode when buffers are not None).
    """

    running_mean = self.running_mean_quantizer(
        self.running_mean) if self.running_mean_quantizer else self.running_mean
    running_var = self.running_var_quantizer(
        self.running_var) if self.running_var_quantizer else self.running_var

    weight = self.weight_quantizer(
        self.weight) if self.weight_quantizer else self.weight
    bias = self.bias_quantizer(self.bias) if self.bias_quantizer else self.bias

    return F.batch_norm(
        input,
        # If buffers are not to be tracked, ensure that they won't be updated
        running_mean if not self.training or self.track_running_stats else None,
        running_var if not self.training or self.track_running_stats else None,
        weight,
        bias,
        bn_training,
        exponential_average_factor,
        self.eps,
    )

  @classmethod
  def from_float(cls, mod, rt_spec):
    """Create a quantized module from a float module."""
    assert rt_spec, 'Runtime spec must be provided for quantized module.'
    assert type(mod) == cls._FLOAT_MODULE, \
        '{}.from_float() only accepts {}, but got {}'.format(
          cls.__name__, cls._FLOAT_MODULE, type(mod))

    norm = cls(
        mod.num_features,
        mod.eps,
        mod.momentum,
        mod.affine,
        mod.track_running_stats,
        rt_spec=rt_spec)

    norm.weight = mod.weight
    norm.bias = mod.bias
    norm.running_mean = mod.running_mean
    norm.running_var = mod.running_var
    norm.num_batches_tracked = mod.num_batches_tracked
    return norm

class QuantizedBatchNorm2d(_QuantizedBatchNorm):
  _FLOAT_MODULE = nn.BatchNorm2d

  def _check_input_dim(self, input):
    nn.BatchNorm2d._check_input_dim(self, input)

class QuantizedBatchNorm3d(_QuantizedBatchNorm):
  _FLOAT_MODULE = nn.BatchNorm3d

  def _check_input_dim(self, input):
    nn.BatchNorm3d._check_input_dim(self, input)
