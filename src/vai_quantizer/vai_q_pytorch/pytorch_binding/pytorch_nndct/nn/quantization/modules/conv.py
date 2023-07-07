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
import torch.nn as nn
import torch.nn.functional as F
import warnings

from distutils.version import LooseVersion
from typing import Optional, List, Tuple, Union

from torch import Tensor
from torch.nn.modules.utils import _pair
from torch.nn.modules.utils import _single
from torch.nn.modules.utils import _triple

class _QuantizedConvNd(nn.modules.conv._ConvNd):

  def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
               dilation, transposed, output_padding, groups, bias, padding_mode,
               rt_spec):
    super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                     dilation, transposed, output_padding, groups, bias,
                     padding_mode)
    assert rt_spec, 'Runtime spec must be provided for quantized module'
    self.rt_spec = rt_spec

    self.weight_quantizer = rt_spec.get_weight_quantizer('weight')
    if bias:
      self.bias_quantizer = rt_spec.get_weight_quantizer('bias')

  @property
  def is_quantized(self):
    return True

class _QuantizedConv(_QuantizedConvNd):

  def forward(self, input: Tensor) -> Tensor:
    quantized_weight = self.weight_quantizer(self.weight)
    quantized_bias = self.bias_quantizer(
        self.bias) if self.bias is not None else None
    return self._conv_forward(input, quantized_weight, quantized_bias)

  @classmethod
  def from_float(cls, mod, rt_spec):
    """Create a qat module from a float module.

    Args:
      mod: The float module to be quantized.
          Must be one of type [nn.Conv1d, nn.Conv2d, nn.Conv3d]
      rt_spec (pytorch_nndct.quantization.config.RuntimeSpec):
          An object that specifies the quantizers and config for the module.
    """

    assert rt_spec, 'Runtime spec must be provided for quantized module'
    if type(mod) != cls._FLOAT_MODULE:
      warnings.warn('{} is expected to create from {}, but given {}.'.format(
          cls.__name__, cls._FLOAT_MODULE.__name__,
          type(mod).__name__))

    conv = cls(
        mod.in_channels,
        mod.out_channels,
        mod.kernel_size,
        stride=mod.stride,
        padding=mod.padding,
        dilation=mod.dilation,
        groups=mod.groups,
        bias=mod.bias is not None,
        padding_mode=mod.padding_mode,
        rt_spec=rt_spec)
    conv.weight = mod.weight
    conv.bias = mod.bias
    return conv

class QuantizedConv1d(_QuantizedConv):
  """A Conv1d module attached with FakeQuantizer modules for weight and bias,
    used for quantization aware training.

    The interface is adopted from `torch.nn.Conv1d`, please see
    https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    for documentation.

  """

  _FLOAT_MODULE = nn.Conv1d

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               stride=1,
               padding=0,
               dilation=1,
               groups=1,
               bias=True,
               padding_mode='zeros',
               rt_spec=None):
    super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                     dilation, False, _pair(0), groups, bias, padding_mode,
                     rt_spec)

  def _conv_forward(self, input: Tensor, weight: Tensor,
                    bias: Optional[Tensor]):
    if self.padding_mode != 'zeros':
      return F.conv1d(
          F.pad(
              input,
              self._reversed_padding_repeated_twice,
              mode=self.padding_mode), weight, bias, self.stride, _single(0),
          self.dilation, self.groups)
    return F.conv1d(input, weight, bias, self.stride, self.padding,
                    self.dilation, self.groups)

class QuantizedConv2d(_QuantizedConv):
  """A Conv2d module attached with FakeQuantizer modules for weight and bias,
    used for quantization aware training.

    The interface is adopted from `torch.nn.Conv2d`, please see
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    for documentation.

  """
  _FLOAT_MODULE = nn.Conv2d

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               stride=1,
               padding=0,
               dilation=1,
               groups=1,
               bias=True,
               padding_mode='zeros',
               rt_spec=None):
    super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                     dilation, False, _pair(0), groups, bias, padding_mode,
                     rt_spec)

  def _conv_forward(self, input: Tensor, weight: Tensor,
                    bias: Optional[Tensor]):
    if self.padding_mode != 'zeros':
      return F.conv2d(
          F.pad(
              input,
              self._reversed_padding_repeated_twice,
              mode=self.padding_mode), weight, bias, self.stride, _pair(0),
          self.dilation, self.groups)
    return F.conv2d(input, weight, bias, self.stride, self.padding,
                    self.dilation, self.groups)

class QuantizedConv3d(_QuantizedConv):
  """A Conv3d module attached with FakeQuantizer modules for weight and bias,
    used for quantization aware training.

    The interface is adopted from `torch.nn.Conv3d`, please see
    https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
    for documentation.

  """
  _FLOAT_MODULE = nn.Conv3d

  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size,
      stride=1,
      padding=0,
      dilation=1,
      groups=1,
      bias=True,
      padding_mode="zeros",
      rt_spec=None,
  ):
    super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                     dilation, False, _triple(0), groups, bias, padding_mode,
                     rt_spec)

  def _conv_forward(self, input: Tensor, weight: Tensor,
                    bias: Optional[Tensor]):
    if self.padding_mode != "zeros":
      return F.conv3d(
          F.pad(
              input,
              self._reversed_padding_repeated_twice,
              mode=self.padding_mode),
          weight,
          bias,
          self.stride,
          _triple(0),
          self.dilation,
          self.groups,
      )
    return F.conv3d(input, weight, bias, self.stride, self.padding,
                    self.dilation, self.groups)

# XXX(yuwang): Must first inherit from _QuantizedConvNd so that its
# __init__ will be selected for calling for super().__init__.
# Have to inherit nn.modules.conv._ConvTransposeMixin for PyTorch 1.4.
class _QuantizedConvTransposeNd(_QuantizedConvNd,
                                nn.modules.conv._ConvTransposeMixin):
#class _QuantizedConvTransposeNd(_QuantizedConvNd):

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               stride=1,
               padding=0,
               output_padding=0,
               groups=1,
               bias=True,
               dilation=1,
               padding_mode='zeros',
               rt_spec=None,
               dim=1):

    super(_QuantizedConvTransposeNd,
          self).__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, True, output_padding, groups, bias,
                         padding_mode, rt_spec)

    self.dim = dim
    self._transpose_fn = [
        F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d
    ][dim - 1]

    if torch.__version__ < LooseVersion('1.12.0'):
      self.forward = self.forward_prior_torch112
    else:
      self.forward = self.forward_from_torch112

  def forward_prior_torch112(self, input, output_size=None):
    if self.padding_mode != 'zeros':
      raise ValueError(
          f'Only `zeros` padding mode is supported for {self.__class__.__name__}'
      )

    output_padding = self._output_padding(input, output_size, self.stride,
                                          self.padding, self.kernel_size)

    quantized_weight = self.weight_quantizer(self.weight)
    quantized_bias = self.bias_quantizer(
        self.bias) if self.bias is not None else None

    return self._transpose_fn(input, quantized_weight, quantized_bias,
                              self.stride, self.padding, output_padding,
                              self.groups, self.dilation)

  def forward_from_torch112(self, input, output_size=None):
    if self.padding_mode != 'zeros':
      raise ValueError(
          f'Only `zeros` padding mode is supported for {self.__class__.__name__}'
      )

    output_padding = self._output_padding(input, output_size, self.stride,
                                          self.dim, self.padding,
                                          self.kernel_size)

    quantized_weight = self.weight_quantizer(self.weight)
    quantized_bias = self.bias_quantizer(
        self.bias) if self.bias is not None else None

    return self._transpose_fn(input, quantized_weight, quantized_bias,
                              self.stride, self.padding, output_padding,
                              self.groups, self.dilation)

  @classmethod
  def from_float(cls, mod, rt_spec):
    """Create a qat module from a float module.

    Args:
      mod: A float module of type torch.nn.ConvTranspose1d/2d/3d.
      rt_spec (pytorch_nndct.quantization.config.RuntimeSpec):
          An object that specifies the quantizers and config for the module.
    """

    assert rt_spec, 'rt_spec must be provided for quantized module'
    if type(mod) != cls._FLOAT_MODULE:
      warnings.warn('{} is expected to create from {}, but given {}.'.format(
          cls.__name__, cls._FLOAT_MODULE.__name__,
          type(mod).__name__))

    conv_transpose = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                         mod.stride, mod.padding, mod.output_padding,
                         mod.groups, mod.bias is not None, mod.dilation,
                         mod.padding_mode, rt_spec)
    conv_transpose.weight = mod.weight
    conv_transpose.bias = mod.bias
    return conv_transpose

class QuantizedConvTranspose2d(_QuantizedConvTransposeNd):
  """A ConvTranspose2d module attached with FakeQuantizer modules for weight and bias,
    used for quantization aware training.

  The interface is adopted from `torch.nn.ConvTranspose2d`, please see
  https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
  for documentation.
  """

  _FLOAT_MODULE = nn.ConvTranspose2d

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               stride=1,
               padding=0,
               output_padding=0,
               groups=1,
               bias=True,
               dilation=1,
               padding_mode='zeros',
               rt_spec=None):
    super(QuantizedConvTranspose2d, self).__init__(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        bias,
        dilation,
        padding_mode,
        rt_spec,
        dim=2)

class QuantizedConvTranspose3d(_QuantizedConvTransposeNd):
  """A ConvTranspose3d module attached with FakeQuantizer modules for weight and bias,
    used for quantization aware training.

  The interface is adopted from `torch.nn.ConvTranspose3d`, please see
  https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
  for documentation.
  """

  _FLOAT_MODULE = nn.ConvTranspose3d

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               stride=1,
               padding=0,
               output_padding=0,
               groups=1,
               bias=True,
               dilation=1,
               padding_mode='zeros',
               rt_spec=None):
    super(QuantizedConvTranspose3d, self).__init__(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        bias,
        dilation,
        padding_mode,
        rt_spec,
        dim=3)
