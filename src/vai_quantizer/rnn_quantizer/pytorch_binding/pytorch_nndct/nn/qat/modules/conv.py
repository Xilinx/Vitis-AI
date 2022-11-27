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

import torch.nn as nn
import torch.nn.functional as F
import warnings

from torch.nn.modules.utils import _pair
from torch.nn.modules.utils import _triple

class _QuantizedConvNd(nn.modules.conv._ConvNd):

  def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
               dilation, transposed, output_padding, groups, bias, padding_mode,
               qconfig):
    super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                     dilation, transposed, output_padding, groups, bias,
                     padding_mode)
    assert qconfig, 'qconfig must be provided for quantized module'
    self.qconfig = qconfig

    self.weight_quantizer = qconfig.weight
    if bias:
      self.bias_quantizer = qconfig.bias

  @property
  def is_quantized(self):
    return True

class QuantizedConv2d(_QuantizedConvNd):
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
               qconfig=None):
    super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                     dilation, False, _pair(0), groups, bias, padding_mode,
                     qconfig)

  def forward(self, input):
    quantized_weight = self.weight_quantizer(self.weight)
    quantized_bias = self.bias_quantizer(
        self.bias) if self.bias is not None else None
    #print('QConv quantized weight & bias:', quantized_weight.sum(), quantized_bias.sum() if self.bias is not None else None)
    if self.padding_mode != 'zeros':
      return F.conv2d(
          F.pad(
              input,
              self._reversed_padding_repeated_twice,
              mode=self.padding_mode), quantized_weight, quantized_bias,
          self.stride, _pair(0), self.dilation, self.groups)

    return F.conv2d(input, quantized_weight, quantized_bias, self.stride,
                    self.padding, self.dilation, self.groups)

  @classmethod
  def from_float(cls, mod, qconfig):
    """Create a qat module from a float module.

    Args:
      mod: A float module of type torch.nn.Conv2d.
      qconfig (pytorch_nndct.quantization.quant_aware_training.QConfig):
          A qconfig object that saves the quantizers for the module.
    """

    assert qconfig, 'qconfig must be provided for quantized module'
    assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + \
        '.from_float only works for ' + cls._FLOAT_MODULE.__name__

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
        qconfig=qconfig)
    conv.weight = mod.weight
    conv.bias = mod.bias
    return conv

class QuantizedConv3d(_QuantizedConvNd):
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
      qconfig=None,
  ):
    super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias, padding_mode, qconfig)

  def forward(self, input):
    quantized_weight = self.weight_quantizer(self.weight)
    quantized_bias = self.bias_quantizer(
        self.bias) if self.bias is not None else None

    if self.padding_mode != 'zeros':
        return F.conv3d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                        quantized_weight, quantized_bias, self.stride, _triple(0),
                        self.dilation, self.groups)
    return F.conv3d(input, quantized_weight, quantized_bias, self.stride,
                    self.padding, self.dilation, self.groups)

  @classmethod
  def from_float(cls, mod, qconfig):
    """Create a qat module from a float module.

    Args:
      mod: A float module of type torch.nn.Conv2d.
      qconfig (pytorch_nndct.quantization.quant_aware_training.QConfig):
          A qconfig object that saves the quantizers for the module.
    """
    assert qconfig, 'qconfig must be provided for quantized module'
    assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + \
        '.from_float only works for ' + cls._FLOAT_MODULE.__name__

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
        qconfig=qconfig,
    )
    conv.weight = mod.weight
    conv.bias = mod.bias
    return conv

class QuantizedConvTranspose2d(nn.modules.conv._ConvTransposeMixin,
                               _QuantizedConvNd):
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
               qconfig=None):
    super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                     dilation, True, output_padding, groups, bias, padding_mode,
                     qconfig)

  def forward(self, input, output_size=None):
    if self.padding_mode != 'zeros':
      raise ValueError(
          'Only `zeros` padding mode is supported for ConvTranspose2d')

    output_padding = self._output_padding(input, output_size, self.stride,
                                          self.padding, self.kernel_size)

    quantized_weight = self.weight_quantizer(self.weight)
    quantized_bias = self.bias_quantizer(
        self.bias) if self.bias is not None else None

    return F.conv_transpose2d(input, quantized_weight, quantized_bias,
                              self.stride, self.padding, output_padding,
                              self.groups, self.dilation)

  @classmethod
  def from_float(cls, mod, qconfig):
    """Create a qat module from a float module.

    Args:
      mod: A float module of type torch.nn.ConvTranspose2d.
      qconfig (pytorch_nndct.quantization.quant_aware_training.QConfig):
          A qconfig object that saves the quantizers for the module.
    """

    assert qconfig, 'qconfig must be provided for quantized module'
    if type(mod) != cls._FLOAT_MODULE:
      warnings.warn('{} is expected to create from {}, but given {}.'.format(
          cls.__name__, cls._FLOAT_MODULE.__name__,
          type(mod).__name__))

    conv_transpose = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                         mod.stride, mod.padding, mod.output_padding,
                         mod.groups, mod.bias is not None, mod.dilation,
                         mod.padding_mode, qconfig)
    conv_transpose.weight = mod.weight
    conv_transpose.bias = mod.bias
    return conv_transpose

class QuantizedConvTranspose3d(nn.modules.conv._ConvTransposeMixin,
                               _QuantizedConvNd):
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
               qconfig=None):
    super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                     dilation, True, output_padding, groups, bias, padding_mode,
                     qconfig)

  def forward(self, input, output_size=None):
    if self.padding_mode != 'zeros':
      raise ValueError(
          'Only `zeros` padding mode is supported for ConvTranspose2d')

    output_padding = self._output_padding(input, output_size, self.stride,
                                          self.padding, self.kernel_size)

    quantized_weight = self.weight_quantizer(self.weight)
    quantized_bias = self.bias_quantizer(
        self.bias) if self.bias is not None else None

    return F.conv_transpose3d(input, quantized_weight, quantized_bias,
                              self.stride, self.padding, output_padding,
                              self.groups, self.dilation)

  @classmethod
  def from_float(cls, mod, qconfig):
    """Create a qat module from a float module.

    Args:
      mod: A float module of type torch.nn.ConvTranspose3d.
      qconfig (pytorch_nndct.quantization.quant_aware_training.QConfig):
          A qconfig object that saves the quantizers for the module.
    """

    assert qconfig, 'qconfig must be provided for quantized module'
    if type(mod) != cls._FLOAT_MODULE:
      warnings.warn('{} is expected to create from {}, but given {}.'.format(
          cls.__name__, cls._FLOAT_MODULE.__name__,
          type(mod).__name__))

    conv_transpose = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                         mod.stride, mod.padding, mod.output_padding,
                         mod.groups, mod.bias is not None, mod.dilation,
                         mod.padding_mode, qconfig)
    conv_transpose.weight = mod.weight
    conv_transpose.bias = mod.bias
    return conv_transpose
