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
#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from torch.nn.modules.utils import _pair
from torch.nn.modules.utils import _triple
from torch.nn.parameter import Parameter

_BN_CLASS_MAP = {
    1: nn.BatchNorm1d,
    2: nn.BatchNorm2d,
    3: nn.BatchNorm3d,
}

# Adopted from https://github.com/pytorch/pytorch/blob/master/torch/nn/intrinsic/qat/modules/conv_fused.py
class _ConvBnNd(nn.modules.conv._ConvNd):

  _version = 2

  def __init__(
      self,
      # ConvNd args
      in_channels,
      out_channels,
      kernel_size,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups,
      bias,
      padding_mode,
      # BatchNormNd args
      # num_features: out_channels
      eps=1e-05,
      momentum=0.1,
      # affine: True
      # track_running_stats: True
      # Args for this module
      freeze_bn_stats=False,
      qconfig=None,
      dim=2):
    nn.modules.conv._ConvNd.__init__(self, in_channels, out_channels,
                                     kernel_size, stride, padding, dilation,
                                     transposed, output_padding, groups, False,
                                     padding_mode)
    assert qconfig and qconfig.weight and qconfig.bias, 'qconfig must be provided for QAT module'
    self.bn_frozen = freeze_bn_stats if self.training else True
    self.dim = dim
    self.bn = _BN_CLASS_MAP[dim](out_channels, eps, momentum, True, True)

    self.weight_quantizer = qconfig.weight
    self.bias_quantizer = qconfig.bias

    if bias:
      self.bias = Parameter(torch.Tensor(out_channels))
    else:
      self.register_parameter('bias', None)
    self.reset_bn_parameters()

    # this needs to be called after reset_bn_parameters,
    # as they modify the same state
    if self.training:
      if freeze_bn_stats:
        self.freeze_bn_stats()
      else:
        self.update_bn_stats()
    else:
      self.freeze_bn_stats()

    self.conv_bn_fused = False

  def reset_running_stats(self):
    self.bn.reset_running_stats()

  def reset_bn_parameters(self):
    self.bn.reset_running_stats()
    init.uniform_(self.bn.weight)
    init.zeros_(self.bn.bias)
    # note: below is actully for conv, not BN
    if self.bias is not None:
      fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
      bound = 1 / math.sqrt(fan_in)
      init.uniform_(self.bias, -bound, bound)

  def batch_stats(self, x, bias=None):
    """Get the batch mean and variance of x and updates the BatchNorm's running mean and average.

      Args:
        x (torch.Tensor): input batch.
        bias (torch.Tensor): the bias that is to be applied to the batch.

      Returns:
        (mean, variance)

      Note:
        In case of `nn.Linear`, x may be of shape (N, C, L) or (N, L)
        where N is batch size, C is number of channels, L is the features size.
        The batch norm computes the stats over C in the first case or L on the second case.
        The batch normalization layer is
        (`nn.BatchNorm1d`)[https://pytorch.org/docs/stable/nn.html#batchnorm1d]

        In case of `nn.Conv2d`, x is of shape (N, C, H, W)
        where H,W are the image dimensions, and the batch norm computes the stats over C.
        The batch normalization layer is
        (`nn.BatchNorm2d`)[https://pytorch.org/docs/stable/nn.html#batchnorm2d]

        In case of `nn.Conv3d`, x is of shape (N, C, D, H, W)
        where H,W are the image dimensions, D is additional channel dimension,
        and the batch norm computes the stats over C.
        The batch normalization layer is
        (`nn.BatchNorm3d`)[https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm3d.html#torch.nn.BatchNorm3d]
    """
    channel_size = self.bn.num_features
    self.bn.num_batches_tracked += 1

    # Calculate current batch stats
    batch_mean = x.transpose(0, 1).contiguous().view(channel_size, -1).mean(1)
    # BatchNorm currently uses biased variance (without Bessel's correction) as was discussed at
    # https://github.com/pytorch/pytorch/issues/1410
    #
    # also see the source code itself:
    # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Normalization.cpp#L216
    batch_var = x.transpose(0, 1).contiguous().view(channel_size, -1).var(
        1, unbiased=False)

    # Update running stats
    with torch.no_grad():
      biased_batch_mean = batch_mean + (bias if bias is not None else 0)
      # However - running_var is updated using unbiased variance!
      # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Normalization.cpp#L223
      n = x.numel() / channel_size
      corrected_var = batch_var * (n / float(n - 1))
      momentum = self.bn.momentum
      if momentum is None:
        # momentum is None - we compute a cumulative moving average
        # as noted in https://pytorch.org/docs/stable/nn.html#batchnorm2d
        momentum = 1. / float(self.bn.num_batches_tracked)
      self.bn.running_mean.mul_(1 - momentum).add_(momentum * biased_batch_mean)
      self.bn.running_var.mul_(1 - momentum).add_(momentum * corrected_var)

    return batch_mean, batch_var

  def reset_parameters(self):
    super(_ConvBnNd, self).reset_parameters()

  def merge_bn_to_conv(self):
    with torch.no_grad():
      # Use the same implementation in nndct_shared/optimzation/fuse_conv_bn.py
      # to make sure the test accruacy is same as the deployable model.
      gamma = self.bn.weight.detach().cpu().numpy()
      beta = self.bn.bias.detach().cpu().numpy()
      running_var = self.bn.running_var.detach().cpu().numpy()
      running_mean = self.bn.running_mean.detach().cpu().numpy()
      epsilon = self.bn.eps

      scale = gamma / np.sqrt(running_var + epsilon)
      offset = beta - running_mean * scale

      weight = self.weight.detach().cpu().numpy()
      # Conv2d
      if self.dim == 2 and not self.transposed:
        # OIHW -> IHWO -> OIHW
        weight = np.multiply(weight.transpose(1, 2, 3, 0),
                             scale).transpose(3, 0, 1, 2)
      # ConvTranspose2d
      elif self.dim == 2 and self.transposed:
        # IOHW -> IHWO -> IOHW
        weight = np.multiply(weight.transpose(0, 2, 3, 1),
                             scale).transpose(0, 3, 1, 2)
      # Conv3D
      elif self.dim == 3 and not self.transposed:
        weight = np.multiply(weight.transpose(1, 2, 3, 4, 0),
                             scale).transpose(4, 0, 1, 2, 3)
      # ConvTranspose3d
      elif self.dim == 3 and self.transposed:
        weight = np.multiply(weight.transpose(2, 3, 4, 0, 1),
                             scale).transpose(3, 4, 0, 1, 2)
      else:
        raise RuntimeError(
            'Unsupported combinations: (dim={}, transposed={})'.format(
                self.dim, self.transposed))
      self.weight.copy_(torch.from_numpy(weight))

      bias = self.bias.detach().cpu().numpy() if self.bias is not None else 0
      bias = torch.from_numpy(bias * scale + offset)
      if self.bias is not None:
        self.bias.copy_(bias)
      else:
        self.bias = nn.Parameter(bias)
    self.conv_bn_fused = True

  def update_bn_stats(self):
    self.bn_frozen = False

  def freeze_bn_stats(self):
    self.bn_frozen = True

  def clear_non_native_bias(self):
    if self.bias is None:
      print('[WARNING] No bias to unmerge')
      return

    with torch.no_grad():
      gamma = self.bn.weight.detach().cpu().numpy()
      beta = self.bn.bias.detach().cpu().numpy()
      running_var = self.bn.running_var.detach().cpu().numpy()
      running_mean = self.bn.running_mean.detach().cpu().numpy()
      epsilon = self.bn.eps

      scale = gamma / np.sqrt(running_var + epsilon)

      bias = self.bias.detach().cpu().numpy()
      beta = torch.from_numpy(bias * scale + beta)
      self.bn.bias.copy_(beta)
      self.bias = None

  def broadcast_correction(self, c):
    """Broadcasts a correction factor to the output for elementwise operations.

    Two tensors are “broadcastable” if the following rules hold:
      - Each tensor has at least one dimension.
      - When iterating over the dimension sizes, starting at the trailing
        dimension, the dimension sizes must either be equal,
        one of them is 1, or one of them does not exist.
    See https://pytorch.org/docs/stable/notes/broadcasting.html
    """
    expected_output_dim = self.dim + 2
    view_fillers_dim = expected_output_dim - c.dim() - 1
    view_filler = (1,) * view_fillers_dim
    expected_view_shape = c.shape + view_filler
    return c.view(*expected_view_shape)

  def broadcast_correction_weight(self, c):
    """Broadcasts a correction factor to the weight."""
    if c.dim() != 1:
      raise ValueError("Correction factor needs to have a single dimension")

    expected_weight_dim = self.dim + 2
    view_fillers_dim = expected_weight_dim - c.dim()
    view_filler = (1,) * view_fillers_dim
    expected_view_shape = c.shape + view_filler
    return c.view(*expected_view_shape)

  def extra_repr(self):
    return super(_ConvBnNd, self).extra_repr()

  def forward(self, x, output_size=None):
    """
    See https://arxiv.org/pdf/1806.08342.pdf section 3.2.2.
    bn(conv(x)) = (conv(x) - E(conv(x))) * gamma / std(conv(x)) + beta
                = (x*W + B - E(x*W + B)) * gamma / sqrt(E((x*W + B - E(x*W + B))^2)) + beta
                = (x*W - E(x*W)) * gamma / std(x*W) + beta
    """
    gamma, beta = self.bn.weight, self.bn.bias
    if self.conv_bn_fused:
      quantized_weight = self.weight_quantizer(self.weight)
      quantized_bias = self.bias_quantizer(self.bias)
      return self._conv_forward(x, quantized_weight, quantized_bias,
                                output_size)

    if self.training and not self.bn_frozen:
      batch_mean, batch_var = self.batch_stats(
          self._conv_forward(x, self.weight, output_size=output_size),
          self.bias)
      recip_sigma_batch = torch.rsqrt(batch_var + self.bn.eps)
      with torch.no_grad():
        running_sigma = torch.sqrt(self.bn.running_var + self.bn.eps)

      w_corrected = self.weight * self.broadcast_correction_weight(
          gamma / running_sigma)
      w_quantized = self.weight_quantizer(w_corrected)
      recip_c = self.broadcast_correction(running_sigma * recip_sigma_batch)
      bias_corrected = beta - gamma * batch_mean * recip_sigma_batch
      bias_quantized = self.broadcast_correction(
          self.bias_quantizer(bias_corrected))

      y = self._conv_forward(x, w_quantized, None, output_size)
      y.mul_(recip_c).add_(bias_quantized)
    else:
      with torch.no_grad():
        recip_running_sigma = torch.rsqrt(self.bn.running_var + self.bn.eps)
      w_corrected = self.weight * self.broadcast_correction_weight(
          gamma * recip_running_sigma)
      w_quantized = self.weight_quantizer(w_corrected)
      mean_corrected = self.bn.running_mean - (
          self.bias if self.bias is not None else 0)
      bias_corrected = beta - gamma * mean_corrected * recip_running_sigma
      bias_quantized = self.bias_quantizer(bias_corrected)
      y = self._conv_forward(x, w_quantized, bias_quantized, output_size)
    return y

  def train(self, mode=True):
    """Batchnorm's training behavior is using the self.training flag. Prevent
    changing it if BN is frozen. This makes sure that calling `model.train()`
    on a model with a frozen BN will behave properly.
    """
    self.training = mode
    if not self.bn_frozen:
      for module in self.children():
        module.train(mode)
    return self

  @property
  def is_quantized(self):
    return True

  @classmethod
  def from_float(cls, conv, bn, qconfig):
    """Create a qat module from a float module."""
    assert qconfig, 'Input float module must have a valid qconfig'
    convbn = cls(conv.in_channels, conv.out_channels, conv.kernel_size,
                 conv.stride, conv.padding, conv.dilation, conv.groups,
                 conv.bias is not None, conv.padding_mode, bn.eps, bn.momentum,
                 False, qconfig)
    convbn.weight = conv.weight
    convbn.bias = conv.bias
    convbn.bn.weight = bn.weight
    convbn.bn.bias = bn.bias
    convbn.bn.running_mean = bn.running_mean
    convbn.bn.running_var = bn.running_var
    convbn.bn.num_batches_tracked = bn.num_batches_tracked
    convbn.bn.eps = bn.eps
    return convbn

class QuantizedConvBatchNorm2d(_ConvBnNd, nn.Conv2d):
  """A QuantizedConvBatchNorm2d module is a module fused from
    Conv2d and BatchNorm2d attached with FakeQuantizer modules for weight and
    batchnorm stuffs used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d`.

    Implementation details: https://arxiv.org/pdf/1806.08342.pdf section 3.2.2

    Similar to :class:`torch.nn.Conv2d`, with FakeQuantizer modules initialized
    to default.
    """

  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size,
      stride=1,
      padding=0,
      dilation=1,
      groups=1,
      bias=None,
      padding_mode='zeros',
      # BatchNorm2d args
      # num_features: out_channels
      eps=1e-05,
      momentum=0.1,
      # affine: True
      # track_running_stats: True
      # Args for this module
      freeze_bn_stats=False,
      qconfig=None):
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    _ConvBnNd.__init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        False,
        _pair(0),
        groups,
        bias,
        padding_mode,
        eps,
        momentum,
        freeze_bn_stats,
        qconfig,
        dim=2)

  def _conv_forward(self, input, weight, bias=None, output_size=None):
    assert output_size is None
    if self.padding_mode != 'zeros':
      return F.conv2d(
          F.pad(
              input,
              self._reversed_padding_repeated_twice,
              mode=self.padding_mode), weight, bias, self.stride, _pair(0),
          self.dilation, self.groups)
    return F.conv2d(input, weight, bias, self.stride, self.padding,
                    self.dilation, self.groups)

class QuantizedConvBatchNorm3d(_ConvBnNd, nn.Conv3d):
  """A QuantizedConvBatchNorm3d module is a module fused from
    Conv3d and BatchNorm3d attached with FakeQuantizer modules for weight and
    batchnorm stuffs used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv3d` and
    :class:`torch.nn.BatchNorm3d`.

    Similar to `QuantizedConvBatchNorm3d`.
    """

  def __init__(
      self,
      # ConvNd args
      in_channels,
      out_channels,
      kernel_size,
      stride=1,
      padding=0,
      dilation=1,
      groups=1,
      bias=None,
      padding_mode="zeros",
      # BatchNorm3d args
      # num_features: out_channels
      eps=1e-05,
      momentum=0.1,
      # affine: True
      # track_running_stats: True
      # Args for this module
      freeze_bn_stats=False,
      qconfig=None,
  ):
    kernel_size = _triple(kernel_size)
    stride = _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)
    _ConvBnNd.__init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        False,
        _triple(0),
        groups,
        bias,
        padding_mode,
        eps,
        momentum,
        freeze_bn_stats,
        qconfig,
        dim=3,
    )

  def _conv_forward(self, input, weight, bias=None, output_size=None):
    assert output_size is None
    if self.padding_mode != 'zeros':
      return F.conv3d(
          F.pad(
              input,
              self._reversed_padding_repeated_twice,
              mode=self.padding_mode), weight, bias, self.stride, _triple(0),
          self.dilation, self.groups)
    return F.conv3d(input, weight, bias, self.stride, self.padding,
                    self.dilation, self.groups)

class _ConvTransposeBnNd(_ConvBnNd, nn.modules.conv._ConvTransposeMixin):

  @classmethod
  def from_float(cls, conv, bn, qconfig):
    """Create a qat module from a float module."""
    assert qconfig, 'Input float module must have a valid qconfig'
    convbn = cls(conv.in_channels, conv.out_channels, conv.kernel_size,
                 conv.stride, conv.padding, conv.output_padding, conv.groups,
                 conv.bias is not None, conv.dilation, conv.padding_mode,
                 bn.eps, bn.momentum, False, qconfig)
    convbn.weight = conv.weight
    convbn.bias = conv.bias
    convbn.bn.weight = bn.weight
    convbn.bn.bias = bn.bias
    convbn.bn.running_mean = bn.running_mean
    convbn.bn.running_var = bn.running_var
    convbn.bn.num_batches_tracked = bn.num_batches_tracked
    convbn.bn.eps = bn.eps
    return convbn

class QuantizedConvTransposeBatchNorm2d(_ConvTransposeBnNd):

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               stride=1,
               padding=0,
               output_padding=0,
               groups=1,
               bias=None,
               dilation=1,
               padding_mode='zeros',
               eps=1e-05,
               momentum=0.1,
               freeze_bn_stats=False,
               qconfig=None):

    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    output_padding = _pair(output_padding)
    super(QuantizedConvTransposeBatchNorm2d,
          self).__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, False, output_padding, groups, bias,
                         padding_mode, eps, momentum, freeze_bn_stats, qconfig)

  def _conv_forward(self, input, weight, bias=None, output_size=None):
    if self.padding_mode != 'zeros':
      raise ValueError(
          'Only `zeros` padding mode is supported for QuantizedConvTransposeBatchNorm2d'
      )

    output_padding = self._output_padding(input, output_size, self.stride,
                                          self.padding, self.kernel_size)

    return F.conv_transpose2d(input, weight, bias, self.stride, self.padding,
                              output_padding, self.groups, self.dilation)

  def broadcast_correction_weight(self, c):
    """Broadcasts a correction factor to the weight."""
    if c.dim() != 1:
      raise ValueError("Correction factor needs to have a single dimension")
    # weight.shape: [in_channels, out_channels // groups, *kernel_size]
    expected_view_shape = (1,) + c.shape + (1,) * 2
    return c.view(*expected_view_shape)

class QuantizedConvTransposeBatchNorm3d(_ConvTransposeBnNd):

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               stride=1,
               padding=0,
               output_padding=0,
               groups=1,
               bias=None,
               dilation=1,
               padding_mode='zeros',
               eps=1e-05,
               momentum=0.1,
               freeze_bn_stats=False,
               qconfig=None):

    #kernel_size = _pair(kernel_size)
    #stride = _pair(stride)
    #padding = _pair(padding)
    #dilation = _pair(dilation)
    #output_padding = _pair(output_padding)
    super(QuantizedConvTransposeBatchNorm3d, self).__init__(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        False,
        output_padding,
        groups,
        bias,
        padding_mode,
        eps,
        momentum,
        freeze_bn_stats,
        qconfig,
        dim=3)

  def _conv_forward(self, input, weight, bias=None, output_size=None):
    if self.padding_mode != 'zeros':
      raise ValueError(
          'Only `zeros` padding mode is supported for QuantizedConvTransposeBatchNorm3d'
      )

    output_padding = self._output_padding(input, output_size, self.stride,
                                          self.padding, self.kernel_size)

    return F.conv_transpose3d(input, weight, bias, self.stride, self.padding,
                              output_padding, self.groups, self.dilation)

  def broadcast_correction_weight(self, c):
    """Broadcasts a correction factor to the weight."""
    if c.dim() != 1:
      raise ValueError("Correction factor needs to have a single dimension")
    # weight.shape: [in_channels, out_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]]
    expected_view_shape = (1,) + c.shape + (1,) * 3
    return c.view(*expected_view_shape)

_FUSED_CLS = [
    QuantizedConvBatchNorm2d, QuantizedConvBatchNorm3d,
    QuantizedConvTransposeBatchNorm2d, QuantizedConvTransposeBatchNorm3d
]

def update_bn_stats(mod):
  if type(mod) in _FUSED_CLS:
    mod.update_bn_stats()

def freeze_bn_stats(mod):
  if type(mod) in _FUSED_CLS:
    mod.freeze_bn_stats()

def fuse_conv_bn(mod):
  if type(mod) in _FUSED_CLS:
    mod.merge_bn_to_conv()

def clear_non_native_bias(mod):
  if type(mod) in _FUSED_CLS:
    mod.clear_non_native_bias()
