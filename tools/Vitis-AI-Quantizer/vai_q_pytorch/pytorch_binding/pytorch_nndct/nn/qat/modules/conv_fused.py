

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

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

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
      freeze_bn=False,
      qconfig=None):
    nn.modules.conv._ConvNd.__init__(self, in_channels, out_channels,
                                     kernel_size, stride, padding, dilation,
                                     transposed, output_padding, groups, False,
                                     padding_mode)
    assert qconfig, 'qconfig must be provided for QAT module'
    self.frozen = freeze_bn if self.training else True
    self.bn = nn.BatchNorm2d(out_channels, eps, momentum, True, True)

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
      if freeze_bn:
        self.freeze_bn()
      else:
        self.update_bn()
    else:
      self.freeze_bn()

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

  def update_bn(self):
    self.frozen = False
    self.bn.training = True
    return self

  def freeze_bn(self):
    if self.frozen:
      return

    with torch.no_grad():
      # The same implementation as nndct_shared/optimzation/fuse_conv_bn.py
      # is used so that the test accruacy is same as the deployable model.
      gamma = self.bn.weight.detach().cpu().numpy()
      beta = self.bn.bias.detach().cpu().numpy()
      running_var = self.bn.running_var.detach().cpu().numpy()
      running_mean = self.bn.running_mean.detach().cpu().numpy()
      epsilon = self.bn.eps

      scale = gamma / np.sqrt(running_var + epsilon)
      offset = beta - running_mean * scale

      weight = self.weight.detach().cpu().numpy()
      weight = np.multiply(
          weight.transpose(1, 2, 3, 0), scale).transpose(3, 0, 1, 2)
      self.weight.copy_(torch.from_numpy(weight))

      bias = self.bias.detach.cpu().numpy() if self.bias is not None else 0
      bias = torch.from_numpy(bias * scale + offset)
      if self.bias is not None:
        self.bias.copy_(bias)
      else:
        self.bias = nn.Parameter(bias)

    self.frozen = True
    self.bn.training = False
    return

  def broadcast_correction(self, c: torch.Tensor):
    """Broadcasts a correction factor to the output for elementwise operations."""
    expected_output_dim = 4
    view_fillers_dim = expected_output_dim - c.dim() - 1
    view_filler = (1,) * view_fillers_dim
    expected_view_shape = c.shape + view_filler
    return c.view(*expected_view_shape)

  def broadcast_correction_weight(self, c):
    """Broadcasts a correction factor to the weight."""
    if c.dim() != 1:
      raise ValueError("Correction factor needs to have a single dimension")
    expected_weight_dim = 4
    view_fillers_dim = expected_weight_dim - c.dim()
    view_filler = (1,) * view_fillers_dim
    expected_view_shape = c.shape + view_filler
    return c.view(*expected_view_shape)

  def extra_repr(self):
    return super(_ConvBnNd, self).extra_repr()

  def forward(self, x):
    gamma, beta = self.bn.weight, self.bn.bias
    if self.frozen:
      quantized_weight = self.weight_quantizer(self.weight)
      quantized_bias = self.bias_quantizer(self.bias)
      return self._conv_forward(x, quantized_weight, quantized_bias)

    if self.training:
      batch_mean, batch_var = self.batch_stats(
          self._conv_forward(x, self.weight), self.bias)
      recip_sigma_batch = torch.rsqrt(batch_var + self.bn.eps)
      with torch.no_grad():
        sigma_running = torch.sqrt(self.bn.running_var + self.bn.eps)

      w_corrected = self.weight * self.broadcast_correction_weight(
          gamma / sigma_running)
      w_quantized = self.weight_quantizer(w_corrected)
      recip_c = self.broadcast_correction(sigma_running * recip_sigma_batch)
      bias_corrected = beta - gamma * batch_mean * recip_sigma_batch
      bias_quantized = self.broadcast_correction(
          self.bias_quantizer(bias_corrected))

      y = self._conv_forward(x, w_quantized, None)
      y.mul_(recip_c).add_(bias_quantized)
    else:
      with torch.no_grad():
        recip_sigma_running = torch.rsqrt(self.bn.running_var + self.bn.eps)
      w_corrected = self.weight * self.broadcast_correction_weight(
          gamma * recip_sigma_running)
      w_quantized = self.weight_quantizer(w_corrected)
      corrected_mean = self.bn.running_mean - (
          self.bias if self.bias is not None else 0)
      bias_corrected = beta - gamma * corrected_mean * recip_sigma_running
      bias_quantized = self.bias_quantizer(bias_corrected)
      y = self._conv_forward(x, w_quantized, bias_quantized)
      #print('w_quantized:', w_quantized.sum())
      #print('bias_quantized:', bias_quantized.sum())
      #print('conv2d output:', y.sum())
    return y

  def train(self, mode=True):
    """Batchnorm's training behavior is using the self.training flag. Prevent
    changing it if BN is frozen. This makes sure that calling `model.train()`
    on a model with a frozen BN will behave properly.
    """
    self.training = mode
    if not self.frozen:
      for module in self.children():
        module.train(mode)
    return self

  # ===== Serialization version history =====
  #
  # Version 1/None
  #   self
  #   |--- weight : Tensor
  #   |--- bias : Tensor
  #   |--- gamma : Tensor
  #   |--- beta : Tensor
  #   |--- running_mean : Tensor
  #   |--- running_var : Tensor
  #   |--- num_batches_tracked : Tensor
  #
  # Version 2
  #   self
  #   |--- weight : Tensor
  #   |--- bias : Tensor
  #   |--- bn : Module
  #        |--- weight : Tensor (moved from v1.self.gamma)
  #        |--- bias : Tensor (moved from v1.self.beta)
  #        |--- running_mean : Tensor (moved from v1.self.running_mean)
  #        |--- running_var : Tensor (moved from v1.self.running_var)
  #        |--- num_batches_tracked : Tensor (moved from v1.self.num_batches_tracked)
  def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                            missing_keys, unexpected_keys, error_msgs):
    version = local_metadata.get('version', None)
    if version is None or version == 1:
      # BN related parameters and buffers were moved into the BN module for v2
      v2_to_v1_names = {
          'bn.weight': 'gamma',
          'bn.bias': 'beta',
          'bn.running_mean': 'running_mean',
          'bn.running_var': 'running_var',
          'bn.num_batches_tracked': 'num_batches_tracked',
      }
      for v2_name, v1_name in v2_to_v1_names.items():
        if prefix + v1_name in state_dict:
          state_dict[prefix + v2_name] = state_dict[prefix + v1_name]
          state_dict.pop(prefix + v1_name)
        elif strict:
          missing_keys.append(prefix + v2_name)

    super(_ConvBnNd,
          self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

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

class ConvBatchNorm2d(_ConvBnNd, nn.Conv2d):
  """A ConvBatchNorm2d module is a module fused from Conv2d and BatchNorm2d,
    attached with FakeQuantize modules for both output activation and weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d`.

    Implementation details: https://arxiv.org/pdf/1806.08342.pdf section 3.2.2

    Similar to :class:`torch.nn.Conv2d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        activation_quant_fn: fake quant module for output activation
        weight_fake_quant: fake quant module for weight

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
      padding_mode='zeros',
      # BatchNorm2d args
      # num_features: out_channels
      eps=1e-05,
      momentum=0.1,
      # affine: True
      # track_running_stats: True
      # Args for this module
      freeze_bn=False,
      qconfig=None):
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    _ConvBnNd.__init__(self, in_channels, out_channels, kernel_size, stride,
                       padding, dilation, False, _pair(0), groups, bias,
                       padding_mode, eps, momentum, freeze_bn, qconfig)

  def _conv_forward(self, input, w, b=None):
    return F.conv2d(input, w, b, self.stride, self.padding, self.dilation,
                    self.groups)

# TODO(yuwang): Move to top api for user.
def update_bn(mod):
  if type(mod) in set([ConvBatchNorm2d]):
    mod.update_bn()

def freeze_bn(mod):
  if type(mod) in set([ConvBatchNorm2d]):
    mod.freeze_bn()
