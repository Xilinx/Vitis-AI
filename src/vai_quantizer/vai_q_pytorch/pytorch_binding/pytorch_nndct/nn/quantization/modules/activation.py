import torch

from functools import partial
from torch import nn
from torch.nn import functional as F

from pytorch_nndct.nn.quantization.ops import quantize_ops
from pytorch_nndct.nn.nonlinear import approx
from pytorch_nndct.nn.nonlinear import mode

class DPULeakyReLU(nn.LeakyReLU):

  def __init__(self, *args, **kwargs):
    # only support the specified slope and inplace operation
    super().__init__(*args, **kwargs)
    self.negative_slope = 0.1015625

  def forward(self, inputs):
    return super().forward(inputs)

class Softmax(nn.Module):

  def __init__(self, dim, approx_mode=None, approx_degree=3, exp_table_size=1):
    super(Softmax, self).__init__()

    self.dim = dim
    self.approx_mode = approx_mode
    self.approx_degree = approx_degree
    self.exp_table_size = exp_table_size

    if mode.is_no_approx(self.approx_mode):
      self._forward_fn = partial(F.softmax, dim=self.dim)
    elif mode.is_exp_poly(self.approx_mode):
      self._forward_fn = self._forward_approx_poly
    elif mode.is_exp_lut(self.approx_mode):
      self._forward_fn = self._forward_approx_lut
    else:
      raise ValueError(
          'Approximate mode `{}` is not valid. Must be one of {}'.format(
              self.approx_mode, mode.available_modes()))

  def _forward_approx_poly(self, input):
    return approx.softmax_approx_poly(
        input,
        axis=self.dim,
        degree=self.approx_degree,
        exp_table_size=self.exp_table_size).float()

  def _forward_approx_lut(self, input):
    return approx.softmax_approx_lut(input, axis=self.dim).float()

  def forward(self, input):
    return self._forward_fn(input)

  def extra_repr(self):
    return 'dim={}, approx_mode={}, approx_degree={}, exp_table_size={}'.format(
        self.dim, self.approx_mode, self.approx_degree, self.exp_table_size)

class GELU(nn.Module):

  def __init__(self, approx_mode, approx_degree):
    super(GELU, self).__init__()

    self.approx_mode = approx_mode
    self.approx_degree = approx_degree

    self._forward_fn = F.gelu if mode.is_no_approx(
        self.approx_mode) else self._forward_gelu_approx

  def _forward_gelu_approx(self, input):
    return approx.gelu_approx(
        input.bfloat16(), degree=self.approx_degree).float()

  def forward(self, input):
    return self._forward_fn(input)

  def extra_repr(self):
    return 'approx_mode={}, approx_degree={}'.format(
        self.approx_mode, self.approx_degree)

class Sigmoid(nn.Module):
  def __init__(self, approx_mode=None, approx_degree=3, exp_table_size=1):
    super(Sigmoid, self).__init__()

    self.approx_mode = approx_mode
    self.approx_degree = approx_degree
    self.exp_table_size = exp_table_size

  def forward(self, input):
    if self.approx_mode == 'no_approx':
      return torch.sigmoid(input)
    elif self.approx_mode == 'exp_poly':
      return approx.sigmoid_with_exp_approx(input, degree=self.approx_degree, exp_table_size=self.exp_table_size).float()
    elif self.approx_mode == 'positive_poly':
      return approx.sigmoid_positive_approx(input, degree=self.approx_degree).float()
    elif self.approx_mode == 'exp_lut':
      return approx.sigmoid_with_exp_approx_lut(input).float()
    else:
      raise NotImplementedError(f'approx_mode {self.approx_mode} is not exists in Sigmoid')

class Tanh(nn.Module):
  def __init__(self, approx_mode=None, approx_degree=3, exp_table_size=1):
    super(Tanh, self).__init__()

    self.approx_mode = approx_mode
    self.approx_degree = approx_degree
    self.exp_table_size = exp_table_size

  def forward(self, input):
    if self.approx_mode == 'no_approx':
      return torch.tanh(input)
    elif self.approx_mode == 'exp_poly':
      return approx.tanh_with_exp_approx(input, degree=self.approx_degree, exp_table_size=self.exp_table_size).float()
    elif self.approx_mode == 'positive_poly':
      return approx.tanh_positive_approx(input, degree=self.approx_degree).float()
    elif self.approx_mode == 'exp_lut':
      return approx.tanh_with_exp_approx_lut(input).float()
    else:
      raise NotImplementedError(f'approx_mode {self.approx_mode} is not exists in Tanh')
