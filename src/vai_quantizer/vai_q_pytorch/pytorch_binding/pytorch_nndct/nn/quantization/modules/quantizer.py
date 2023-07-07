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

from pytorch_nndct.nn.modules import fix_ops

class BFloat16Quantizer(nn.Module):

  def __init__(self):
    super(BFloat16Quantizer, self).__init__()

  def forward(self, x):
    return x.bfloat16().float()

class FP32Quantizer(nn.Module):

  def __init__(self):
    super(FP32Quantizer, self).__init__()

  def forward(self, x):
    return x.float()

class FakeQuantizer(nn.Module):
  """Simulate the quantize and dequantize operations in training time.

  In general, the output of this module is given by
  x_out = (clamp(round(x / scale + zero_point), quant_min, quant_max) - zero_point) * scale
  See https://arxiv.org/pdf/1903.08066.pdf

  In nndct, we use symmetric quantization and power-of-2 scaling. That is,
    zero_point = 0,
    quant_min = -2^(bitwidth - 1),
    quant_max = 2^(bitwidth - 1) - 1
  """
  _version = 2

  def __init__(self, bitwidth):
    super(FakeQuantizer, self).__init__()
    # quant_enabled is registered as buffer to support their replication in DDP.
    # Data type is uint8 because NCCL does not support bool tensors.
    self.register_buffer('quant_enabled', torch.tensor([1], dtype=torch.uint8))
    self.register_buffer('bitwidth', torch.tensor([bitwidth],
                                                  dtype=torch.uint8))
    self.register_buffer('domain', torch.tensor([2**(bitwidth - 1)]).float())

  def forward(self, x):
    raise NotImplementedError(
        'Do not use FakeQuantizer directly, please use its derivatives.')

  # PyTorch has been using _save_to_state_dict since 1.2.0.
  # See https://github.com/pytorch/pytorch/blob/v1.2.0/torch/nn/modules/module.py.
  def _save_to_state_dict(self, destination, prefix, keep_vars):
    super(FakeQuantizer, self)._save_to_state_dict(destination, prefix,
                                                   keep_vars)
    destination.pop(prefix + 'quant_enabled')
    destination.pop(prefix + 'domain')

  def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                            missing_keys, unexpected_keys, error_msgs):
    # We save 'bitwidth' to state_dict but not load it.
    # In low-bit tranining, bitwidth incrementally decreases from 8 -> 6 -> 4.
    # So the bitwidth should be get from quantizer's initialization argument
    # instead of state dict

    # For checkpoint BC with version 1.
    replace_map = {'num_bits': 'bitwidth'}

    version = local_metadata.get('version', None)
    if version is None or version < 2:
      keys = list(state_dict.keys())
      for key in keys:
        key_parts = key.split('.')
        weight_name = key_parts[-1]
        if weight_name in replace_map:
          key_parts[-1] = replace_map[weight_name]
          new_key = '.'.join(key_parts)
          assert new_key not in state_dict
          state_dict[new_key] = state_dict[key]
          state_dict.pop(key)

    # Check if bitwidth in the state dict but not load it.
    missing_bitwidth = False
    bitwidth_key = prefix + 'bitwidth'
    if bitwidth_key not in state_dict:
      missing_bitwidth = True
    else:
      # The value of bitwidth should be set at initilization.
      state_dict.pop(bitwidth_key)

    super(FakeQuantizer,
          self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    ignored_params = ['bitwidth', 'quant_enabled', 'domain']
    ignored_keys = [prefix + name for name in ignored_params]
    for key in ignored_keys:
      if key in missing_keys:
        if key == bitwidth_key and missing_bitwidth:
          continue
        missing_keys.remove(key)
      else:
        print('[WARNING] Unexpected key in state dict:', key)

class TQTQuantize(torch.autograd.Function):
  """Trained Quantization Thresholds.

  See https://arxiv.org/pdf/1903.08066.pdf
  """

  @staticmethod
  def forward(ctx, x, logt, domain, method):
    scale = 2**(torch.ceil(logt)) / domain
    quant_max = domain - 1
    quant_min = -domain

    ctx.save_for_backward(x, scale, quant_max, quant_min, logt)

    x = x.clone()
    #ctx.mark_dirty(x)
    return fix_ops.NndctFixNeuron(x, x, (domain, 1 / scale), method)
    #return torch.clamp(torch.round(x/scale), quant_min.item(), quant_max.item()) * scale

  @staticmethod
  def backward(ctx, grad_output):
    x, scale, quant_max, quant_min, logt = ctx.saved_tensors

    scaled_x = x / scale

    # Python equivalent to NndctFixNeuron rounding implementation which is
    # consistent with hardware runtime.
    # See nndct/include/cuda/nndct_fix_kernels.cuh::_fix_neuron_v2_device
    # Round -1.5 to -1 instead of -2.
    rounded_scaled_x = torch.where(
        (scaled_x < 0) & (scaled_x - torch.floor(scaled_x) == 0.5),
        torch.ceil(scaled_x), torch.round(scaled_x))

    is_lt_min = rounded_scaled_x < quant_min
    is_gt_max = rounded_scaled_x > quant_max
    is_ge_min_and_le_max = ~is_lt_min & ~is_gt_max

    # Equation (7) in section 3.3
    #grad_logt = torch.ones(grad_output.shape, dtype=grad_output.dtype, device=grad_output.device) * scale * math.log(2)
    grad_logt = grad_output * scale * math.log(2)
    grad_logt = torch.where(is_ge_min_and_le_max,
                            grad_logt * (rounded_scaled_x - scaled_x),
                            grad_logt)
    grad_logt = torch.where(is_lt_min, grad_logt * quant_min, grad_logt)
    grad_logt = torch.where(is_gt_max, grad_logt * quant_max, grad_logt)
    grad_logt = grad_logt.sum().expand_as(logt)

    # Equation (8)
    grad_x = grad_output.clone()
    grad_x = torch.where(is_ge_min_and_le_max, grad_x, 0 * grad_x)

    return grad_x, grad_logt, None, None

def _cdf_measure(x, y, measure_name='Kullback-Leibler-J'):
  """
    Ref paper:
    "Non-parametric Information-Theoretic Measures of One-Dimensional
    Distribution Functions from Continuous Time Series" - Paolo D Alberto et al.
    https://epubs.siam.org/doi/abs/10.1137/1.9781611972795.59
    https://epubs.siam.org/doi/pdf/10.1137/1.9781611972795.59

    measure_names_symm = ['Camberra', 'Chi-Squared', 'Cramer-von Mises', 'Euclidean',
               'Hellinger', 'Jin-L', 'Jensen-Shannon', 'Kolmogorov-Smirnov',
               'Kullback-Leibler-J', 'Variational']
    measure_names_asym = ['Jin-K', 'Kullback-Leibler-I']
    measure_names_excl = ['Bhattacharyya', 'Phi', 'Xi']
    """
  if measure_name == 'Bhattacharyya':
    return np.sum(np.sqrt(x * y))
  else:
    if measure_name == 'Camberra':
      return np.sum(np.abs(x - y) / (x + y))
    else:
      if measure_name == 'Chi-Squared':
        return np.sum(np.power(x - y, 2.0) / x)
      else:
        if measure_name == 'Cramer-von Mises':
          return np.sum(np.power(x - y, 2.0))
        else:
          if measure_name == 'Euclidean':
            return np.power(np.sum(np.power(x - y, 2.0)), 0.5)
          else:
            if measure_name == 'Hellinger':
              return np.power(np.sum(np.sqrt(x) - np.sqrt(y)), 2.0) / 2.0
            else:
              if measure_name == 'Jin-K':
                return _cdf_measure(x, (x + y) / 2.0, 'Kullback-Leibler-I')
              else:
                if measure_name == 'Jin-L':
                  return _cdf_measure(
                      x, (x + y) / 2.0, 'Kullback-Leibler-I') + _cdf_measure(
                          y, (x + y) / 2.0, 'Kullback-Leibler-I')
                if measure_name == 'Jensen-Shannon':
                  return (
                      _cdf_measure(x, (x + y) / 2.0, 'Kullback-Leibler-I') +
                      _cdf_measure(y,
                                   (x + y) / 2.0, 'Kullback-Leibler-I')) / 2.0
                if measure_name == 'Kolmogorov-Smirnov':
                  return np.max(np.abs(x - y))
              if measure_name == 'Kullback-Leibler-I':
                return np.sum(x * np.log2(x / y))
            if measure_name == 'Kullback-Leibler-J':
              return np.sum((x - y) * np.log2(x / y))
          if measure_name == 'Phi':
            return np.max(
                np.abs(x - y) /
                np.sqrt(np.minimum((x + y) / 2.0, 1 - (x + y) / 2.0)))
        if measure_name == 'Variational':
          return np.sum(np.abs(x - y))
      if measure_name == 'Xi':
        return np.max(
            np.abs(x - y) / np.sqrt((x + y) / 2.0 * (1 - (x + y) / 2.0)))
    return _cdf_measure(x, y, 'Kullback-Leibler-J')

class TQTQuantizer(FakeQuantizer):

  def __init__(self, bitwidth, tensor_type):
    super(TQTQuantizer, self).__init__(bitwidth)

    valid_tensor_types = ['weight', 'act']
    if tensor_type not in valid_tensor_types:
      raise ValueError(
          "'tensor_type' must be one of {}".format(valid_tensor_types))
    self.tensor_type = tensor_type

    # See TorchQuantizer::do_quantize() in quantization/torchquantizer.py
    self.method = 3 if tensor_type == 'weight' else 2
    self.quantize_fn_cls = TQTQuantize

    self.log_threshold = nn.Parameter(torch.tensor([0.0]))
    self.register_buffer('warmup_enabled', torch.tensor([1], dtype=torch.uint8))

  def _init_threshold(self, x):
    """See Table 2 in https://arxiv.org/pdf/1903.08066.pdf"""

    def _max(x):
      return np.max(np.abs(x))

    def _3sd(x):
      y = x.astype(np.float32) if x.dtype == np.float16 else x
      return np.abs(np.mean(y + 1e-6)) + 3 * np.std(y)

    def _kl_j(x):
      """
      Ref paper (Algorithm 1):
      "Quantizing Convolutional Neural Networks for Low-Power
      High-Throughput Inference Engines" - Sean Settle et al.
      https://arxiv.org/pdf/1805.07941.pdf
      """

      def calculate_kl_j(x, y):
        return np.sum((x - y) * np.log2(x / y))

      mn = 0
      mx = np.max(np.abs(x))
      y = x.astype(np.float32) if x.dtype == np.float16 else x
      hist, bin_edges = np.histogram((np.abs(y)),
                                     'sqrt',
                                     range=(mn, mx),
                                     density=True)
      hist = hist.astype(x.dtype)
      bin_edges = bin_edges.astype(x.dtype)
      pdf = hist / np.sum(hist)
      cdf = np.cumsum(pdf)
      n = pow(2, self.bitwidth.item() - 1)
      threshold = []
      d = []
      if n + 1 > len(bin_edges) - 1:
        return bin_edges[(-1)]
      else:
        for i in range(n + 1, len(bin_edges), 1):
          threshold_tmp = (i + 0.5) * (bin_edges[1] - bin_edges[0])
          threshold = np.concatenate((threshold, [threshold_tmp]))
          p = np.copy(cdf)
          p[i - 1:] = 1
          x = np.linspace(0.0, 1.0, n)
          xp = np.linspace(0.0, 1.0, i)
          fp = p[:i]
          p_interp = np.interp(x, xp, fp)
          x = np.linspace(0.0, 1.0, i)
          xp = np.linspace(0.0, 1.0, n)
          fp = p_interp
          q_interp = np.interp(x, xp, fp)
          q = np.copy(p)
          q[:i] = q_interp
          #d_tmp = _cdf_measure(cdf[np.nonzero(cdf)], q[np.nonzero(cdf)],
          #                     'Kullback-Leibler-J')
          d_tmp = calculate_kl_j(cdf[np.nonzero(cdf)], q[np.nonzero(cdf)])
          d = np.concatenate((d, [d_tmp]))

        return threshold[np.argmin(d)]

    init_scheme = {'weight': _3sd, 'act': _kl_j}
    #init_scheme = {'weight': _max, 'act': _kl_j}
    data = x.detach().cpu().numpy()
    th = init_scheme[self.tensor_type](data)
    # TODO(yuwang): Check if th < 0.
    return torch.tensor([th], dtype=x.dtype, device=x.device)

  def forward(self, x):
    if self.quant_enabled[0] == 0:
      return x

    if self.warmup_enabled[0] == 1:
      self.warmup_enabled[0] = 0
      threshold = self._init_threshold(x)
      self.log_threshold.data = torch.log2(threshold)

    return self.quantize_fn_cls.apply(x, self.log_threshold, self.domain,
                                      self.method)

  def enable_quant(self, enabled=True):
    self.quant_enabled[0] = 1 if enabled else 0
    return self

  def disable_quant(self):
    return self.enable_quant(False)

  def enable_warmup(self, enabled=True):
    self.warmup_enabled[0] = 1 if enabled else 0
    return self

  def disable_warmup(self):
    return self.enable_warmup(False)

  def freeze_quant(self, frozen=True):
    self.log_threshold.requires_grad = (not frozen)

  def unfreeze_quant(self):
    self.freeze_quant(False)

  def extra_repr(self):
    return 'quant_enabled={}, bitwidth={}, method={}'.format(
        self.quant_enabled, self.bitwidth, self.method)

  def _save_to_state_dict(self, destination, prefix, keep_vars):
    super(TQTQuantizer, self)._save_to_state_dict(destination, prefix,
                                                  keep_vars)

  def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                            missing_keys, unexpected_keys, error_msgs):
    super(TQTQuantizer,
          self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    #ignored_keys = ['']
    #for key in ignored_keys:
    #  missing_keys.remove(prefix + key)

  def export_quant_info(self):
    """Export trained threshold to TorchQuantizer's quant info [bitwidth, fp].

    (1) TQT: qx = clip(round(fx / scale)) * scale, scale = 2^ceil(log2t) / 2^(b-1)
    (2) NndctFixNeron: qx = clip(round(fx * scale)) * (1 / scale), scale = 2^fp
    Let (1) equals (2), we can get
    (3): 2^(b-1) / 2^ceil(log2t) = 2^fp
     => fp = b - 1 - ceil(log2t)

    For more details, see nndct/include/cuda/nndct_fix_kernels.cuh::_fix_neuron_v2_device
    """
    bitwidth = self.bitwidth.item()
    ceil_log2t = torch.ceil(self.log_threshold).item()
    return [[bitwidth, int(bitwidth - 1 - ceil_log2t)]]

  def import_quant_info(self, qinfo):
    bitwidth, fp = qinfo
    self.bitwidth[0] = bitwidth
    self.log_threshold.data = torch.tensor([bitwidth - 1 - fp],
                                           dtype=self.log_threshold.dtype)
    self.warmup_enabled[0] = 0

def enable_quant(mod):
  if isinstance(mod, FakeQuantizer):
    mod.enable_quant()

def disable_quant(mod):
  if isinstance(mod, FakeQuantizer):
    mod.disable_quant()

def enable_warmup(mod):
  if isinstance(mod, TQTQuantizer):
    mod.enable_warmup()

def disable_warmup(mod):
  if isinstance(mod, TQTQuantizer):
    mod.disable_warmup()

def freeze_quant(mod):
  if isinstance(mod, TQTQuantizer):
    mod.freeze_quant()

def unfreeze_quant(mod):
  if isinstance(mod, TQTQuantizer):
    mod.unfreeze_quant()
