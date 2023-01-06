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
from pytorch_nndct.nn.quantization.ops import tqt_ops

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

class TQTQuantizer(FakeQuantizer):

  def __init__(self, bitwidth, tensor_type, method = None):
    super(TQTQuantizer, self).__init__(bitwidth)

    valid_tensor_types = ['weight', 'act']
    if tensor_type not in valid_tensor_types:
      raise ValueError(
          "'tensor_type' must be one of {}".format(valid_tensor_types))
    self.tensor_type = tensor_type

    # See TorchQuantizer::do_quantize() in quantization/torchquantizer.py
    if method is not None:
      self.method = method
    else:
      self.method = 3 if tensor_type == 'weight' else 2
    self.quantize_fn_cls = tqt_ops.TQTQuantize

    self.log_threshold = nn.Parameter(torch.tensor([0.0]))
    self.register_buffer('warmup_enabled', torch.tensor([1], dtype=torch.uint8))

    self._forward_fn = self._quantize_with_warmup

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
          d_tmp = calculate_kl_j(cdf[np.nonzero(cdf)], q[np.nonzero(cdf)])
          d = np.concatenate((d, [d_tmp]))

        return threshold[np.argmin(d)]

    init_scheme = {'weight': _3sd, 'act': _kl_j}
    #init_scheme = {'weight': _max, 'act': _kl_j}
    data = x.detach().cpu().numpy()
    th = init_scheme[self.tensor_type](data)
    # TODO(yuwang): Check if th < 0.
    return torch.tensor([th], dtype=x.dtype, device=x.device)

  def _forward_pass_input(self, x, log_threshold, domain, method):
    return x

  def _quantize(self, x, log_threshold, domain, method):
    return self.quantize_fn_cls.apply(x, log_threshold, domain,
                                      method)
  def _quantize_with_warmup(self, x, log_threshold, domain, method):
    self.disable_warmup()
    log_threshold.data = torch.log2(self._init_threshold(x))
    return self._quantize(x, log_threshold, domain, method)

  def forward(self, x):
    #if self.quant_enabled[0] == 0:
    #  return x

    #if self.warmup_enabled[0] == 1:
    #  self.warmup_enabled[0] = 0
    #  threshold = self._init_threshold(x)
    #  self.log_threshold.data = torch.log2(threshold)

    #return self.quantize_fn_cls.apply(x, self.log_threshold, self.domain,
    #                                  self.method)
    return self._forward_fn(x, self.log_threshold, self.domain, self.method)

  def enable_quant(self, enabled=True):
    self.quant_enabled[0] = 1 if enabled else 0

    if enabled:
      self._forward_fn = self._quantize_with_warmup if self.warmup_enabled[
          0] == 1 else self._quantize
    else:
      self._forward_fn = self._forward_pass_input
    return self

  def disable_quant(self):
    return self.enable_quant(False)

  def enable_warmup(self, enabled=True):
    self.warmup_enabled[0] = 1 if enabled else 0
    self._forward_fn = self._quantize_with_warmup if enabled else self._quantize
    return self

  def disable_warmup(self):
    return self.enable_warmup(False)

  def is_warmup_enabled(self):
    return self.warmup_enabled[0] == 1

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
    self._forward_fn = self._quantize_with_warmup if self.warmup_enabled[
        0] == 1 else self._quantize
    if self.quant_enabled[0] == 0:
      self._forward_fn = self._forward_pass_input

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
    self.disable_warmup()

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
