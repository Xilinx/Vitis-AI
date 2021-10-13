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
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_nndct.nn.modules import fix_ops
from torch.autograd import Variable

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
    grad_x = torch.where(
        is_ge_min_and_le_max, grad_x, 0 * grad_x)

    return grad_x, grad_logt, None, None

class TQTWarmup(torch.autograd.Function):
  """Warmup step for initializing log threshold."""

  @staticmethod
  def forward(ctx, x, logt, lr, num_devices):
    max_x = torch.max(torch.abs(x))
    ctx.save_for_backward(max_x, lr, num_devices)
    return x

  @staticmethod
  def backward(ctx, grad_output):
    max_x, lr, num_devices = ctx.saved_tensors
    grad_logt = torch.tensor([torch.log(max_x) / math.log(2)],
        dtype=max_x.dtype, device=max_x.device)
    grad_logt = grad_logt / (-lr * num_devices)
    return grad_output, grad_logt, None, None

class FakeQuantizer(nn.Module):
  """Simulate the quantize and dequantize operations in training time.

  In general, the output of this module is given by
  x_out = (clamp(round(x/scale + zero_point), quant_min, quant_max)-zero_point)*scale
  See https://arxiv.org/pdf/1903.08066.pdf

  In nndct, we use symmetric quantization and power-of-2 scaling. That is,
    zero_point = 0,
    quant_min = -2^(num_bits - 1),
    quant_max = 2^(num_bits - 1) - 1
  """

  def __init__(self, num_bits):
    super(FakeQuantizer, self).__init__()
    # quant_enabled is registered as buffer to support their replication in DDP.
    # Data type is uint8 because NCCL does not support bool tensors.
    self.register_buffer('quant_enabled', torch.tensor([1], dtype=torch.uint8))
    self.register_buffer('num_bits', torch.tensor([num_bits],
                                                  dtype=torch.uint8))
    self.register_buffer('domain', torch.tensor([2**(num_bits-1)]).float())

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
    # We save 'num_bits' to state_dict but not load it.
    num_bits_key = prefix + 'num_bits'
    if num_bits_key not in state_dict:
      raise ValueError(('Missing key "{}" in state dict, '
          'please make sure the loaded weights was saved from quantized '
          'model').format(num_bits_key))

    state_dict.pop(num_bits_key)
    super(FakeQuantizer,
          self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    ignored_params = ['num_bits', 'quant_enabled', 'domain']
    ignored_keys = [prefix + name for name in ignored_params]
    for key in ignored_keys:
      if key in missing_keys:
        missing_keys.remove(key)
      else:
        print('[WARN] Unexpected key in state dict:', key)

class TQTQuantizer(FakeQuantizer):

  def __init__(self, num_bits, tensor_type):
    super(TQTQuantizer, self).__init__(num_bits)

    if tensor_type not in ['param', 'blob']:
      raise ValueError("'tensor_type' must be one of ['param', 'blob']")
    # See TorchQuantizer::do_quantize() in quantization/torchquantizer.py
    self.method = 3 if tensor_type == 'param' else 2

    self.quantize_fn_cls = TQTQuantize

    self.log_threshold = nn.Parameter(torch.tensor([0.0]))
    self.register_buffer('warmup_enabled', torch.tensor([1], dtype=torch.uint8))

  def forward(self, x):
    if self.quant_enabled[0] == 0:
      return x

    if self.training and self.warmup_enabled[0] == 1:
      self.warmup_enabled[0] = 0
      max_x = torch.tensor([torch.max(torch.abs(x))],
          dtype=x.dtype, device=x.device)
      self.log_threshold.data = torch.log(max_x) / math.log(2)
    else:
      x = self.quantize_fn_cls.apply(
          x, self.log_threshold, self.domain, self.method)
    return x

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

  def extra_repr(self):
    return 'quant_enabled={}, num_bits={}, method={}'.format(
        self.quant_enabled, self.num_bits, self.method)

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
    """Generate quant info as TorchQuantizer's format, which is [num_bits, fp].
    (1) TQT: qx = clip(round(fx / scale)) * scale, scale = 2^ceil(log2t) / 2^(b-1)
    (2) NndctFixNeron: qx = clip(round(fx * scale)) * (1 / scale), scale = 2^fp
    Let (1) equals (2), we can get
    (3): 2^(b-1) / 2^ceil(log2t) = 2^fp
     => fp = b - 1 - ceil(log2t)
    For more details, see nndct/include/cuda/nndct_fix_kernels.cuh::_fix_neuron_v2_device
    """
    num_bits = self.num_bits.item()
    logt_ceil = torch.ceil(self.log_threshold).item()
    return [num_bits, int(num_bits - 1 - logt_ceil)]

  def import_quant_info(self, qinfo):
    num_bits, fp = qinfo
    self.num_bits[0] = num_bits
    self.log_threshold.data = torch.tensor([num_bits - 1 - fp], dtype=self.log_threshold.dtype)
    self.warmup_enabled[0] = 0

def enable_quant(mod):
  if isinstance(mod, FakeQuantizer):
    mod.enable_quant()

def disable_quant(mod):
  if isinstance(mod, FakeQuantizer):
    mod.disable_quant()

def enable_warmup(mod):
  if isinstance(mod, FakeQuantizer):
    mod.enable_warmup()

def disable_warmup(mod):
  if isinstance(mod, FakeQuantizer):
    mod.disable_warmup()
