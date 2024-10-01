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

from pytorch_nndct.nn.modules import fix_ops

class TQTQuantize(torch.autograd.Function):
  """Trained Quantization Thresholds.

  See https://arxiv.org/pdf/1903.08066.pdf
  """

  @staticmethod
  def forward(ctx, x, logt, domain, method):
    #scale = torch.pow(torch.tensor(2.0, device=x.device), torch.ceil(logt)) / domain
    #scale = torch.pow(2.0, torch.ceil(logt)) / domain
    scale = 2**(torch.ceil(logt)) / domain
    quant_max = domain - 1
    quant_min = -domain

    ctx.save_for_backward(x, scale, quant_max, quant_min, logt)

    x = x.clone()
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
