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

from torch import Tensor
from torch._ops import ops
from torch.nn import functional as F
from typing import List

import abc
import torch

# Adopted from https://github.com/pytorch/pytorch/blob/master/torch/nn/quantized/modules/functional_modules.py

class Functional(abc.ABC, torch.nn.Module):
  """State collector class for float operations.

  The instance of this class can be used instead of the ``torch.`` prefix for
  some operations. See example usage below."""

  def __init__(self):
    super(Functional, self).__init__()

  @abc.abstractmethod
  def forward(self, x):
    raise RuntimeError(
        "Functional is an abstract class, please use its derived classes.")

class Add(Functional):
  """Operation equivalent to ``torch.add(Tensor, Tensor)``"""

  def forward(self, input, other, alpha=1, out=None):
    # type: (Tensor, Tensor) -> Tensor
    if out is None:
      return torch.add(input, other, alpha=alpha)
    else:
      return torch.add(input, other, alpha=alpha, out=out)

class AddScalar(Functional):
  """Operation equivalent to ``torch.add(Tensor, float)``"""

  def forward(self, x, y):
    # type: (Tensor, float) -> Tensor
    return torch.add(x, y)

class Cat(Functional):
  """Operation equivalent to ``torch.cat``"""

  def forward(self, x, dim=0):
    # type: (List[Tensor], int) -> Tensor
    return torch.cat(x, dim=dim)

class Mul(Functional):
  """Operation equivalent to ``torch.mul(Tensor, Tensor)``"""

  def forward(self, x, y):
    # type: (Tensor, Tensor) -> Tensor
    return torch.mul(x, y)

class Matmul(Functional):
  """Operation equivalent to `torch.matmul(input, other, out=None)`"""

  def forward(self, input, other, out=None):
    return torch.matmul(input, other, out=out)

class Max(Functional):
  """Operation equivalent to ``torch.max(input, dim, keepdim=False, out=None)``"""

  def forward(self, input, dim, keepdim=False, out=None):
    if out is None:
      output = torch.max(input, dim, keepdim)
    else:
      output = torch.max(input, dim, keepdim, out)
    return output

class Sum(Functional):
  """Operation equivalent to `torch.sum(input, dim, keepdim=False, dtype=None)`"""

  def forward(self, input, dim, keepdim=False, dtype=None):
    #if not isinstance(dim, tuple):
    #  dim = (dim, )
    if dtype is not None:
      return torch.sum(input, dim, keepdim, dtype=dtype)
    else:
      return torch.sum(input, dim, keepdim)

class Interpolate(Functional):
  """Operation equivalent to``torch.nn.functional.interpolate``"""

  if torch.__version__ < '1.5.0':

    def forward(self,
                input,
                size=None,
                scale_factor=None,
                mode='nearest',
                align_corners=None):
      return F.interpolate(input, size, scale_factor, mode, align_corners)
  else:

    def forward(self,
                input,
                size=None,
                scale_factor=None,
                mode='nearest',
                align_corners=None,
                recompute_scale_factor=None):
      return F.interpolate(input, size, scale_factor, mode, align_corners,
                           recompute_scale_factor)

class Upsample(Functional):

  def forward(self,
              input,
              size=None,
              scale_factor=None,
              mode='nearest',
              align_corners=None):
    return F.upsample(input, size, scale_factor, mode, align_corners)

class Pad(Functional):

  def forward(self, input, pad, mode='constant', value=0):
    if mode != 'replicate':
      print((
          '[WARN] DPU only supports padding mode=replicate. '
          'Other modes of padding will be run on CPU, which will results in poor performance.'
      ))
    return F.pad(input, pad, mode, value)

class Mean(Functional):

  def forward(self, x, dim, keepdim=False, out=None):
    if out is not None:
      return torch.mean(x, dim, keepdim, out)
    return torch.mean(x, dim, keepdim)

class Softmax(Functional):
  """Operation equivalent to `torch.nn.functional.softmax(input, dim=None, dtype=None)`"""

  def forward(self, input, dim=None, dtype=None):
    return torch.nn.functional.softmax(input, dim, dtype)

class Clamp(Functional):
  """Operation equivalent to ``torch.clamp(input, min=None, max=None)``"""

  def forward(self, input, min=None, max=None):
    return torch.clamp(input, min, max)

  @property
  def is_quantized(self):
    return True

class Const(Functional):
  """Const Module """

  def __init__(self, value):
    super(Const, self).__init__()

    if not isinstance(value, torch.Tensor):
      value = torch.tensor(value)
    self.register_buffer('value', value)

  def forward(self):
    return self.value

