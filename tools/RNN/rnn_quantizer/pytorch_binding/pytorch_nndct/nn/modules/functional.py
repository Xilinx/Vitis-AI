

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

from typing import List
from torch import Tensor
from torch._ops import ops

import abc
import torch

# Adopted from https://github.com/pytorch/pytorch/blob/master/torch/nn/quantized/modules/functional_modules.py

class Functional(abc.ABC, torch.nn.Module):
  """State collector class for float operations.

    The instance of this class can be used instead of the ``torch.`` prefix for
    some operations. See example usage below.

    .. note::

        This class does not provide a ``forward`` hook. Instead, you must use
        one of the underlying functions (e.g. ``add``).

    Examples::

        >>> f_add = Functional()
        >>> a = torch.tensor(3.0)
        >>> b = torch.tensor(4.0)
        >>> f_add.add(a, b)  # Equivalent to ``torch.add(a, b)``

    Valid operation names:
        - add
        - cat
        - mul
        - add_relu
        - add_scalar
        - mul_scalar
    """

  def __init__(self):
    super(Functional, self).__init__()

  @abc.abstractmethod
  def forward(self, x):
    raise RuntimeError(
        "Functional is an abstract class, please use its derived classes.")


class Add(Functional):
  """Operation equivalent to ``torch.add(Tensor, Tensor)``"""
  def forward(self, x, y, alpha=1):
    # type: (Tensor, Tensor) -> Tensor
    r = torch.add(x, y)
    return r

  r"""Operation equivalent to ``torch.add(Tensor, float)``"""

  def add_scalar(self, x, y):
    # type: (Tensor, float) -> Tensor
    r = torch.add(x, y)
    r = self.activation_post_process(r)
    return r

  r"""Operation equivalent to ``torch.mul(Tensor, Tensor)``"""

  def mul(self, x, y):
    # type: (Tensor, Tensor) -> Tensor
    r = torch.mul(x, y)
    r = self.activation_post_process(r)
    return r

  r"""Operation equivalent to ``torch.mul(Tensor, float)``"""

  def mul_scalar(self, x, y):
    # type: (Tensor, float) -> Tensor
    r = torch.mul(x, y)
    r = self.activation_post_process(r)
    return r

  r"""Operation equivalent to ``torch.cat``"""

  def cat(self, x, dim=0):
    # type: (List[Tensor], int) -> Tensor
    r = torch.cat(x, dim=dim)
    r = self.activation_post_process(r)
    return r

  r"""Operation equivalent to ``relu(torch.add(x,y))``"""

  def add_relu(self, x, y):
    # type: (Tensor, Tensor) -> Tensor
    r = torch.add(x, y)
    r = torch.nn.functional.relu(r)
    r = self.activation_post_process(r)
    return r

class QFunctional(torch.nn.Module):
  r"""Wrapper class for quantized operations.

    The instance of this class can be used instead of the
    ``torch.ops.quantized`` prefix. See example usage below.

    .. note::

        This class does not provide a ``forward`` hook. Instead, you must use
        one of the underlying functions (e.g. ``add``).

    Examples::

        >>> q_add = QFunctional()
        >>> a = torch.quantize_per_tensor(torch.tensor(3.0), 1.0, 0, torch.qint32)
        >>> b = torch.quantize_per_tensor(torch.tensor(4.0), 1.0, 0, torch.qint32)
        >>> q_add.add(a, b)  # Equivalent to ``torch.ops.quantized.add(a, b, 1.0, 0)``

    Valid operation names:
        - add
        - cat
        - mul
        - add_relu
        - add_scalar
        - mul_scalar
    """

  def __init__(self):
    super(QFunctional, self).__init__()
    self.scale = 1.0
    self.zero_point = 0
    self.activation_post_process = torch.nn.Identity()

  def _save_to_state_dict(self, destination, prefix, keep_vars):
    super(QFunctional, self)._save_to_state_dict(destination, prefix, keep_vars)
    destination[prefix + 'scale'] = torch.tensor(self.scale)
    destination[prefix + 'zero_point'] = torch.tensor(self.zero_point)

  def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                            missing_keys, unexpected_keys, error_msgs):

    self.scale = float(state_dict.pop(prefix + 'scale'))
    self.zero_point = int(state_dict.pop(prefix + 'zero_point'))
    super(QFunctional,
          self)._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                      missing_keys, unexpected_keys, error_msgs)

  def _get_name(self):
    return 'QFunctional'

  def extra_repr(self):
    return 'scale={}, zero_point={}'.format(self.scale, self.zero_point)

  def forward(self, x):
    raise RuntimeError("Functional is not intended to use the " +
                       "'forward'. Please use the underlying operation")

  r"""Operation equivalent to ``torch.ops.quantized.add``"""

  def add(self, x, y):
    # type: (Tensor, Tensor) -> Tensor
    r = ops.quantized.add(x, y, scale=self.scale, zero_point=self.zero_point)
    r = self.activation_post_process(r)
    return r

  r"""Operation equivalent to ``torch.ops.quantized.add(Tensor, float)``"""

  def add_scalar(self, x, y):
    # type: (Tensor, float) -> Tensor
    r = ops.quantized.add_scalar(x, y)
    r = self.activation_post_process(r)
    return r

  r"""Operation equivalent to ``torch.ops.quantized.mul(Tensor, Tensor)``"""

  def mul(self, x, y):
    # type: (Tensor, Tensor) -> Tensor
    r = ops.quantized.mul(x, y, scale=self.scale, zero_point=self.zero_point)
    r = self.activation_post_process(r)
    return r

  r"""Operation equivalent to ``torch.ops.quantized.mul(Tensor, float)``"""

  def mul_scalar(self, x, y):
    # type: (Tensor, float) -> Tensor
    r = ops.quantized.mul_scalar(x, y)
    r = self.activation_post_process(r)
    return r

  r"""Operation equivalent to ``torch.ops.quantized.cat``"""

  def cat(self, x, dim=0):
    # type: (List[Tensor], int) -> Tensor
    r = ops.quantized.cat(
        x, scale=self.scale, zero_point=self.zero_point, dim=dim)
    r = self.activation_post_process(r)
    return r

  r"""Operation equivalent to ``torch.ops.quantized.add_relu``"""

  def add_relu(self, x, y):
    # type: (Tensor, Tensor) -> Tensor
    r = ops.quantized.add_relu(
        x, y, scale=self.scale, zero_point=self.zero_point)
    r = self.activation_post_process(r)
    return r

  @classmethod
  def from_float(cls, mod):
    assert type(mod) == Functional,\
        "QFunctional.from_float expects an instance of Functional"
    scale, zero_point = mod.activation_post_process.calculate_qparams()
    new_mod = QFunctional()
    new_mod.scale = float(scale)
    new_mod.zero_point = int(zero_point)
    return new_mod
