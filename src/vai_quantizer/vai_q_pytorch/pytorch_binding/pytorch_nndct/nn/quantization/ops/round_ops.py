# Copyright 2022 Xilinx Inc.
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

import torch

class RoundEven(torch.autograd.Function):

  @staticmethod
  def forward(ctx, t):
    ctx.save_for_backward(t)
    t = torch.round(t)
    return t

  @staticmethod
  def backward(ctx, grad_output):
    t, = ctx.saved_tensors
    grad_input = grad_output.clone()
    return grad_input, None

class RoundAway(torch.autograd.Function):

  @staticmethod
  def forward(ctx, t):
    ctx.save_for_backward(t)
    t = torch.where(
        torch.eq(t - torch.floor(t), 0.5), torch.trunc(t + torch.sign(t)),
        torch.round(t))

    return t

  @staticmethod
  def backward(ctx, grad_output):
    t, = ctx.saved_tensors
    grad_input = grad_output.clone()
    return grad_input, None

class RoundCeil(torch.autograd.Function):

  @staticmethod
  def forward(ctx, t):
    ctx.save_for_backward(t)
    t = torch.where(
        torch.eq(t - torch.floor(t), 0.5), torch.ceil(t), torch.round(t))

    return t

  @staticmethod
  def backward(ctx, grad_output):
    t, = ctx.saved_tensors
    grad_input = grad_output.clone()
    return grad_input, None

class Trunc(torch.autograd.Function):

  @staticmethod
  def forward(ctx, t):
    ctx.save_for_backward(t)
    t = torch.trunc(t)
    return t

  @staticmethod
  def backward(ctx, grad_output):
    t, = ctx.saved_tensors
    grad_input = grad_output.clone()
    return grad_input, None

class Floor(torch.autograd.Function):

  @staticmethod
  def forward(ctx, t):
    ctx.save_for_backward(t)
    t = torch.floor(t)
    return t

  @staticmethod
  def backward(ctx, grad_output):
    t, = ctx.saved_tensors
    grad_input = grad_output.clone()
    return grad_input, None

round_even = RoundEven.apply
round_away = RoundAway.apply
round_ceil = RoundCeil.apply
trunc = Trunc.apply
floor = Floor.apply

def get(identifier):
  """Returns rounding function.

  Args:
      identifier: String identifier.

  Returns:
      Function corresponding to the input string.

  For example:

  >>> round.get('round_even')
   <function round_even at 0x1222a3d90>

  Raises:
      ValueError: Input is an unknown string.
      TypeError: If input is not string.
  """
  if identifier is None:
    return linear

  if isinstance(identifier, str):
    globs = globals()
    if identifier not in globs:
      raise ValueError('Unknown rouding function: {}'.format(identifier))
    return globs[identifier]
  else:
    raise TypeError(
        'Could not interpret rounding function identifier: {}'.format(
            identifier))
