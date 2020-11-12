
# This code is based on: https://github.com/nutonomy/second.pytorch.git
# 
# MIT License
# Copyright (c) 2018 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from collections import defaultdict, Iterable

import torch
from copy import deepcopy
from itertools import chain
from torch.autograd import Variable

required = object()

def param_fp32_copy(params):
    param_copy = [
        param.clone().type(torch.cuda.FloatTensor).detach() for param in params
    ]
    for param in param_copy:
        param.requires_grad = True
    return param_copy

def set_grad(params, params_with_grad, scale=1.0):
    for param, param_w_grad in zip(params, params_with_grad):
        if param.grad is None:
            param.grad = torch.nn.Parameter(
                param.data.new().resize_(*param.data.size()))
        grad = param_w_grad.grad.data
        if scale is not None:
            grad /= scale
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            return True # invalid grad
        param.grad.data.copy_(grad)
    return False

class MixedPrecisionWrapper(object):
    """mixed precision optimizer wrapper.
    Arguments:
        optimizer (torch.optim.Optimizer): an instance of 
            :class:`torch.optim.Optimizer`
        scale: (float): a scalar for grad scale.
        auto_scale: (bool): whether enable auto scale.
            The algorihm of auto scale is discribled in 
            http://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
    """

    def __init__(self,
                 optimizer,
                 scale=None,
                 auto_scale=True,
                 inc_factor=2.0,
                 dec_factor=0.5,
                 num_iters_be_stable=500):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError("must provide a torch.optim.Optimizer")
        self.optimizer = optimizer
        if hasattr(self.optimizer, 'name'):
            self.name = self.optimizer.name  # for ckpt system
        param_groups_copy = []
        for i, group in enumerate(optimizer.param_groups):
            group_copy = {n: v for n, v in group.items() if n != 'params'}
            group_copy['params'] = param_fp32_copy(group['params'])
            param_groups_copy.append(group_copy)

        # switch param_groups, may be dangerous
        self.param_groups = optimizer.param_groups
        optimizer.param_groups = param_groups_copy
        self.grad_scale = scale
        self.auto_scale = auto_scale
        self.inc_factor = inc_factor
        self.dec_factor = dec_factor
        self.stable_iter_count = 0
        self.num_iters_be_stable = num_iters_be_stable

    def __getstate__(self):
        return self.optimizer.__getstate__()

    def __setstate__(self, state):
        return self.optimizer.__setstate__(state)

    def __repr__(self):
        return self.optimizer.__repr__()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self.optimizer.load_state_dict(state_dict)

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def step(self, closure=None):
        for g, g_copy in zip(self.param_groups, self.optimizer.param_groups):
            invalid = set_grad(g_copy['params'], g['params'], self.grad_scale)
            if invalid:
                if self.grad_scale is None or self.auto_scale is False:
                    raise ValueError("nan/inf detected but auto_scale disabled.")
                self.grad_scale *= self.dec_factor
                print('scale decay to {}'.format(self.grad_scale))
                return
        if self.auto_scale is True:
            self.stable_iter_count += 1
            if self.stable_iter_count > self.num_iters_be_stable:
                if self.grad_scale is not None:
                    self.grad_scale *= self.inc_factor
                self.stable_iter_count = 0

        if closure is None:
            self.optimizer.step()
        else:
            self.optimizer.step(closure)
        for g, g_copy in zip(self.param_groups, self.optimizer.param_groups):
            for p_copy, p in zip(g_copy['params'], g['params']):
                p.data.copy_(p_copy.data)

