
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

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Functions to build DetectionModel training optimizers."""

from torchplus.train import learning_schedules
import torch


def build(optimizer_config, params, name=None):
    """Create optimizer based on config.

  Args:
    optimizer_config: A Optimizer proto message.

  Returns:
    An optimizer and a list of variables for summary.

  Raises:
    ValueError: when using an unsupported input data type.
  """
    optimizer_type = optimizer_config.WhichOneof('optimizer')
    optimizer = None

    if optimizer_type == 'rms_prop_optimizer':
        config = optimizer_config.rms_prop_optimizer
        optimizer = torch.optim.RMSprop(
            params,
            lr=_get_base_lr_by_lr_scheduler(config.learning_rate),
            alpha=config.decay,
            momentum=config.momentum_optimizer_value,
            eps=config.epsilon,
            weight_decay=config.weight_decay)

    if optimizer_type == 'momentum_optimizer':
        config = optimizer_config.momentum_optimizer
        optimizer = torch.optim.SGD(
            params,
            lr=_get_base_lr_by_lr_scheduler(config.learning_rate),
            momentum=config.momentum_optimizer_value,
            weight_decay=config.weight_decay)

    if optimizer_type == 'adam_optimizer':
        config = optimizer_config.adam_optimizer
        optimizer = torch.optim.Adam(
            params,
            lr=_get_base_lr_by_lr_scheduler(config.learning_rate),
            weight_decay=config.weight_decay)

    if optimizer is None:
        raise ValueError('Optimizer %s not supported.' % optimizer_type)

    if optimizer_config.use_moving_average:
        raise ValueError('torch don\'t support moving average')
    if name is None:
        # assign a name to optimizer for checkpoint system
        optimizer.name = optimizer_type
    else:
        optimizer.name = name
    return optimizer


def _get_base_lr_by_lr_scheduler(learning_rate_config):
    base_lr = None
    learning_rate_type = learning_rate_config.WhichOneof('learning_rate')
    if learning_rate_type == 'constant_learning_rate':
        config = learning_rate_config.constant_learning_rate
        base_lr = config.learning_rate

    if learning_rate_type == 'exponential_decay_learning_rate':
        config = learning_rate_config.exponential_decay_learning_rate
        base_lr = config.initial_learning_rate

    if learning_rate_type == 'manual_step_learning_rate':
        config = learning_rate_config.manual_step_learning_rate
        base_lr = config.initial_learning_rate
        if not config.schedule:
            raise ValueError('Empty learning rate schedule.')

    if learning_rate_type == 'cosine_decay_learning_rate':
        config = learning_rate_config.cosine_decay_learning_rate
        base_lr = config.learning_rate_base
    if base_lr is None:
        raise ValueError(
            'Learning_rate %s not supported.' % learning_rate_type)

    return base_lr
