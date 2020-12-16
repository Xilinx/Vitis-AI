
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

def build(optimizer_config, optimizer, last_step=-1):
  """Create lr scheduler based on config. note that
  lr_scheduler must accept a optimizer that has been restored.

  Args:
    optimizer_config: A Optimizer proto message.

  Returns:
    An optimizer and a list of variables for summary.

  Raises:
    ValueError: when using an unsupported input data type.
  """
  optimizer_type = optimizer_config.WhichOneof('optimizer')

  if optimizer_type == 'rms_prop_optimizer':
    config = optimizer_config.rms_prop_optimizer
    lr_scheduler = _create_learning_rate_scheduler(
      config.learning_rate, optimizer, last_step=last_step)

  if optimizer_type == 'momentum_optimizer':
    config = optimizer_config.momentum_optimizer
    lr_scheduler = _create_learning_rate_scheduler(
      config.learning_rate, optimizer, last_step=last_step)

  if optimizer_type == 'adam_optimizer':
    config = optimizer_config.adam_optimizer
    lr_scheduler = _create_learning_rate_scheduler(
      config.learning_rate, optimizer, last_step=last_step)

  return lr_scheduler

def _create_learning_rate_scheduler(learning_rate_config, optimizer, last_step=-1):
  """Create optimizer learning rate scheduler based on config.

  Args:
    learning_rate_config: A LearningRate proto message.

  Returns:
    A learning rate.

  Raises:
    ValueError: when using an unsupported input data type.
  """
  lr_scheduler = None
  learning_rate_type = learning_rate_config.WhichOneof('learning_rate')
  if learning_rate_type == 'constant_learning_rate':
    config = learning_rate_config.constant_learning_rate
    lr_scheduler = learning_schedules.Constant(
      optimizer, last_step=last_step)

  if learning_rate_type == 'exponential_decay_learning_rate':
    config = learning_rate_config.exponential_decay_learning_rate
    lr_scheduler = learning_schedules.ExponentialDecay(
      optimizer, config.decay_steps, 
      config.decay_factor, config.staircase, last_step=last_step)

  if learning_rate_type == 'manual_step_learning_rate':
    config = learning_rate_config.manual_step_learning_rate
    if not config.schedule:
      raise ValueError('Empty learning rate schedule.')
    learning_rate_step_boundaries = [x.step for x in config.schedule]
    learning_rate_sequence = [config.initial_learning_rate]
    learning_rate_sequence += [x.learning_rate for x in config.schedule]
    lr_scheduler = learning_schedules.ManualStepping(
      optimizer, learning_rate_step_boundaries, learning_rate_sequence, 
      last_step=last_step)

  if learning_rate_type == 'cosine_decay_learning_rate':
    config = learning_rate_config.cosine_decay_learning_rate
    lr_scheduler = learning_schedules.CosineDecayWithWarmup(
      optimizer, config.total_steps, 
      config.warmup_learning_rate, config.warmup_steps, 
      last_step=last_step)

  if lr_scheduler is None:
    raise ValueError('Learning_rate %s not supported.' % learning_rate_type)

  return lr_scheduler