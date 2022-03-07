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

"""Statistics utilities of model complexity."""

import torch
from torch import nn

from nndct_shared.utils import common

class MetricName(object):
  Flops = 'flops'
  TrainableParams = 'trainable'
  NonTrainableParams = 'non-trainable'

def _accumulate_metric_value(module, metric_name, value):
  setattr(module, metric_name, getattr(module, metric_name) + value)

def count_convNd(module, input, output):
  # kw x kh
  num_kernels = (module.weight.size()[2:]).numel()
  bias = 1 if module.bias is not None else 0

  # c_out x w x h  x (c_in x kw x kh + bias)
  single_batch_output = output[0]
  _accumulate_metric_value(module, MetricName.Flops,
          single_batch_output.numel() *
          (module.in_channels // module.groups * num_kernels + bias))

def count_linear(module, input, output):
  # (N, *, Hin) x (Hin, Hout) = (N, *, Hout)
  _accumulate_metric_value(module, MetricName.Flops,
          module.in_features * output[0].numel())

flops_counters = {
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.ConvTranspose1d: count_convNd,
    nn.ConvTranspose2d: count_convNd,
    nn.ConvTranspose3d: count_convNd,
    nn.Linear: count_linear,
}

class MetricHook(object):

  def __init__(self):
    self._module_hooks = []

  def register(self, module):
    raise NotImplementedError('Not implemented')

  def clear(self, module):
    pass

  @property
  def module_hooks(self):
    return self._module_hooks

  @property
  def value(self):
    raise NotImplementedError('Not implemented')

class FlopsMetric(MetricHook):

  def __init__(self):
    super(FlopsMetric, self).__init__()
    self.name = MetricName.Flops

  def register(self, module):
    handle = None
    fn = flops_counters.get(type(module), None)
    if fn:
      module.register_buffer(MetricName.Flops, torch.zeros(1, dtype=torch.int64))
      handle = module.register_forward_hook(fn)
      self._module_hooks.append((module, handle))
    return handle

  def clear(self, module):
    if type(module) in flops_counters:
      module._buffers.pop(MetricName.Flops)

  def value(self, module):
    if type(module) in flops_counters:
      return module._buffers.get(MetricName.Flops).item()
    return None

class HookedStat(object):

  def __init__(self):
    self._metrics = []
    self._module_hooks = {}

  def _register_metric(self, module):
    for metric in self._metrics:
      hook = metric.register(module)
      if not hook:
        continue
      if module not in self._module_hooks:
        self._module_hooks[module] = []
      self._module_hooks[module].append(hook)

  def clear_metrics(self):
    for module, hooks in self._module_hooks.items():
      for hook in hooks:
        hook.remove()

      for metric in self._metrics:
        metric.clear(module)
    self._metrics = []

  def _aggregate_metric_values(self):
    values = {}
    for idx, module in enumerate(self._module_hooks.keys()):
      class_name = str(module.__class__).split(".")[-1].split("'")[0]
      module_key = "%s-%i" % (class_name, idx + 1)
      metric_values = {
          metric.name: metric.value(module) for metric in self._metrics
      }
      values[module_key] = metric_values
    return values

  def add_metric(self, metric):
    self._metrics.append(metric)

  def run(self, model, inputs):
    model.apply(self._register_metric)

    if torch.cuda.is_available():
      model.cuda()
      if isinstance(inputs, (tuple, list)):
        inputs = [input.cuda() for input in inputs]
      else:
        inputs = [inputs.cuda()]
    model.eval()
    model(*inputs)

    values = self._aggregate_metric_values()
    self.clear_metrics()
    return values

def stat_flops(model, inputs):
  stat = HookedStat()
  stat.add_metric(FlopsMetric())
  metrics = stat.run(model, inputs)
  flops = {}
  # {module: {metric.name: metric.value}}
  for module, metric in metrics.items():
    flops[module] = metric[MetricName.Flops]
  return flops

def stat_params(model):
  params = {}

  for name, module in model.named_modules():
    if len(list(module.children())) > 0:
      continue

    params[name] = {}
    for p in module.parameters():
      key = (
          MetricName.TrainableParams
          if p.requires_grad else MetricName.NonTrainableParams)
      if key not in params[name]:
        params[name][key] = 0
      params[name][key] += p.numel()

  return params

def model_complexity(model, inputs, readable=False):
  """Stat the complexity of the given model. Currently includes flops and params.
  MACs: multiply–accumulate operations that performs a += b x c
  Params: total number of parameters of a model.

  Args:
    model: An `nn.Module` object.
    inputs: A list or tuple of inputs used to run forward passes on the model.
    readable: Whether to return readable numbers.

  Returns:
    Statistically obtained flops and params by given inputs.
  """
  flops = stat_flops(model, inputs)
  total_flops = 0
  for module, value in flops.items():
    total_flops += value

  total_params = 0
  params = stat_params(model)
  for name in params:
    if MetricName.TrainableParams in params[name]:
      total_params += params[name][MetricName.TrainableParams]

  if readable:
    total_flops = common.readable_num(total_flops)
    total_params = common.readable_num(total_params)
  return total_flops, total_params
