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
  MACs = 'MACs'
  Flops = 'Flops'
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
  MACs = single_batch_output.numel() * (module.in_channels // module.groups * num_kernels)
  Flops = 2 * MACs + bias * single_batch_output.numel()
  _accumulate_metric_value(module, MetricName.MACs, MACs)
  _accumulate_metric_value(module, MetricName.Flops, Flops)


def count_linear(module, input, output):
  # (N, *, Hin) x (Hin, Hout) = (N, *, Hout)
  bias = 1 if module.bias is not None else 0

  MACs = module.in_features * output[0].numel()
  Flops = 2 * MACs + bias * output.shape[-1]
  _accumulate_metric_value(module, MetricName.MACs, MACs)
  _accumulate_metric_value(module, MetricName.Flops, Flops)

def zero_ops(module, input, output):
  MACs = 0
  Flops = 2 * MACs
  _accumulate_metric_value(module, MetricName.MACs, MACs)
  _accumulate_metric_value(module, MetricName.Flops, Flops)

def count_normalization(module, input, output):
  # y = (x - mean) / sqrt(eps + var) * weight + bias
  MACs = 2*input[0].numel()
  Flops = 2 * MACs
  _accumulate_metric_value(module, MetricName.MACs, MACs)
  _accumulate_metric_value(module, MetricName.Flops, Flops)

def count_relu(module, input, output):
  MACs = 0
  Flops = input[0].numel()
  _accumulate_metric_value(module, MetricName.MACs, MACs)
  _accumulate_metric_value(module, MetricName.Flops, Flops)

def count_prelu(module, input, output):
  MACs = input[0].numel()
  Flops = 2 * MACs
  _accumulate_metric_value(module, MetricName.MACs, MACs)
  _accumulate_metric_value(module, MetricName.Flops, Flops)

def count_softmax(module, input, output):
  nfeatures = input[0].size()[module.dim]
  total_exp = nfeatures
  total_add = nfeatures - 1
  total_div = nfeatures
  total_ops = total_exp + total_div
  MACs = total_ops
  Flops = MACs + total_add
  _accumulate_metric_value(module, MetricName.MACs, MACs)
  _accumulate_metric_value(module, MetricName.Flops, Flops)

def count_sigmoid(module, input, output):
  MACs = 0
  Flops = 4 * input[0].numel()
  _accumulate_metric_value(module, MetricName.MACs, MACs)
  _accumulate_metric_value(module, MetricName.Flops, Flops)

def count_pool(module, input, output):
  MACs = 0
  Flops = output.numel()
  _accumulate_metric_value(module, MetricName.MACs, MACs)
  _accumulate_metric_value(module, MetricName.Flops, Flops)


def count_adap_pool(module, input, output):
  kernel = torch.div(
      torch.DoubleTensor([*(input[0].shape[2:])]), 
      torch.DoubleTensor([*(output.shape[2:])])
  )
  total_add = torch.prod(kernel)
  num_elements = output.numel()
  total_div = 1
  kernel_op = total_add + total_div
  total_ops = int(kernel_op * num_elements)
  MACs = 0
  Flops = total_ops
  _accumulate_metric_value(module, MetricName.MACs, MACs)
  _accumulate_metric_value(module, MetricName.Flops, Flops)

def count_upsample(module, input, output):
  if module.mode not in (
      "nearest",
      "linear",
      "bilinear",
      "bicubic",
  ):  # "trilinear"
    logging.warning("mode %s is not implemented yet, take it a zero op" % m.mode)
    MACs = 0
    Flops = 0
    _accumulate_metric_value(module, MetricName.MACs, MACs)
    _accumulate_metric_value(module, MetricName.Flops, Flops)
  else:
    total_ops = output.nelement()
    if module.mode == "linear":
      total_ops *= 5
    elif module.mode == "bilinear":
      total_ops *= 11
    elif module.mode == "bicubic":
      ops_solve_A = 224  # 128 muls + 96 adds
      ops_solve_p = 35  # 16 muls + 12 adds + 4 muls + 3 adds
      total_ops *= ops_solve_A + ops_solve_p
    elif module.mode == "trilinear":
      total_ops *= 13 * 2 + 5
    MACs = 0
    Flops = total_ops
    _accumulate_metric_value(module, MetricName.MACs, MACs)
    _accumulate_metric_value(module, MetricName.Flops, Flops)

macs_counters = {
    nn.ZeroPad2d: zero_ops,  # padding does not involve any multiplication.
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.ConvTranspose1d: count_convNd,
    nn.ConvTranspose2d: count_convNd,
    nn.ConvTranspose3d: count_convNd,
    nn.BatchNorm1d: count_normalization,
    nn.BatchNorm2d: count_normalization,
    nn.BatchNorm3d: count_normalization,
    nn.LayerNorm: count_normalization,
    nn.InstanceNorm1d: count_normalization,
    nn.InstanceNorm2d: count_normalization,
    nn.InstanceNorm3d: count_normalization,
    nn.PReLU: count_prelu,
    nn.Softmax: count_softmax,
    nn.Sigmoid: count_sigmoid,
    nn.ReLU: count_relu,
    nn.ReLU6: count_relu,
    nn.LeakyReLU: count_relu,
    nn.MaxPool1d: count_pool,
    nn.MaxPool2d: count_pool,
    nn.MaxPool3d: count_pool,
    nn.AdaptiveMaxPool1d: count_adap_pool,
    nn.AdaptiveMaxPool2d: count_adap_pool,
    nn.AdaptiveMaxPool3d: count_adap_pool,
    nn.AvgPool1d: count_pool,
    nn.AvgPool2d: count_pool,
    nn.AvgPool3d: count_pool,
    nn.AdaptiveAvgPool1d: count_adap_pool,
    nn.AdaptiveAvgPool2d: count_adap_pool,
    nn.AdaptiveAvgPool3d: count_adap_pool,
    nn.Linear: count_linear,
    nn.Dropout: zero_ops,
    nn.Upsample: count_upsample,
    nn.UpsamplingBilinear2d: count_upsample,
    nn.UpsamplingNearest2d: count_upsample,
    nn.Sequential: zero_ops,
    nn.PixelShuffle: zero_ops,
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

class macsMetric(MetricHook):

  def __init__(self):
    super(macsMetric, self).__init__()

  def register(self, module):
    handle = None
    fn = macs_counters.get(type(module), None)
    if fn:
      module.register_buffer(MetricName.MACs, torch.zeros(1, dtype=torch.int64))
      module.register_buffer(MetricName.Flops, torch.zeros(1, dtype=torch.int64))
      handle = module.register_forward_hook(fn)
      self._module_hooks.append((module, handle))
    return handle

  def clear(self, module):
    if type(module) in macs_counters:
      module._buffers.pop(MetricName.MACs)
      module._buffers.pop(MetricName.Flops)

  def value(self, module):
    if type(module) in macs_counters:
      return [module._buffers.get(MetricName.MACs).item(), module._buffers.get(MetricName.Flops).item()]
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
          'MACs and Flops': metric.value(module) for metric in self._metrics
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

def stat_macs(model, inputs):
  stat = HookedStat()
  stat.add_metric(macsMetric())
  metrics = stat.run(model, inputs)

  model_analysis_info = {}

  for module, metric in metrics.items():
    class_name = module.split("-")[0]
    if class_name not in model_analysis_info:
      model_analysis_info[class_name] = {'number':1 , 'MACs':metric['MACs and Flops'][0], 'Flops':metric['MACs and Flops'][1]}
    else:
      model_analysis_info[class_name]['number'] += 1 
      model_analysis_info[class_name]['MACs'] += metric['MACs and Flops'][0] 
      model_analysis_info[class_name]['Flops'] += metric['MACs and Flops'][1] 

  sorted(model_analysis_info.items(), key = lambda x:x[0][2] ,reverse = True)
  return metrics, model_analysis_info

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

def model_complexity(model, inputs, return_flops=False, readable=False, print_model_analysis=False):
  """Stat the complexity of the given model. Currently includes macs and params.
  MACs: multiply–accumulate operations that performs a += b x c
  Flops = 2*MACs + BiasAdd
  Params: total number of parameters of a model.

  Args:
    model: An `nn.Module` object.
    inputs: A list or tuple of inputs used to run forward passes on the model.
    readable: Whether to return readable numbers.

  Returns:
    Statistically obtained macs and params by given inputs.
  """
  macs, model_analysis_info = stat_macs(model, inputs)
  total_macs = 0
  total_flops = 0

  for module, value in macs.items():
    total_macs += value['MACs and Flops'][0]
    total_flops += value['MACs and Flops'][1]

  if print_model_analysis:
    header_fields = ['Module Name', 'Number', 'MACs', 'Flops']
    rows_fields = []
    for module, value in model_analysis_info.items():
      rows_fields.append([module, value['number'], value['MACs'], value['Flops']])
    spu.print_summary(header_fields, rows_fields)
    logging.info("Total multiply–accumulate operations (MACs) : {0}".format(total_macs))
    logging.info("Total floating point operations (Flops) : {0}".format(total_flops))
  
  if return_flops:
    total_macs = total_flops

  total_params = 0
  params = stat_params(model)
  for name in params:
    if MetricName.TrainableParams in params[name]:
      total_params += params[name][MetricName.TrainableParams]

  if readable:
    total_macs = common.readable_num(total_macs)
    total_params = common.readable_num(total_params)
  return total_macs, total_params
