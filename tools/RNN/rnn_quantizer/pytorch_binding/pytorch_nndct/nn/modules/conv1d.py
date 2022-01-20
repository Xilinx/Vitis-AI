

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

import torch
from torch.autograd import Variable
import math

from nndct_shared.utils import NndctOption
from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import process_inputs_and_params
from nndct_shared.quantization import post_quant_process
import pytorch_nndct.utils as py_utils
import torch.nn.functional as F
__all__ = ['Conv1d']

class deephi_Conv1d(torch.nn.modules.conv.Conv1d):
  r"""DeePhi Conv1d operation, support float and double"""

  def __init__(self, *args, **kwards):
    super(deephi_Conv1d, self).__init__(*args, **kwards)
    self.params_name = None
    self.node = None
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.param_saved = False
    self.param_quantized = False

    self.weight_f = None
    self.bias_f = None
    self.err = None
    self.stop = False
    self.efficency = 0.0
    self.deviation = 0.0

  def forward(self, input):
    if self.bias is not None:
      params = [self.weight, self.bias]
    else:
      params = [self.weight]

    [input], __ = process_inputs_and_params(
        self.node,
        self.quantizer,
        inputs=[input],
        params=[],
        param_names=[])
    if (not self.param_saved):
      if NndctOption.nndct_param_corr.value > 0:
        # backup orignal float parameters
        if self.quant_mode == 1:
          self.weight_f = self.weight.detach().clone()
          if self.bias is not None:
            self.bias_f = self.bias.detach().clone()
        # adjust bias
        if self.quant_mode == 2:
          if (self.bias is not None):
            self.bias.data = torch.sub(self.bias.data, torch.tensor(
                self.quantizer.bias_corr[self.node.name],
                device=self.bias.data.device))
      self.param_saved = True
    #print('deephi_conv1d forward:', self.node.name)
    #print('float weight & bias:', self.weight.sum(), self.bias.sum() if self.bias is not None else None)
    if (not self.param_quantized):
      # quantize weights and bias
      __, __ = process_inputs_and_params(
          self.node,
          self.quantizer,
          inputs=[],
          params=params,
          param_names=self.params_name)
      self.param_quantized = True

    #print('quantized weight & bias:', self.weight.sum(), self.bias.sum() if self.bias is not None else None)
    output = super().forward(input)

    # quantize output
    [output] = post_quant_process(self.node, [output])

    # correct weights and bias in calibation
    if NndctOption.nndct_param_corr.value > 0:
      rate = NndctOption.nndct_param_corr_rate.value
      # statistic of quantization error
      if (self.quant_mode == 1 and not self.stop):
        res_f = torch.nn.functional.conv1d(input,
                                           self.weight_f,
                                           bias = self.bias_f,
                                           stride = self.stride,
                                           padding = self.padding,
                                           dilation = self.dilation,
                                           groups = self.groups)
        error = torch.add(output, res_f, alpha=-1).data
        noise = error.pow(2).mean()
        if noise > 0:
          eff = 1.25 * res_f.pow(2).mean().div(noise).log10().detach().cpu().numpy()
          dev = math.fabs(eff - self.efficency)
          if dev > 0:
            self.efficency = (self.efficency * 4 + eff) * 0.2
            self.deviation = (self.deviation * 4 + dev) * 0.2
            #print(self.node.name, self.efficency, self.deviation)
            if self.efficency > 4.0:
              rate = rate * 0.5
            if (self.efficency > 4.3 or
                (self.deviation / self.efficency) < 0.05 or
                math.fabs(dev - self.deviation / dev) < 0.05):
              self.stop = True
          else:
            self.stop = True
        else:
          self.stop = True
        if (not self.stop) and (self.bias is not None):
          error = error.mean(dim = [0, 1, 2])
          self.bias.data = torch.sub(self.bias.data, error, alpha=rate)
        self.param_quantized = False

    return output

  def bias_corr(self):
    if self.bias is not None and self.bias_f is not None:
      bias_err = torch.sub(self.bias_f, self.bias.data)
      return bias_err.cpu().numpy().tolist()
    else:
      return None


@py_utils.register_quant_op
def Conv1d(*args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode == None:
    return torch.nn.Conv1d(*args, **kwargs)
  return deephi_Conv1d(*args, **kwargs)
