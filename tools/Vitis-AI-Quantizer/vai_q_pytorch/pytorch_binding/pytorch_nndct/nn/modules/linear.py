

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
import math
from nndct_shared.utils import NndctOption
from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import process_inputs_and_params
from nndct_shared.quantization import post_quant_process
import pytorch_nndct.utils as py_utils
from .fix_ops import NndctScale
from .add import Add
from .multiply import Mul

__all__ = ['Linear']

class deephi_Linear(torch.nn.modules.linear.Linear):
  r"""DeePhi Linear operation, support float and double"""

  def __init__(self, *args, **kwards):
    super(deephi_Linear, self).__init__(*args, **kwards)
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
        # adjust weight and bias
        if self.quant_mode == 2:
          if (self.bias is not None):
            self.bias.data = torch.sub(self.bias.data, torch.tensor(
                self.quantizer.bias_corr[self.node.name],
                device=self.bias.data.device))
      self.param_saved = True
    if (not self.param_quantized):
      __, __ = process_inputs_and_params(
          self.node,
          self.quantizer,
          inputs=[],
          params=params,
          param_names=self.params_name)
      self.param_quantized = True

    # split linear to mul and add operations
    if (self.quant_mode == 2 and self.quantizer.is_lstm):
      # i * w
      output = torch.matmul(input, torch.transpose(self.weight, 0, 1))
      #if (self.node.name == 'EncoderModuleLayer0LayerLstmCellModule::AugmentedLstmCell/Linear[input_linearity]/10'):
      #  print('---- target value = {}'.format(float(output[0][1687])), flush=True)
      #[output] = post_quant_process(self.node, [output])
      output = self.quantizer.do_quantize(output, self.node.name, self.node, tensor_type='output')
      # i*w + bias
      if self.bias is not None:
        output = torch.add(output, self.bias)
    else:
      output = super().forward(input)

    [output] = post_quant_process(self.node, [output])

    if NndctOption.nndct_param_corr.value > 0:
      rate = NndctOption.nndct_param_corr_rate.value
      # statistic of quantization error
      if (self.quant_mode == 1 and not self.stop):
        res_f = torch.matmul(input, torch.transpose(self.weight_f, 0, 1))
        if self.bias is not None:
          res_f = torch.add(res_f, self.bias_f)
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
          if error.dim() == 3:
            error = error.mean(dim = [0, 1])
          else:
            error = error.mean(dim = 0)
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
def Linear(*args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode == None:
    return torch.nn.Linear(*args, **kwargs)
  return deephi_Linear(*args, **kwargs)

