

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
from nndct_shared.utils import NndctOption, NndctScreenLogger, QError, QWarning
from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import quantize_tensors 
from .quant_noise import eval_qnoise
import pytorch_nndct.utils as py_utils
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

    # self.weight and self.bias are not quantized float parameters
    self.weight_bak = None # backup of float bias for bias correction
    self.bias_bak = None   # backup of float bias for bias correction
    self.stop = False
    self.rate = NndctOption.nndct_param_corr_rate.value
    self.efficency = 0.0
    self.deviation = 0.0

  def forward(self, input):
    # backup bias for bias correction feature
    if (not self.param_saved):
      if NndctOption.nndct_param_corr.value > 0:
        # backup orignal float parameters
        if self.quant_mode == 1:
          self.weight_bak = self.weight.detach().clone()
          if self.bias is not None:
            self.bias_bak = self.bias.detach().clone()
        # adjust bias
        if self.quant_mode == 2 and self.bias is not None:
          if self.node.name not in self.quantizer.bias_corr.keys():
            NndctScreenLogger().error2user(QError.BIAS_CORRECTION, f"Bias correction file in quantization result directory does not match current model.")
            exit(2)
          self.bias.data = torch.sub(self.bias.data, torch.tensor(
              self.quantizer.bias_corr[self.node.name],
              device=self.bias.data.device,
              dtype=self.bias.data.dtype))
      self.param_saved = True

    # quantize parameters
    qweight = None
    qbias = None
    inplace = (NndctOption.nndct_quant_off.value or 
        self.quantizer is not None and self.quantizer.inplace)
    if (not self.param_quantized):
      if inplace:
        _ = quantize_tensors(
            [self.weight],
            self.node,
            tensor_names = [self.params_name[0]],
            tensor_type = 'param')[0]
        qweight = self.weight
        if self.bias is not None:
          _ = quantize_tensors(
              [self.bias],
              self.node,
              tensor_names = [self.params_name[1]],
              tensor_type = 'param')[0]
          qbias = self.bias
      else:
        qweight = quantize_tensors(
            [self.weight],
            self.node,
            tensor_names = [self.params_name[0]],
            tensor_type = 'param')[0]
        if self.bias is not None:
          qbias = quantize_tensors(
              [self.bias],
              self.node,
              tensor_names = [self.params_name[1]],
              tensor_type = 'param')[0]
      if not NndctOption.nndct_quant_off.value:
        self.param_quantized = True
    else:
      qweight = self.weight
      qbias = self.bias


    # quantize input tensor
    qinput = quantize_tensors([input], self.node, tensor_type='input')[0]
    # split linear to mul and add operations
    if (self.quant_mode == 2 and self.quantizer.is_lstm):
      # i * w
      output = torch.matmul(qinput, torch.transpose(qweight, 0, 1))
      output = self.quantizer.do_quantize(output, self.node.name, self.node, tensor_type='output')
      # i*w + bias
      if self.bias is not None:
        output = torch.add(output, qbias)
    else:
      output = torch.nn.functional.linear(qinput, qweight, qbias)
    output = quantize_tensors([output], self.node)[0]

    if NndctOption.nndct_param_corr.value > 0:
      #rate = NndctOption.nndct_param_corr_rate.value
      # statistic of quantization error
      if (self.quant_mode == 1 and not self.stop):
        res_f = torch.matmul(input, torch.transpose(self.weight_bak, 0, 1))
        if self.bias is not None:
          res_f = torch.add(res_f, self.bias_bak)
        error, rate, self.stop, self.efficency, self.deviation = eval_qnoise(
                            output, 
                            res_f, 
                            self.efficency, 
                            self.deviation, 
                            self.rate, 
                            self.stop)
        if (not self.stop) and (self.bias is not None):
          error = error.mean(dim = [k for k in range(error.dim()-1)])
          self.bias.data = torch.sub(self.bias.data, error, alpha=rate)
        self.param_quantized = False

    return output

  def bias_corr(self):
    if self.bias is not None and self.bias_bak is not None:
      bias_err = torch.sub(self.bias_bak, self.bias.data)
      return bias_err.cpu().numpy().tolist()
    else:
      return None


@py_utils.register_quant_op
def Linear(*args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode == None:
    return torch.nn.Linear(*args, **kwargs)
  return deephi_Linear(*args, **kwargs)

