

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
from nndct_shared.quantization import quantize_tensors
from nndct_shared.quantization import maybe_get_quantizer
import pytorch_nndct.utils as py_utils
import torch.nn.functional as F
__all__ = ['BatchNorm']

class deephi_BatchNorm(torch.nn.modules.batchnorm._BatchNorm):
  r"""DeePhi batchnorm operation, support float and double"""

  def __init__(self, *args, **kwards):
    super(deephi_BatchNorm, self).__init__(*args, **kwards)
    self.params_name = None
    self.node = None
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.param_saved = False
    self.param_quantized = False

    
  def forward(self, input):
    params = [self.weight, self.bias]
    param_names = self.params_name[:2]
    
    qinput = quantize_tensors([input], self.node, tensor_type='input')[0]
    
    if (not self.param_quantized):
      inplace = (NndctOption.nndct_quant_off.value or self.quantizer is not None and self.quantizer.inplace)
      # quantize weights and bias
      if inplace:
        _ = quantize_tensors(
            params,
            self.node,
            tensor_names=param_names,
            tensor_type='param')
        qparams = [p for p in params]
      else:
        qparams = quantize_tensors(
            params,
            self.node,
            tensor_names=param_names,
            tensor_type='param')
      if not NndctOption.nndct_quant_off.value:
        self.param_quantized = True
    else:
      qparams = [p for p in params]

    if self.momentum is None:
      exponential_average_factor = 0.0
    else:
      exponential_average_factor = self.momentum

    if self.training and self.track_running_stats:    
      # TODO: if statement only here to tell the jit to skip emitting this when it is None
      if self.num_batches_tracked is not None:  # type: ignore[has-type]
        self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
        if self.momentum is None:  # use cumulative moving average
          exponential_average_factor = 1.0 / float(self.num_batches_tracked)
        else:  # use exponential moving average
          exponential_average_factor = self.momentum

    if self.training:
      bn_training = True
    else:
      bn_training = (self.running_mean is None) and (self.running_var is None)

    output = torch.nn.functional.batch_norm(
            qinput,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            qparams[0],
            qparams[1],
            bn_training,
            exponential_average_factor,
            self.eps,
        )

    # quantize output
    output = quantize_tensors([output], self.node)[0]
    return output

  def _check_input_dim(self, input):
    pass

 
@py_utils.register_quant_op
def BatchNorm(*args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode == None:
    def _check_input_dim(self, input):
      pass
    import types
    nn = torch.nn.modules.batchnorm._BatchNorm(*args, **kwargs)
    
    nn._check_input_dim = types.MethodType(_check_input_dim, nn)
    return nn
  return deephi_BatchNorm(*args, **kwargs)
