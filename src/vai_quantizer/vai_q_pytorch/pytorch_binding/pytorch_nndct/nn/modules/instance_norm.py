

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
__all__ = ['InstanceNorm']

class deephi_InstanceNorm(torch.nn.modules.instancenorm._InstanceNorm):
  r"""DeePhi instancenorm operation, support float and double"""

  def __init__(self, *args, **kwards):
    super(deephi_InstanceNorm, self).__init__(*args, **kwards)
    self.params_name = None
    self.node = None
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.param_saved = False
    self.param_quantized = False

    
  def forward(self, input):
    if NndctOption.nndct_quant_off.value or not self.quantizer.configer.is_node_quantizable(self.node, False):
      return super().forward(input)
    
    params = []
    if self.weight is not None:
      params.append(self.weight)
    if self.bias is not None:
      params.append(self.bias)
    param_names = self.params_name[:len(params)]
    if len(params) != len(param_names):
      NndctScreenLogger().error(f"Parameter number in Instance operator error!")
      exit(2)

    qinput = quantize_tensors([input], self.node, tensor_type='input')[0]
 
    if (not self.param_quantized) and len(params) > 0:
      inplace = self.quantizer is not None and self.quantizer.inplace
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

    if NndctOption.nndct_op_instancenorm_mode.value == "ipu" and (not self.quantizer.exporting):
      output = self.simulateInstanceNorm(qinput, qparams)
    else:
      output = torch.nn.functional.instance_norm(
              qinput,
              self.running_mean,
              self.running_var,
              qparams[0] if self.param_quantized else None,
              qparams[1] if self.param_quantized else None,
              self.training or not self.track_running_stats,
              self.momentum,
              self.eps)

    # quantize output
    output = quantize_tensors([output], self.node)[0]
    return output

  def _check_input_dim(self, input):
    pass

  def simulateInstanceNorm(self, inp, params):
    # input: x, gamma, beta and fixed-points 
    def instancenorm_process(inp, ifp, weight, wfp, bias, bfp):
      # inp shape: 1d (N, C, L)
      #            2d (N, C, H, W)
      #            3d (N, C, D, H, W)
      e_val = 0.0000152587890625
      dim = [i for i in range(2, len(inp.shape))] # e.g., if inp=(N, C, H, W), then dim=[2, 3]
      numel = inp[0][0].numel()
      coff = 1.0/numel # float16
      inp_mean = torch.mean(input=inp, dim=dim, keepdim=True, dtype=torch.float32) # float32
      inp_mean = inp_mean.to(torch.bfloat16) 

      inp_sub = inp - inp_mean # bfloat16
      inp_sub = inp_sub.to(torch.float32) # float32 

      # 1/sqrt{x}: 1/sigma
      inp_square = torch.square(inp_sub) # float32
      inp_sum = torch.sum(inp_square, dim=dim, keepdim=True, dtype=torch.float32) # sum of squares
      inp_sum = inp_sum*coff + e_val # float32
      inverse_sqrt = py_utils.invsqrt(mean_arr)
      inverse_sqrt = torch.from_numpy(inverse_sqrt).to(torch.bfloat16).to(v.device)

      # (x - mu)/sigma
      mul = torch.mul(inp_sub, inverse_sqrt) # float32
      mul = mul.to(torch.bfloat16) # bfloat16

      # (x - mu)/sigma*gamma + beta
      if weight is not None and bias is not None: # (x - mu)/sigma*gamma + beta
        weight = weight.to(torch.float32).repeat(mul.shape[0], 1) # (N, C)
        bias = bias.to(torch.float32).repeat(mul.shape[0], 1)     # (N, C)
        for i in range(len(weight.shape), len(mul.shape)): # unsqueeze
          weight = torch.unsqueeze(weight, -1) # 1d (N, C, 1); 2d (N, C, 1, 1); 3d (N, C, 1, 1, 1)
          bias = torch.unsqueeze(bias, -1)     # 1d (N, C, 1); 2d (N, C, 1, 1); 3d (N, C, 1, 1, 1)

        out = mul.to(torch.float32)*weight + bias # (x-mu)/sigma*gamma + beta, float32
        out = out.to(torch.bfloat16).to(torch.float32) # float32
      else: # (x - mu)/sigma 
        out = mul.to(torch.float32)

      return out

    # main process
    input_name = self.node.in_nodes[0]
    input_node = self.quantizer.configer.get_Nndctnode(input_name)
    if not self.quantizer.configer.node_output_quantizable(input_node):
      input_name = input_node.in_nodes[0]
    ifp = self.quantizer.get_quant_config(input_name, False)[1]
    wfp = self.quantizer.get_quant_config(self.params_name[0], False, tensor_type='param')[1] if self.affine else None 
    bfp = self.quantizer.get_quant_config(self.params_name[1], False, tensor_type='param')[1] if self.affine else None
    if params:
      weight, bias = params[0:2]
    else:
      weight, bias = None, None

    # fake int8: [-256, 255]
    inp = (inp * (2**ifp)).to(torch.bfloat16) # bfloat16 
    out = instancenorm_process(inp, ifp, weight, wfp, bias, bfp)
    
    return out
 
@py_utils.register_quant_op
def InstanceNorm(*args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode == None:
    def _check_input_dim(self, input):
      pass
    import types
    nn = torch.nn.modules.instancenorm._InstanceNorm(*args, **kwargs)
    
    nn._check_input_dim = types.MethodType(_check_input_dim, nn)
    return nn
  return deephi_InstanceNorm(*args, **kwargs)
