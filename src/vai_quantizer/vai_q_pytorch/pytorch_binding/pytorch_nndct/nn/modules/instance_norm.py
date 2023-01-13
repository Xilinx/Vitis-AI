

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

from nndct_shared.utils import NndctOption, NndctScreenLogger, QError
from nndct_shared.quantization import quantize_tensors, kernel_need_quant
from nndct_shared.quantization import maybe_get_quantizer
import pytorch_nndct.utils as py_utils
import torch.nn.functional as F
from pytorch_nndct.utils import Const
from .fix_ops import NndctISqrt
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
    qinput = quantize_tensors([input], self.node, tensor_type='input')[0]
 
    if not kernel_need_quant(self.quantizer, self.node):
      output = super().forward(qinput)
      output = quantize_tensors([output], self.node)[0]
      return output
    
    params = []
    if self.weight is not None:
      params.append(self.weight)
    if self.bias is not None:
      params.append(self.bias)
    param_names = self.params_name[:len(params)]
    if len(params) != len(param_names):
      NndctScreenLogger().error2user(QError.PARAM_NUMBER, f"Parameter number error in node {self.node} for InstanceNorm operator!")

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

    if NndctOption.nndct_op_instancenorm_mode.value == "ipu_8bw" and (not self.quantizer.exporting):
      output = self.simulateInstanceNorm(qinput, qparams)
      output = quantize_tensors([output], self.node, method=4)[0]
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
      output = quantize_tensors([output], self.node)[0]

    return output

  def simulateInstanceNorm(self, inp, params):
    # input: x, gamma, beta and fixed-points 
    def instancenorm_process(inp, ifp, weight, wfp, bias, bfp):
      # inp shape: 1d (N, C, L)
      #            2d (N, C, H, W)
      #            3d (N, C, D, H, W)
      inp = (inp * (2**ifp)).to(torch.bfloat16) # bfloat16 
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
      inp_sum = inp_sum*coff + Const.EPS.value # float32
      inverse_sqrt = py_utils.invsqrt(inp_sum.cpu().detach().numpy())
      inverse_sqrt = torch.from_numpy(inverse_sqrt).to(torch.bfloat16).to(inp.device)

      # (x - mu)/sigma
      mul = torch.mul(inp_sub, inverse_sqrt) # float32
      mul = mul.to(torch.bfloat16) # bfloat16

      # (x - mu)/sigma*gamma + beta
      if weight is not None and bias is not None: # (x - mu)/sigma*gamma + beta
        weight = weight.to(torch.float32).repeat(mul.shape[0], 1) # repeat C into (N, C)
        bias = bias.to(torch.float32).repeat(mul.shape[0], 1)     # repeat C into (N, C)
        for i in range(len(weight.shape), len(mul.shape)): # unsqueeze (N, C) into (N, C, 1), (N, C, 1, 1) or (N, C, 1, 1, 1)
          weight = torch.unsqueeze(weight, -1) # 1d (N, C, 1); 2d (N, C, 1, 1); 3d (N, C, 1, 1, 1)
          bias = torch.unsqueeze(bias, -1)     # 1d (N, C, 1); 2d (N, C, 1, 1); 3d (N, C, 1, 1, 1)

        out = mul.to(torch.float32)*weight + bias # (x-mu)/sigma*gamma + beta, float32
        out = out.to(torch.bfloat16).to(torch.float32) # float32
      else: # (x - mu)/sigma 
        out = mul.to(torch.float32)

      return out

    # e.g. inp: (N, C, H, W)
    def simulateInstanceNorm2(inp, affine, weight, bias, ifp, wfp, bfp):
      # inp
      inp = torch.floor(inp * (2**ifp))  # float32 
      inp = inp.to(torch.bfloat16) # bfloat16
      
      dim = [i for i in range(2, inp.dim())] # dim=[2, 3]
      numel = inp[0][0].numel()
      
      # mean: aie_mean equals to torch.mean(inp, dim, keepdim=True) except adding
      # aid_add input reshape
      N, C = inp.shape[0], inp.shape[1]
      inp_temp1 = inp.reshape(N, C, -1) # (N, C, H*W), bfloat16
      numel = inp_temp1[0][0].numel() # element number
      inp_temp1 = inp.reshape(-1, inp_temp1.shape[-1]) # (N*C, H*W), bfloat16
      inp_temp2 = inp_temp1.reshape(inp_temp1.shape[0], -1, 8) # (N*C, H*W/8, 8), bfloat16
      inp_v8 = inp_temp2.sum(dim=-2, keepdim=False, dtype=torch.float32) # (N*C, 8), float32
      # aie add v8
      sum = py_utils.aie_add_v8(inp_v8) # aie sum: (N*C,), float32
      # aie mean
      sum = sum.reshape(N, C) # (N, C), float32
      mean = sum/numel  # (N, C), float32
      # mean unsqueeze: (N, C, 1, 1)
      for i in range(mean.dim(), inp.dim()):
        mean = torch.unsqueeze(mean, -1)
      mean = mean.to(torch.bfloat16)  # (N, C, 1, 1)

      # x - mu
      sub = inp - mean # bfloat16
      sub = sub.to(torch.float32) # float32 

      # var
      square = torch.square(sub) # float32
      var = square.mean(dim, dtype=torch.float32) # float32, (N, C)
      
      # isqrt: 1/sqrt(var)
      isqrt = torch.empty_like(var)
      NndctISqrt(var, isqrt) # CUDA/CPU: float32
      isqrt = isqrt.to(torch.bfloat16)
     
      # mul: (x-mu)*(1/sigma)
      for i in range(isqrt.dim(), sub.dim()): # isqrt shape: (N, C, 1, 1)
        isqrt = torch.unsqueeze(isqrt, dim=-1)
      mul = torch.mul(sub, isqrt) # float32, (N, C, H, W)
      mul = mul.to(torch.bfloat16).to(torch.float32) # float32

      # affine: (x - mu)/sigma*gamma + beta
      if affine:
        weight = weight.repeat(mul.shape[0], 1) # repeat C into (N, C)
        bias = bias.repeat(mul.shape[0], 1)     # repeat C into (N, C)
        for i in range(weight.dim(), mul.dim()): # unsqueeze (N, C) into (N, C, 1), (N, C, 1, 1) or (N, C, 1, 1, 1)
          weight = torch.unsqueeze(weight, -1) # 1d (N, C, 1); 2d (N, C, 1, 1); 3d (N, C, 1, 1, 1)
          bias = torch.unsqueeze(bias, -1)     # 1d (N, C, 1); 2d (N, C, 1, 1); 3d (N, C, 1, 1, 1)

        out = mul*weight + bias # (x-mu)/sigma*gamma + beta, float32
        out = out.to(torch.bfloat16).to(torch.float32) # float32
      else: # (x - mu)/sigma 
        out = mul

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

    # out = instancenorm_process(inp, ifp, weight, wfp, bias, bfp) # low efficiency
    out = simulateInstanceNorm2(inp, self.affine, weight, bias, ifp, wfp, bfp) # high efficiency

    return out
 
@py_utils.register_quant_op
def InstanceNorm(*args, **kwargs):
  return deephi_InstanceNorm(*args, **kwargs)
