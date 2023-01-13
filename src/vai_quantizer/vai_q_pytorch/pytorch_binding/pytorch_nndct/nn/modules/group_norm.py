

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
__all__ = ['GroupNorm']

class deephi_GroupNorm(torch.nn.modules.normalization.GroupNorm):
  r"""DeePhi group normalization operation, support float and double"""

  def __init__(self, *args, **kwards):
    super(deephi_GroupNorm, self).__init__(*args, **kwards)
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

    # quantization configure
    input_name = self.node.in_nodes[0]
    input_node = self.quantizer.configer.get_Nndctnode(input_name)
    if not self.quantizer.configer.node_output_quantizable(input_node):
      input_name = input_node.in_nodes[0]
    ifp = self.quantizer.get_quant_config(input_name, False)[1]
    wfp = self.quantizer.get_quant_config(self.params_name[0], False, tensor_type='param')[1] if self.affine else None
    bfp = self.quantizer.get_quant_config(self.params_name[1], False, tensor_type='param')[1] if self.affine else None

    if NndctOption.nndct_op_groupnorm_mode.value == "ipu_8bw" and (not self.quantizer.exporting):
      output = self.simulateGroupNorm(qinput, self.num_groups, self.affine, qparams[0], qparams[1], ifp, wfp, bfp)
      output = quantize_tensors([output], self.node, method=4)[0]
    else:
      output = torch.nn.functional.group_norm(
              qinput,
              self.num_groups,
              qparams[0] if self.param_quantized else None,
              qparams[1] if self.param_quantized else None,
              self.eps
          )
      output = quantize_tensors([output], self.node)[0]

    return output

  # e.g. inp (N, C, H, W)
  def simulateGroupNorm(self, inp, num_groups, affine, weight, bias, ifp, wfp, bfp):
    # inp fake int8: [-256, 255]
    inp = torch.floor(inp * (2**ifp))  # float32 
    inp = inp.to(torch.bfloat16) # bfloat16
    
    # inp reshape: (N, G, C//G, H, W)
    G = self.num_groups
    if inp.dim() == 3: # [N, C, L]
      N, C, L = inp.shape
      inp_group = inp.reshape(N, G, C//G, L)
      dim = [2, 3]
    elif inp.dim() == 4: # [N, C, H, W]
      N, C, H, W = inp.shape
      inp_group = inp.reshape(N, G, C//G, H, W)
      dim = [2, 3, 4]
    elif inp.dim() == 5: # [N, C, L, H, W]
      N, C, D, H, W = inp.shape
      inp_group = inp.reshape(N, G, C//G, D, H, W)
      dim = [2, 3, 4, 5]
    else: 
      NndctScreenLogger().error2user(QError.INPUT_DIMENSION, f"Input dimension error in node {self.node}! The dimension of input is {inp.dim()}.")

    numel = inp_group[0][0].numel()

    # mean: aie_mean equals to torch.mean(inp, dim, keepdim=True) except adding
    # aid_add input reshape
    NG, CG = inp_group.shape[0], inp_group.shape[1] # inp_group: (N, G, C//G, H, W)=(NG, CG, DG, HG, WG)
    inp_temp1 = inp_group.reshape(NG, CG, -1) # (NG, CG, DG*HG*WG), bfloat16
    numel = inp_temp1[0][0].numel() # element number
    inp_temp1 = inp_group.reshape(-1, inp_temp1.shape[-1]) # (NG*CG, DG*HG*WG), bfloat16
    inp_temp2 = inp_temp1.reshape(inp_temp1.shape[0], -1, 8) # (NG*CG, DG*HG*WG/8, 8), bfloat16
    inp_v8 = inp_temp2.sum(dim=-2, keepdim=False, dtype=torch.float32) # (NG*CG, 8), float32
    # aie add v8
    sum = py_utils.aie_add_v8(inp_v8) # aie sum: (NG*CG,), float32
    # aie mean
    sum = sum.reshape(NG, CG) # (NG, CG), float32
    mean = sum/numel  # (NG, CG), float32
    # mean unsqueeze: (NG, CG, 1, 1)
    for i in range(mean.dim(), inp_group.dim()):
      mean = torch.unsqueeze(mean, -1)
    mean = mean.to(torch.bfloat16)  # (NG, CG, 1, 1, 1)

    # x - mu
    sub = inp_group - mean # (NG, CG, DG, HG, WG), bfloat16
    sub = sub.to(torch.float32) # float32 

    # var
    square = torch.square(sub) # float32
    var = square.mean(dim, keepdim=False, dtype=torch.float32) # (NG, CG), float32: dim=(2, 3, 4)
    
    # isqrt: 1/sqrt(var)
    isqrt = torch.empty_like(var)
    NndctISqrt(var, isqrt) # CUDA/CPU: float32
    isqrt = isqrt.to(torch.bfloat16)
   
    # mul: (x-mu)*(1/sigma)
    for i in range(isqrt.dim(), sub.dim()): # isqrt shape: (NG, CG, 1, 1, 1)
      isqrt = torch.unsqueeze(isqrt, dim=-1)
    mul = torch.mul(sub, isqrt) # float32, (NG, CG, DG, HG, WG)
    mul = mul.to(torch.bfloat16).to(torch.float32) # float32
    mul = mul.reshape(N, C, H, W) # (N, C, H, W)

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

@py_utils.register_quant_op
def GroupNorm(*args, **kwargs):
  return deephi_GroupNorm(*args, **kwargs)
