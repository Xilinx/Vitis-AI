

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
from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.utils import NndctOption
from nndct_shared.quantization import kernel_need_quant
from nndct_shared.quantization import quantize_tensors
import pytorch_nndct.utils as py_utils
import numpy as np
from pytorch_nndct.utils import Const
from .fix_ops import NndctISqrt, NndctAIEISqrt
__all__ = ['LayerNorm']

class deephi_LayerNorm(torch.nn.LayerNorm):
  r"""DeePhi ReLU operation"""

  def __init__(self, *args, **kwargs):
    super(deephi_LayerNorm, self).__init__(*args, **kwargs)
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.node = None
    self.param_quantized = False

  def forward(self, input):
    if not kernel_need_quant(self.quantizer, self.node) or NndctOption.nndct_gemm88.value:
      output = super().forward(input)
      output = quantize_tensors([output], self.node)[0]
      return output
    
    # quantize input tensor
    qinput = quantize_tensors([input], self.node, tensor_type='input')[0]

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

    if (NndctOption.nndct_quant_off.value or self.quantizer is None or self.quantizer.exporting or
        self.quantizer.configer.will_merge_with_table(self.node, (not NndctOption.nndct_cv_app.value)) or
        (not self.quantizer.configer.is_node_quantizable(self.node, self.quantizer.lstm))):
      output = torch.nn.functional.layer_norm(qinput, 
                                              self.normalized_shape,
                                              qweight,
                                              qbias,
                                              self.eps)
      output = quantize_tensors([output], self.node)[0]
      return output

    # quantization configure
    input_name = self.node.in_nodes[0]
    input_node = self.quantizer.configer.get_Nndctnode(input_name)
    if not self.quantizer.configer.node_output_quantizable(input_node):
      input_name = input_node.in_nodes[0]
    
    # quantization method
    if NndctOption.nndct_op_layernorm_mode.value == "aie2_16bw" or NndctOption.nndct_ip_asr.value:
      fragpos = self.quantizer.get_quant_config(input_name, False)[1]
      wfp, bfp = None, None
      wfp = self.quantizer.get_quant_config(self.params_name[0], False, tensor_type='param')[1] if self.elementwise_affine else None
      bfp = self.quantizer.get_quant_config(self.params_name[1], False, tensor_type='param')[1] if self.elementwise_affine else None
      output = self._simulatedLayerNorm1(qinput, self.normalized_shape, self.elementwise_affine, qweight, qbias, fragpos, wfp, bfp)
      output = quantize_tensors([output], self.node, method=4)[0]
    elif NndctOption.nndct_op_layernorm_mode.value == "bert_8bw" or NndctOption.nndct_ip_v70_bert.value:
      fragpos = self.quantizer.get_quant_config(input_name, False)[1]
      wfp, bfp = None, None
      wfp = self.quantizer.get_quant_config(self.params_name[0], False, tensor_type='param')[1] if self.elementwise_affine else None
      bfp = self.quantizer.get_quant_config(self.params_name[1], False, tensor_type='param')[1] if self.elementwise_affine else None
      # bfloat16 Newton iteration for 1/sqrt(x)
      output = self._simulatedLayerNorm2(qinput, self.normalized_shape, self.elementwise_affine, qweight, qbias, fragpos, wfp, bfp)
      output = quantize_tensors([output], self.node, method=4)[0] # 4: floor
    else:
      output = torch.nn.functional.layer_norm(qinput, self.normalized_shape, qweight, qbias, self.eps)
      output = quantize_tensors([output], self.node)[0] # 2: half up
    
    return output

  # e.g. inp: (N, C, H, W), normalized_shape=(H, W)
  def _simulatedLayerNorm1(self, inp, normalized_shape, elementwise_affine, weight, bias, ifp, wfp, bfp):
    inp = torch.floor(inp * (2**ifp))  # float32
    
    dim = [i for i in range(inp.dim()) if i >= inp.dim()-len(normalized_shape)] # normlized_shape dim=[2, 3]
    
    # mean: aie_mean equals to torch.mean(inp, dim, keepdim=True) except adding details
    # aid_add input 
    inp_temp1 = inp.reshape((-1,) + normalized_shape) # (N*C, H, W), bfloat16
    numel = inp_temp1[0].numel() # element number

    inp_temp2 = inp_temp1.reshape(inp_temp1.shape[0], -1)
    sum_aie = torch.sum(inp_temp2, dim = -1, dtype=torch.float32)
    # aie mean
    numel_inv = 1/numel
    mean_aie = sum_aie * numel_inv  # float32
    # reshape mean
    dim_mean = []  # (N, C, 1, 1)
    for i in range(inp.dim()):
      if i < inp.dim() - len(normalized_shape):
        dim_mean.append(inp.shape[i])
      else:
        dim_mean.append(1)
    mean = mean_aie.reshape(dim_mean)  # (N, C, 1, 1)
    
    # x-mu
    inp = inp.to(torch.bfloat16)  # bfloat16
    mean = mean.to(torch.bfloat16)  # (N, C, 1, 1)
    sub = inp - mean           # bfloat16
    sub = sub.to(torch.float32) # float32
    # var
    square = torch.square(sub) # float32
    
    sum_sq = torch.sum(square, dim = dim, dtype=torch.float32)
    sum_sq = sum_sq.to(torch.bfloat16)
    var = sum_sq/numel
    # var = square.mean(dim, dtype=torch.float32) # float32, (N, C)
    var = var + Const.EPS.value 
    var = var.to(torch.float32)
    
    # isqrt: 1/sqrt(var)
    isqrt = torch.empty_like(var)
    NndctISqrt(var, isqrt) # CUDA/CPU: float32
    isqrt = isqrt.to(torch.bfloat16).to(torch.float32)

    # mul: (x-mu)*(1/sigma)
    for i in range(isqrt.dim(), sub.dim()): # isqrt shape: (N, C, 1, 1)
      isqrt = torch.unsqueeze(isqrt, dim=-1)
    mul = torch.mul(sub, isqrt).to(torch.bfloat16).to(torch.float32) # float32, (N, C, H, W)
   
    # affine: layernom*gamma + beta
    if elementwise_affine:
      weight = torch.floor(weight * (2 ** wfp)) # float32
      weight = weight.to(torch.bfloat16) / (2 ** wfp) # bfloat16
      weight = weight.to(torch.float32)  # float32
      bias = torch.floor(bias * (2 ** bfp))  # float32
      bias = bias.to(torch.bfloat16) / (2 ** bfp)  # bfloat16
      bias =bias.to(torch.float32)  # float32
      axb = mul * weight + bias # ax+b, float32; weight,bias: normalized_shape 
      out = axb.to(torch.bfloat16).to(torch.float32)
    else:
      out = mul

    return out

  # e.g. inp: (N, C, H, W), normalized_shape=(H, W)
  def _simulatedLayerNorm2(self, inp, normalized_shape, elementwise_affine, weight, bias, ifp, wfp, bfp):
    inp = inp*(2**ifp)  # float32
    inp = inp.to(torch.bfloat16).to(torch.float32)  # float32 
    dim = [i for i in range(inp.dim()) if i >= inp.dim()-len(normalized_shape)] # normlized_shape dim=[2, 3]
    
    # mean: aie_mean equals to torch.mean(inp, dim, keepdim=True) except adding details
    # aid_add input 
    inp_temp1 = inp.reshape((-1,) + normalized_shape) # (N*C, H, W), float32
    numel = inp_temp1[0].numel() # element number
    inp_temp2 = inp_temp1.reshape(inp_temp1.shape[0], -1, 8) # (N*C, H*W/8, 8), float32
    inp_v8 = inp_temp2.sum(dim=-2, keepdim=False, dtype=torch.float32) # (N*C, 8), float32
    # aie add v8
    sum_aie = py_utils.aie_add_v8(inp_v8) # aie add: (N*C, ), float32
    numel_inv = 1/numel
    mean_aie = sum_aie*numel_inv  # float32
    # reshape mean
    dim_mean = []  # (N, C, 1, 1)
    for i in range(inp.dim()):
      if i < inp.dim() - len(normalized_shape):
        dim_mean.append(inp.shape[i])
      else:
        dim_mean.append(1)
    mean = mean_aie.reshape(dim_mean)  # (N, C, 1, 1), float32
    
    # x-mu
    sub = inp - mean           # float32
    sub = sub.to(torch.bfloat16).to(torch.float32) # float32
    
    # var
    square = torch.square(sub) # float32
    var = square.mean(dim, dtype=torch.float32) # float32, (N, C)
    var = var + Const.EPS.value 
    
    # isqrt: 1/sqrt(var)
    isqrt = torch.empty_like(var)
    NndctAIEISqrt(var, isqrt) # CUDA/CPU: float32
    isqrt = isqrt.to(torch.bfloat16).to(torch.float32) # float32

    # mul: (x-mu)*(1/sigma)
    for i in range(isqrt.dim(), sub.dim()): # isqrt shape: (N, C, 1, 1)
      isqrt = torch.unsqueeze(isqrt, dim=-1)
    mul = torch.mul(sub, isqrt) # float32, (N, C, H, W)
    mul = mul.to(torch.bfloat16).to(torch.float32) # float32
   
    # affine: layernom*gamma + beta
    if elementwise_affine:
      weight = weight.to(torch.bfloat16).to(torch.float32)
      bias = bias.to(torch.bfloat16).to(torch.float32)
      axb = mul*weight + bias
      out = axb.to(torch.bfloat16).to(torch.float32)
    else:
      out = mul

    return out

@py_utils.register_quant_op
def LayerNorm(*args, **kwargs):
  quant_mode,_ = maybe_get_quantizer()
  if quant_mode is None:
    return torch.nn.LayerNorm(*args, **kwargs)
  return deephi_LayerNorm(*args, **kwargs)
