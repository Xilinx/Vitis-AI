

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

from unicodedata import name
import torch
from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.utils import NndctOption
from nndct_shared.quantization import quantize_tensors 
import pytorch_nndct.utils as py_utils
import numpy as np
import struct 

__all__ = ['LayerNorm']
EPS = 0.0000152587890625

class deephi_LayerNorm(torch.nn.LayerNorm):
  r"""DeePhi ReLU operation"""

  def __init__(self, *args, **kwargs):
    super(deephi_LayerNorm, self).__init__(*args, **kwargs)
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.node = None
    self.param_quantized = False

  def forward(self, input):
    if NndctOption.nndct_quant_off.value or not self.quantizer.configer.is_node_quantizable(self.node, lstm=False):
      return super().forward(input)
    
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

    def layernorm_process(layernorm_onebt, inp, ifp, norm_shape, weight, wfp, bias, bfp,eps):
      def process_case(inp, scale, bias, n, w, wfp, bfp, is_axb):  
        
        inp = inp.reshape(n,w)
        out = torch.empty_like(inp).to(torch.float32)

        for i in range(n):
          out[i] = layernorm_onebt(inp[i], scale, bias, w, wfp, bfp, is_axb)
        return out
    
      inp = torch.floor(inp * (2**ifp))
      weight = torch.floor(weight * (2 ** wfp))
      bias = torch.floor(bias * (2 ** bfp))

      inp_shape = inp.shape
      n_num = tuple(inp_shape)[:-len(norm_shape)]
      n = 1
      for i in n_num:
        n *= i
      w = 1
      for i in norm_shape:
        w *= i
      
      out = process_case(inp, weight, bias, n, w, wfp, bfp, 1)
      out = out.reshape(inp_shape)
      
      return out

    def simulatedLayerNorm1(v, scale, bias, w, wfp, bfp, is_axb):
      v = v.to(torch.bfloat16)
      w_inv = 1.0/w
      sum_t = torch.sum(v, dtype=torch.float32)
      mean_t = sum_t * w_inv
      mean_t = mean_t.to(torch.bfloat16)
      v = v.to(torch.bfloat16)
      sub = v - mean_t
      sub = sub.to(torch.float32)

      square = torch.square(sub)
      sum_sq = torch.sum(square, dtype=torch.float32)
      sum_sq = sum_sq.to(torch.bfloat16)
      mean_sq = sum_sq/w
      mean_sq = mean_sq + EPS 

      mean_sq = mean_sq.to(torch.float32).cpu().numpy()
      mean_arr = np.empty([1])
      mean_arr[0] = mean_sq
      isqrt = py_utils.invsqrt(mean_arr)
      isqrt = torch.from_numpy(isqrt).to(torch.bfloat16).to(v.device)

      mul = torch.mul(sub, isqrt).to(torch.bfloat16)
      
      if(is_axb):
        scale = scale.to(torch.bfloat16) / (2 ** wfp)
        bias = bias.to(torch.bfloat16) / (2 ** bfp)
        axb = mul.reshape(-1,).to(torch.float32) * scale.reshape(-1,).to(torch.float32) + bias.reshape(-1,).to(torch.float32) #bias 后處理
        y = axb.to(torch.bfloat16)
        y = y.to(torch.float32)

      return y

    def simulatedLayerNorm2(v, scale, bias, w, wfp, bfp, is_axb): # v70, bert
      v = v.to(torch.float32)
      w_inv = 1.0/w
      sum_t = torch.sum(v, dtype=torch.float32)
      mean_t = sum_t * w_inv
      mean_t = mean_t.to(torch.bfloat16)
      v = v.to(torch.bfloat16)
      sub = v - mean_t
      sub = sub.to(torch.float32)

      square = torch.square(sub)
      sum_sq = torch.sum(square, dtype=torch.float32)
      mean_sq = sum_sq/w
      mean_sq = mean_sq + EPS 

      mean_sq = mean_sq.to(torch.float32).cpu().detach().numpy()
      mean_arr = np.empty([1])
      mean_arr[0] = mean_sq
      isqrt = py_utils.invsqrt(mean_arr)
      isqrt = torch.from_numpy(isqrt).to(torch.bfloat16).to(v.device)

      mul = torch.mul(sub, isqrt).to(torch.bfloat16)
      
      if(is_axb):
        scale = scale.to(torch.bfloat16) / (2 ** wfp)
        bias = bias.to(torch.bfloat16) / (2 ** bfp)
        axb = mul.reshape(-1,).to(torch.float32) * scale.reshape(-1,).to(torch.float32) + bias.reshape(-1,).to(torch.float32) #bias 后處理
        y = axb.to(torch.bfloat16)
        y = y.to(torch.float32)
      else:
        y = mul.reshape(-1,).to(torch.float32)

      return y
        
    # quantization configure
    input_name = self.node.in_nodes[0]
    input_node = self.quantizer.configer.get_Nndctnode(input_name)
    if not self.quantizer.configer.node_output_quantizable(input_node):
      input_name = input_node.in_nodes[0]
    fragpos = self.quantizer.get_quant_config(input_name, False)[1]
    wfp = self.quantizer.get_quant_config(self.params_name[0], False, tensor_type='param')[1]
    bfp = self.quantizer.get_quant_config(self.params_name[1], False, tensor_type='param')[1]

    # quantization method
    if NndctOption.nndct_op_layernorm_mode.value == "aie2_16bw" or NndctOption.nndct_ip_asr.value:
      output = layernorm_process(simulatedLayerNorm1, qinput, fragpos, self.normalized_shape, qweight, wfp, qbias, bfp, self.eps)
      output = quantize_tensors([output], self.node, method=4)[0]
    elif NndctOption.nndct_op_layernorm_mode.value == "bert_8bw" or NndctOption.nndct_ip_v70_bert.value: 
      output = layernorm_process(simulatedLayerNorm2, qinput, fragpos, self.normalized_shape, qweight, wfp, qbias, bfp, self.eps)
      output = quantize_tensors([output], self.node, method=4)[0] # 4: floor
    else:
      output = torch.nn.functional.layer_norm(qinput, self.normalized_shape, qweight, qbias, self.eps)
      output = quantize_tensors([output], self.node)[0] # 2: half up
    
    return output

@py_utils.register_quant_op
def LayerNorm(*args, **kwargs):
  quant_mode,_ = maybe_get_quantizer()
  if quant_mode is None:
    return torch.nn.LayerNorm(*args, **kwargs)
  return deephi_LayerNorm(*args, **kwargs)
