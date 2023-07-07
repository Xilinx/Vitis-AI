

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

import math
import torch
import numpy as np

from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import quantize_tensors, kernel_need_quant
from nndct_shared.utils import NndctOption
from .fix_ops import NndctSoftmaxExpApproximate, NndctSoftmaxLOD, NndctSoftmaxSimulationPart1, NndctSoftmaxSimulationPart2, NndctExpApprAIE2, NndctInverseAIE2
import pytorch_nndct.utils as py_utils

__all__ = ['Softmax']

class deephi_Softmax(torch.nn.modules.Softmax):
  r"""DeePhi Softmax operation"""

  def __init__(self, dim = None):
    super(deephi_Softmax, self).__init__()
    self.dim = dim
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.node = None

  def forward(self, input):
    qinput = quantize_tensors([input], self.node, tensor_type='input')[0]

    if (not kernel_need_quant(self.quantizer, self.node) or
        self.quantizer.exporting):
      # Method 0: quant input and output 
      output = super().forward(qinput)
      output = quantize_tensors([output], self.node)[0]

    else:
      input_name = self.node.in_nodes[0]
      input_node = self.quantizer.configer.get_Nndctnode(input_name)
      if not self.quantizer.configer.node_output_quantizable(input_node):
        input_name = input_node.in_nodes[0]
      bw = self.quantizer.get_quant_config(self.node.name, False)[0]
      fragpos = self.quantizer.get_quant_config(input_name, False)[1]

      # Method 1: Hardware PL Softmax with 8 bw
      if NndctOption.nndct_op_softmax_mode.value == "hardware_pl":
        x_max = torch.max(qinput, dim = self.dim, keepdim = True).values
        Exp_sum_appr = 0.0
        
        uvi = 47274 / math.pow(2,15) * (qinput - x_max)
        exp_appr = torch.empty_like(uvi)
        NndctSoftmaxExpApproximate(uvi, exp_appr)
        
        exp_appr = torch.round(exp_appr*10**5)
        exp_appr = exp_appr/(10**5)
        Exp_sum_appr = torch.sum(exp_appr, dim = self.dim, keepdim = True)  

        F = Exp_sum_appr
        w = torch.empty_like(F)
        NndctSoftmaxLOD(F, w)
        m = F/(2**w) 

        lnF = torch.round((22713/(2**15))*(m-1+w)*10**5)/10**5
        uvi = 47274 / (2**15) * (qinput - x_max - lnF)
        exp_appr = torch.empty_like(uvi)
        NndctSoftmaxExpApproximate(uvi, exp_appr)
        exp_appr = torch.round(exp_appr*10**5)/10**5
        output = exp_appr
        output = quantize_tensors([output], self.node)[0]

      # Method 2: Liyi Softmax with any bw
      elif NndctOption.nndct_op_softmax_mode.value == "liyi":
        x_max = torch.max(qinput, dim = self.dim, keepdim = True).values
        qinput = qinput - x_max
        
        exp_appr = torch.empty_like(qinput)
        NndctSoftmaxSimulationPart1(qinput, exp_appr)
        sum = torch.sum(exp_appr, dim = self.dim, keepdim = True)

        sum1 = torch.empty_like(sum)
        NndctSoftmaxSimulationPart2(sum, sum1)
        output = (exp_appr*sum1).bfloat16().float()
        output = quantize_tensors([output], self.node)[0]
      
      # Method 3: Table Look up for AIE2 Softmax with 8 bw and 16 bw(based on LUT)
      elif NndctOption.nndct_op_softmax_mode.value == "aie2_lut_16bw":

        if bw == 8 and fragpos < 2:
          if fragpos < 2:
            qinput_max = torch.max(qinput, dim = self.dim, keepdim = True).values
            qinput -= qinput_max
          else:
            qinput -= 31.75
        else: # bw == 16
          if fragpos < 10:
            qinput_max = torch.max(qinput, dim = self.dim, keepdim = True).values
            qinput -= qinput_max
          else:
            qinput -= 32
        
        exp_appr = torch.empty_like(qinput)
        NndctExpApprAIE2(qinput, exp_appr, bw)
        sum = torch.sum(exp_appr, dim = self.dim, keepdim = True)
        sum_inv = torch.empty_like(sum)
        NndctInverseAIE2(sum, sum_inv)
        output = (exp_appr*sum_inv).bfloat16().float()
        output = quantize_tensors([output], self.node)[0]
      
      # Method 4: Bert with 8 bw
      elif NndctOption.nndct_op_softmax_mode.value == "bert_8bw" or NndctOption.nndct_ip_v70_bert.value:
        def generate_exp_table(bw, input_scale, table_name):

          inputs_table_positive = np.arange(0, 2**(bw - 1)).astype(np.float32)
          inputs_table_negatice = np.arange(-(2**(bw - 1)), 0).astype(np.float32)
          inputs_table = np.hstack((inputs_table_positive, inputs_table_negatice))

          outputs_table = np.exp(inputs_table / (2**(input_scale)))

          return outputs_table

        def aie_reduce_add_v16(acc_v16):
          v2 = np.empty([2]).astype(np.float32)
          acc_v8 = np.empty([8])
          acc_v4 = np.empty([4])
          acc_v2 = np.empty([2])
          acc_v1 = np.empty([1])

          acc_v16 = acc_v16.astype(np.float32)
          for i in range(8):
              v2[0],v2[1] = acc_v16[i],acc_v16[i+8]
              acc_v8[i] = v2.sum()
          for j in range(4):
              v2[0],v2[1] = acc_v8[j],acc_v8[j+4]
              acc_v4[j] = v2.sum()
          for k in range(2):
              v2[0],v2[1] = acc_v4[k],acc_v4[k+2]
              acc_v2[k] = v2.sum()
          for m in range(1):
              v2[0],v2[1] = acc_v2[m],acc_v2[m+1]
              acc_v1[m] = v2.sum()
          return acc_v1

        def compute_inv(x):
          exp_mask     = 0x7F800000
          mantissa_mask= 0x007FFFFF
          mantissa_Q   = 0x00008000
          x_f = x.astype(np.float32)
          B_x = x_f.view(np.uint32)
          exponent = (B_x & exp_mask)>>23
          mantissa = np.where(((B_x & mantissa_Q)==0), ((B_x & mantissa_mask)>>16), ((B_x & mantissa_mask)>>16)+1)
          inv_exponent = 253-exponent
          inv_mantissa = np.round(256*128/(128+mantissa)-128)
          inv_x_val = (np.int32(inv_exponent)<<23) + (np.int32(inv_mantissa)<<16)
          inv_x = inv_x_val.view(np.float32)
          return inv_x
        
        if self.quant_mode <= 1:
          output = super().forward(qinput)
          output = quantize_tensors([output], self.node, method=4)[0]
          return output
        exp_appr = torch.exp(qinput).bfloat16().float()
        exp_appr = torch.where(exp_appr > -128/(2 ** fragpos), exp_appr, 0)


        # # Code for Golden Verification
        # exp_table = generate_exp_table(bw, fragpos, self.node.name)
        # qinput_int = torch.where(qinput >=0, qinput * (2**(fragpos)), qinput * (2**(fragpos)) + 2**bw)
        # exp_arr = exp_table[qinput_int.cpu().numpy().astype(np.int8)]
        # exp_appr = torch.from_numpy(exp_arr).to(input.device)

        if self.dim == -1:       
          sum_v32 = exp_appr.reshape((exp_appr.shape[0],exp_appr.shape[1],exp_appr.shape[2],exp_appr.shape[3]//16,16))
          sum_v32 = sum_v32.permute((0,1,2,4,3))
          sum_v32 = sum_v32.sum(dim = -1)
          sum_v32 = sum_v32.sum(dim = -1, keepdim = True)
          
          
        else:
          sum_v32 = torch.sum(exp_appr, dim = self.dim, keepdim = True).cpu().numpy()

        sum_inv = compute_inv(sum_v32.cpu().numpy())
        sum_inv = torch.from_numpy(sum_inv).bfloat16().float().to(qinput.device)

        output = (exp_appr*sum_inv).bfloat16().float()
        output = quantize_tensors([output], self.node, method=4)[0]

      # Method 0: Quant input and output of Softmax
      else:
        output = super().forward(qinput)
        output = quantize_tensors([output], self.node)[0]
    
    return output


@py_utils.register_quant_op
def Softmax(*args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode == None:
    return torch.nn.Softmax(*args, **kwargs)
  return deephi_Softmax(*args, **kwargs)
