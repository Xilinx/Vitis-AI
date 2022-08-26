

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


# import os
# import sys
import torch
# from ..load_kernels import nndct_kernels
from ..load_kernels import *
import copy
import numpy as np
from nndct_shared.utils import NndctOption, NndctScreenLogger

# from torch.utils.cpp_extension import load
__all__ = ["NndctFixNeuron",
           "NndctDiffsFixPos",\
           "NndctDiffsFixPosChannel",\
           "NndctSigmoidTableLookup",\
           "NndctSigmoidSimulation",\
           "NndctTanhTableLookup",\
           "NndctTanhSimulation", \
           "FixNeuronWithBackward",\
           "fake_quantize_per_tensor",\
           "NndctSoftmaxExpApproximate",\
           "NndctSoftmaxLOD",\
           "NndctSoftmaxSimulationPart1",\
           "NndctSoftmaxSimulationPart2",\
           "fake_quantize_per_channel",\
           "fake_quantize_per_tensor_tensorrt",\
           "fake_quantize_per_channel_tensorrt"]     

def support_onnx_export():
  torch_version = torch.__version__.split('.')
  if int(torch_version[0]) == 1 and int(torch_version[1]) >= 7:
    return True
  else:
    return False
  
class FixNeuronFunc(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input, scale_inv, zero_point, quant_max, method=2):
    ctx.save_for_backward(input)
    return fake_quantize_per_tensor(input, scale_inv, zero_point, -quant_max, quant_max-1, method)
  @staticmethod
  def backward(ctx, grad_output):
    grad_input = grad_scale_inv = grad_zero_point = grad_quant_max = grad_method = None
    if ctx.needs_input_grad[0]:
      grad_input = grad_output
    return grad_input, grad_scale_inv, grad_zero_point, grad_quant_max, grad_method

  
class FixNeuronWithBackward(torch.nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__()

  def forward(self, input, scale_inv, zero_point, quant_max, method=2):
    output = FixNeuronFunc.apply(input, scale_inv, zero_point, quant_max, method)
    return output


def NndctFixNeuron(Tinput, Toutput, maxamp, method=2):
  valmax, valamp = maxamp[0], maxamp[1]
  valmin = -valmax
  valmax = valmax - 1
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  if support_onnx_export():
    Toutput = torch.ops.vai.fix_neuron(Tinput, valmin, valmax, 
                                       valamp, 0, method, device_id, 1)
  else:
    nndct_kernels.FixNeuronV2(Tinput, Toutput, valmin, valmax, 
                              valamp, 0, method, device_id)
  return Toutput
  '''
  if Tinput.device == torch.device("cpu"):
    output = Tinput.cuda()
    nndct_kernels.FixNeuronV2(output, output, valmax,
                              valamp, method)
    Tinput.copy_(output.cpu())
    return Tinput

    # cpu fix neuron
    """
    # output = Tinput.cpu().detach().numpy()
    # output = output * valamp
    # if method == 2:
    #   output = np.where(output > valmax - 1, (valmax - 1), output)
    #   output = np.where(output < (-valmax), -valmax, output)
    #   output = np.where(np.logical_and(output > 0, np.logical_and(np.floor(output) % 2 == 0, output - np.floor(output) == 0.5)), np.ceil(output), output)
    #   output = np.where(output >= 0, np.round(output), output)
    #   output = np.where(np.logical_and(output < 0, output - np.floor(output) == 0.5), np.ceil(output), output)
    #   output = np.where(output < 0, np.round(output), output)

    # elif method == 3:
    #   output = np.where(output > valmax - 1, (valmax - 1), output)
    #   output = np.where(output < (-valmax), -valmax, output)
    #   output = np.where(np.logical_and(output > 0, np.logical_and(np.floor(output) % 2 == 0, output - np.floor(output) == 0.5)), np.ceil(output), output)
    #   output = np.where(output >= 0, np.round(output), output)
    #   output = np.where(np.logical_and(output < 0, np.logical_and(np.ceil(output) % 2 == 0, output - np.floor(output) == 0.5)), np.floor(output), output)
    #   output = np.where(output < 0, np.round(output), output)

    # Tinput.copy_(torch.from_numpy(output))
    # Tinput.div_(valamp)
    # return Tinput
    """
  else:
    nndct_kernels.FixNeuronV2(Tinput, Toutput, valmax,
                              valamp, method)
  return Toutput
  '''


def NndctDiffsFixPos(Tinput, Tbuffer, Tfixpos, bit_width=8, range=5, method=2):
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  if support_onnx_export():
    torch.ops.vai.diffs_fix_pos(Tinput, Tbuffer, Tfixpos, bit_width, range, method, device_id)
  else:
    nndct_kernels.DiffsFixPos(Tinput, Tbuffer, Tfixpos, bit_width, range, method, device_id)

def NndctDiffsFixPosChannel(Tinput, Tbuffer, Tfixpos, axis, bit_width=8, scope=5, method=2):
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  input_split = torch.split(Tinput, 1, dim=axis)
  buffer_split = torch.split(Tbuffer, 1, dim=axis)
  if support_onnx_export():
    for i in range(len(input_split)):
      torch.ops.vai.diffs_fix_pos(input_split[i], buffer_split[i], Tfixpos[i], bit_width, scope, method, device_id)
  else:
    for i in range(len(input_split)):
      nndct_kernels.DiffsFixPos(input_split[i], buffer_split[i], Tfixpos[i], bit_width, scope, method, device_id)

def NndctSigmoidTableLookup(Tinput, Ttable, Toutput, fragpos):
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  if support_onnx_export():
    torch.ops.vai.SigmoidTableLookup(Tinput, Ttable, Toutput, fragpos, device_id)
  else:
    nndct_kernels.SigmoidTableLookup(Tinput, Ttable, Toutput, fragpos, device_id)

def NndctSigmoidSimulation(Tinput, Toutput):
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  if device_id == 1:
    print("Sigmoid simulation dose not support CPU")
  else:
    if support_onnx_export():
      torch.ops.vai.SigmoidSimulation(Tinput, Toutput, device_id)
    else:
      nndct_kernels.SigmoidSimulation(Tinput, Toutput, device_id)

def NndctTanhTableLookup(Tinput, Ttable, Toutput, fragpos):
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  if support_onnx_export():
    torch.ops.vai.TanhTableLookup(Tinput, Ttable, Toutput, fragpos, device_id)
  else:
    nndct_kernels.TanhTableLookup(Tinput, Ttable, Toutput, fragpos, device_id)

def NndctTanhSimulation(Tinput, Toutput):
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  if device_id == 1:
    print("Tanh simulation dose not support CPU")
  else:
    if support_onnx_export():
      torch.ops.vai.TanhSimulation(Tinput, Toutput, device_id)
    else:
      nndct_kernels.TanhSimulation(Tinput, Toutput, device_id)


def NndctSoftmaxExpApproximate(Tinput, Toutput):
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  if device_id == 1:
    print("Softmax Exponent Approximate dose not support CPU")
  else:
    if support_onnx_export():
      torch.ops.vai.SoftmaxExpApproximate(Tinput, Toutput, device_id)
    else:
      nndct_kernels.SoftmaxExpApproximate(Tinput, Toutput, device_id)

def NndctSoftmaxLOD(Tinput, Toutput):
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  if device_id == 1:
    print("Softmax LOD dose not support CPU")
  else:
    if support_onnx_export():
      torch.ops.vai.SoftmaxLOD(Tinput, Toutput, device_id)
    else:
      nndct_kernels.SoftmaxLOD(Tinput, Toutput, device_id)

def NndctSoftmaxSimulationPart1(Tinput, Toutput):
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  if device_id == 1:
    print("Softmax Simulation Part 1 dose not support CPU")
  else:
    if support_onnx_export():
      torch.ops.vai.SoftmaxSimulationPart1(Tinput, Toutput, device_id)
    else:
      nndct_kernels.SoftmaxSimulationPart1(Tinput, Toutput, device_id)

def NndctSoftmaxSimulationPart2(sum, Toutput):
  device_id = 1 if Toutput.device == torch.device("cpu") else 0
  if device_id == 1:
    print("Softmax Simulation Part 2 dose not support CPU")
  else:
    if support_onnx_export():
      torch.ops.vai.SoftmaxSimulationPart2(sum, Toutput, device_id)
    else:
      nndct_kernels.SoftmaxSimulationPart2(sum, Toutput, device_id)


def fake_quantize_per_tensor(input, scale_inv, zero_point, quant_min, quant_max, method, inplace):
  if method == -1:
    return torch.fake_quantize_per_tensor_affine(input, 1.0 / scale_inv, zero_point, quant_min, quant_max)
  else:
    device_id = 1 if input.device == torch.device("cpu") else 0
    if support_onnx_export():
      output = torch.ops.vai.fix_neuron(input, quant_min, quant_max, 
                                        scale_inv, zero_point, method, 
                                        device_id, inplace)
      return output
    else:
      nndct_kernels.FixNeuronV2(input, input, quant_min, 
                                quant_max, scale_inv, zero_point, 
                                method, device_id)
      return input

def fake_quantize_per_channel(input, scale_inv, zero_point, axis, quant_min, quant_max, method, inplace):
  if method == -1:
    if (int(torch.__version__.split('.')[1]) > 9) and (int(torch.__version__.split('.')[0]) > 0):
      zero_point = zero_point.to(torch.int32)
    else:
      zero_point = zero_point.to(torch.long)
    return torch.fake_quantize_per_channel_affine(input, 1.0 / scale_inv, zero_point, axis, quant_min, quant_max)
  else:
    device_id = 1 if input.device == torch.device("cpu") else 0
    input_split = torch.split(input, 1, dim=axis)
    input_cat = []
    if support_onnx_export():
      for i in range(len(input_split)):
        input_cat.append(torch.ops.vai.fix_neuron(input_split[i], quant_min, quant_max, 
                                                  scale_inv[i], zero_point[i], method, 
                                                  device_id, inplace))
      output = torch.cat(input_cat, axis)
      return output
    else:
      for i in range(len(input_split)):
        nndct_kernels.FixNeuronV2(input_split[i], input_split[i], quant_min, 
                                  quant_max, scale_inv, zero_point, 
                                  method, device_id)
        input_cat.append(input_split[i])
      output = torch.cat(input_cat, axis)
      return output
  
def fake_quantize_per_channel_tensorrt(inputs, amax, min_bound, max_bound, axis=None):
  # Computation must be in FP32 to prevent potential over flow.
  if not isinstance(max_bound, torch.Tensor):
    max_bound = torch.tensor(float(max_bound))
  #max_bound = max_bound.double()
  
  input_dtype = inputs.dtype
  if inputs.dtype == torch.half:
    inputs = inputs.float()

  min_amax = amax.min()
  if min_amax < 0:
    raise ValueError("Negative values in amax")

  scale = max_bound / amax

  epsilon = 1. / (1<<24)
  if min_amax <= epsilon:  # Treat amax smaller than minimum representable of fp16 0
    zero_amax_mask = (amax <= epsilon)
    scale[zero_amax_mask] = 0  # Value quantized with amax=0 should all be 0
  
  if axis != None:
    for x_dim in range(inputs.ndim):
      if x_dim != axis:
        scale = torch.unsqueeze(scale, x_dim)

  outputs = torch.clamp((inputs * scale).round_(), min_bound, max_bound)

  if min_amax <= epsilon:
    scale[zero_amax_mask] = 1.  # Return 1 makes more sense for values quantized to 0 with amax=0

  if input_dtype == torch.half:
    outputs = outputs.half()
  outputs = outputs / scale
  return outputs

def fake_quantize_per_tensor_tensorrt(inputs, amax, min_bound, max_bound):
  # Computation must be in FP32 to prevent potential over flow.
  if not isinstance(max_bound, torch.Tensor):
    max_bound = torch.tensor(float(max_bound))
  #max_bound = max_bound.double()
  
  if not isinstance(amax, torch.Tensor):
    amax = torch.tensor(float(amax))
  #amax = amax.double()
  
  input_dtype = inputs.dtype
  if inputs.dtype == torch.half:
    inputs = inputs.float()

  if amax < 0:
    raise ValueError("Negative values in amax")

  scale = max_bound / amax
  epsilon = 1. / (1<<24)
  if amax <= epsilon:  # Treat amax smaller than minimum representable of fp16 0
    scale = 0  # Value quantized with amax=0 should all be 0

  outputs = torch.clamp((inputs * scale).round_(), min_bound, max_bound)

  if amax <= epsilon:
    scale = 1.  # Return 1 makes more sense for values quantized to 0 with amax=0

  if input_dtype == torch.half:
    outputs = outputs.half()
  
  outputs = outputs / scale
  return outputs

# def fake_quantize_tensorrt(inputs, amax, num_bits=8, unsigned=False, narrow_range=True):
#     """Shared function body between TensorQuantFunction and FakeTensorQuantFunction"""
#     # Fine scale, per channel scale will be handled by broadcasting, which could be tricky. Pop a warning.
#     if isinstance(amax, torch.Tensor) and inputs.dim() != amax.dim():
#       logging.debug("amax %s has different shape than inputs %s. Make sure broadcast works as expected!",
#                     amax.size(), inputs.size())

#     if unsigned:
#       if inputs.min() < 0.:
#         raise TypeError("Negative values encountered in unsigned quantization.")

#     # Computation must be in FP32 to prevent potential over flow.
#     input_dtype = inputs.dtype
#     if inputs.dtype == torch.half:
#       inputs = inputs.float()
#     if amax.dtype == torch.half:
#       amax = amax.float()

#     min_amax = amax.min()
#     if min_amax < 0:
#       raise ValueError("Negative values in amax")

#     max_bound = torch.tensor((2.0**(num_bits - 1 + int(unsigned))) - 1.0, device=amax.device)
#     if unsigned:
#       min_bound = 0
#     elif narrow_range:
#       min_bound = -max_bound
#     else:
#       min_bound = -max_bound - 1
#     scale = max_bound / amax

#     epsilon = 1. / (1<<24)
#     if min_amax <= epsilon:  # Treat amax smaller than minimum representable of fp16 0
#       zero_amax_mask = (amax <= epsilon)
#       scale[zero_amax_mask] = 0  # Value quantized with amax=0 should all be 0

#     outputs = torch.clamp((inputs * scale).round_(), min_bound, max_bound)

#     if min_amax <= epsilon:
#       scale[zero_amax_mask] = 1.  # Return 1 makes more sense for values quantized to 0 with amax=0

#     if input_dtype == torch.half:
#       outputs = outputs.half()

#     return outputs, scale
