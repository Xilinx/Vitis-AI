# Copyright 2022 Xilinx Inc.
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
import numpy as np
from nndct_shared.base import NNDCT_OP
from nndct_shared.utils import PatternType
from nndct_shared.nndct_graph import GraphSearcher, Tensor
from pytorch_nndct.parse.torch_op_def import TorchConvTranspose2d, TorchConv2d
from .device import DeviceInfo, DeviceType
from .target_helper import DPUTargetHelper

# refer to xcomplier src/pass/passes/PartitionPass.cpp
def check_bilinear_upsample_fake_weight(node, scale_h, scale_w):
  msg = ""
  ic = node.in_tensors[0].shape[3]
  wsize = (2 * scale_h) * (2 * scale_w) * ic
  weights = [0.0] * wsize
  kernel_h = 2 * scale_h
  kernel_w = 2 * scale_w
  half_pixel_centers = node.node_attr(node.op.AttrName.HALF_PIXEL_CENTERS)
  if half_pixel_centers:
    h_s = math.ceil(scale_h * 0.5 - 0.5)
    w_s = math.ceil(scale_w * 0.5 - 0.5)
    delta = 0.5
    scale2pos = {2: 4, 4: 6}
  else:
    h_s = 0
    w_s = 0
    delta = 0.5
    scale2pos = {2: 2, 4: 4, 8: 6}
  
  for j in range(h_s, scale_h + h_s):
    for i in range(w_s, scale_w + w_s):
      lerp_x = (i + delta) / scale_w - delta
      lerp_x_f = lerp_x - math.floor(lerp_x)
      lerp_y = (j + delta) / scale_h - delta
      lerp_y_f = lerp_y - math.floor(lerp_y)
      right_idx = kernel_w - 1 - i + w_s
      left_idx = kernel_w - 1 - scale_w - i + w_s
      bottom_idx = kernel_h - 1 - j + h_s
      top_idx = kernel_h - 1 - scale_h - j + h_s
      for c in range(ic):
        weights[top_idx * kernel_w * ic + left_idx * ic + c] = (1 - lerp_x_f) * (1 - lerp_y_f)
        weights[top_idx * kernel_w * ic + right_idx * ic + c] = lerp_x_f * (1 - lerp_y_f)
        weights[bottom_idx * kernel_w * ic + left_idx * ic + c] = (1 - lerp_x_f) * lerp_y_f
        weights[bottom_idx * kernel_w * ic + right_idx * ic + c] = lerp_x_f * lerp_y_f
  
  bit_width = 8
  bound_low = 0
  bound_hight = 2 ** bit_width - 1 
  
  def is_within_bound(val):
    fix_val = val * (2 ** scale2pos[scale_h])
    return not(fix_val > bound_hight or fix_val < bound_low or (fix_val == 0 and val != 0))
  
  within_bound = all(map(is_within_bound, weights))
  if not within_bound:
    msg = "weights fixed data out of range."
    return False, msg

  return True, msg

  
def check_bilinear_upsample_scale(node):
  msg = ""
  input_shape = node.in_tensors[0].shape
  output_shape = node.out_tensors[0].shape
  i_h = input_shape[1]
  i_w = input_shape[2]
  o_h = output_shape[1]
  o_w = output_shape[2]
  scale_f = [1.0, 1.0] # [scale_w, scale_h]
  scale = []
  scale_f[0] = float(o_w) / float(i_w)
  scale_f[1] = float(o_h) / float(i_h)
  half_pixel_centers = node.node_attr(node.op.AttrName.HALF_PIXEL_CENTERS)
  if half_pixel_centers:
    allowed_scale = [2, 4]
  else:
    allowed_scale = [2, 4, 8]
    
  for s_f in scale_f:
    if not (math.ceil(s_f) == s_f and math.floor(s_f) == s_f and
            any([s== s_f for s in allowed_scale])):
      msg = f"{node.op.type} output / input scale is {scale_f}"
      return False, msg
  
    scale.append(int(s_f))
  
  if not all([scale[0] == s for s in scale]):
    msg = "scale_w is not equal with scale_h"  
    return False, msg
  
  ret, msg = check_bilinear_upsample_fake_weight(node, scale[1], scale[0])
  if not ret:
    return ret, msg
  
  node.set_node_attr(node.op.AttrName.SCALE, [float(scale[0]), float(scale[1])])
  return True, msg
  
def create_transpose_dwconv2d_from_bilinear_upsample(node):
  transpose_dwconv2d = TorchConvTranspose2d(NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D)
  scale_w, scale_h = node.node_attr(node.op.AttrName.SCALE)
  scale_w = int(scale_w)
  scale_h = int(scale_h)
  half_pixel_centers = node.node_attr(node.op.AttrName.HALF_PIXEL_CENTERS)
  input_shape = node.in_tensors[0].shape
  output_shape = node.out_tensors[0].shape
  kernel_h = 2 * scale_h
  kernel_w = 2 * scale_w
  transpose_dwconv2d.set_config('output_padding', [0, 0])
  transpose_dwconv2d.set_config('kernel_size', [kernel_h, kernel_w])
  transpose_dwconv2d.set_config('stride', [scale_h, scale_w])
  input_w = input_shape[2] + 2
  input_h = input_shape[1] + 2
  if half_pixel_centers:
    pad_l = int(math.floor(float(scale_w) / 2.0 - 0.5))
    pad_r = output_shape[2] + int(kernel_w) - 2 - (input_w - 1) * scale_w - pad_l
    pad_t = int(math.floor(float(scale_h) / 2.0 - 0.5))
    pad_b = output_shape[1] + int(kernel_h) - 2 - (input_h - 1) * scale_h - pad_t
  else:
    pad_l = scale_w - 1
    pad_r = scale_w - 1
    pad_t = scale_h - 1
    pad_b = scale_h - 1

  padding = [int(kernel_w) - 1 - pad_l, 
             int(kernel_w) - 1 - pad_r,
             int(kernel_h) - 1 - pad_t, 
             int(kernel_h) - 1 - pad_b]

  transpose_dwconv2d.set_attr(transpose_dwconv2d.AttrName.PAD_MODE, 0)
  transpose_dwconv2d.set_attr(transpose_dwconv2d.AttrName.PAD, padding)
  return transpose_dwconv2d

def check_nonlinear(engine, node):
  op_nonlinear_map = {
    NNDCT_OP.CONV2D: [NNDCT_OP.RELU, NNDCT_OP.RELU6, NNDCT_OP.PRELU, NNDCT_OP.LEAKY_RELU, NNDCT_OP.HSWISH, NNDCT_OP.HSIGMOID],
    NNDCT_OP.CONVTRANSPOSE2D: [NNDCT_OP.RELU, NNDCT_OP.RELU6, NNDCT_OP.PRELU, NNDCT_OP.LEAKY_RELU],
    NNDCT_OP.DEPTHWISE_CONV2D: [NNDCT_OP.RELU, NNDCT_OP.RELU6, NNDCT_OP.PRELU, NNDCT_OP.LEAKY_RELU],
    NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D: [],
    
  }
  msg = ""
  nonlinear_types = op_nonlinear_map.get(node.op.type)
  if nonlinear_types is None:
    nonlinear_types = []
  children_nodes = node.owning_graph.children(node)
  if len(children_nodes) == 1 and children_nodes[0].op.type in nonlinear_types:
    if children_nodes[0].op.type == NNDCT_OP.LEAKY_RELU:
      alpha = children_nodes[0].node_attr(children_nodes[0].op.AttrName.ALPHA)
      dpu_alpha = 26.0 / 256
      if alpha != dpu_alpha:
        msg = f"Its alpa is {alpha}, but DPU only support {dpu_alpha}."
        return False, msg
    
    children_nodes[0].target_device = DeviceInfo(DeviceType.DPU)

  return True, msg

def check_kernel(kernels, kernel_limit):
  msg = ""
  if any([k not in kernel_limit for k in kernels]):
    msg = f"'kernel'({kernels[0]} x {kernels[1]}) is not in DPU supported range({kernel_limit})."
    return False, msg
  return True, msg


def check_stride(strides, stride_limit):
  msg = ""
  if any([s not in stride_limit for s in stride_limit]):
    msg = f"'stride'({strides}) is not in DPU supported range({stride_limit})."
    return False, msg
  return True, msg

def check_load_jump_write(ic, channel_parallel, dilation=None):
  msg = ""
  dilation = dilation if dilation is not None else [1, 1]
  cp_limit = 256 * channel_parallel
  if ic > cp_limit:
    msg = f"DPU only supports 'input_channel'({ic}) less than ({cp_limit})"
    return False, msg
  return True, msg

def check_save_jump_read(oc, channel_parallel):
  msg = ""
  cp_limit = 256 * channel_parallel
  if oc > cp_limit:
    msg = f"DPU only support 'output_channel'({oc}) less than {cp_limit}"
    return False, msg
  return True, msg

def check_pad(pad, kernel):
  msg = ""
  if any([p < 0 for p in pad]):
    msg = f"DPU only support non-negative 'pad'({pad})"
    return False, msg

  if pad[0] > kernel[0]:
    msg = f"DPU only supports 'pad_left'({pad[0]}) less than 'kernel_width'({kernel[0]})"
    return False, msg
  
  if pad[1] > kernel[0]:
    msg = f"DPU only supports 'pad_right'({pad[1]}) less than 'kernel_width'({kernel[0]})"
    return False, msg
  
  if pad[2] > kernel[1]:
    msg = f"DPU only supports 'pad_top'({pad[2]}) less than 'kernel_width'({kernel[1]})"
    return False, msg

  if pad[3] > kernel[1]:
    msg = f"DPU only supports 'pad_bottom'({pad[3]}) less than 'kernel_width'({kernel[1]})"
    return False, msg
  
  return True, msg

def check_pad_with_limit(pad, kernel, pad_limit):
  msg = ""
  if any([p < 0 for p in pad]):
    msg = f"DPU only support non-negative 'pad'({pad})"
    return False, msg

  pad_idx_kernel_map = {
    "pad_left": [0, 0],  # [pad_idx, kernel_idx]
    "pad_right": [1, 0],
    "pad_top": [2, 1],
    "pad_bottom": [3, 1]
  }


  for key, pad_idx_kernel_idx in pad_idx_kernel_map.items():
    pad_idx, kernel_idx = pad_idx_kernel_idx
    if pad_limit[key]:
      if pad[pad_idx] not in pad_limit[key]:
        msg = f"{key}({pad[pad_idx]}) is not in range."
        return False, msg
    else:
      if pad[pad_idx] > kernel[kernel_idx]:
        msg = f"DPU only supports {key}({pad[pad_idx]}) less than 'kernel'({kernel[kernel_idx]})."
        return False, msg
    
  return True, msg

    
def check_conv_weights_bank_depth(target, engine, kernel_shape):
  msg = ""
  weight_bank_name = engine.weight_bank
  bank_groups = DPUTargetHelper.get_bank_group(target)
  weights_bank = None
  for bank_group in bank_groups:
    if bank_group.name  == weight_bank_name:
      weights_bank = bank_group
      break

  if weights_bank is None:
    msg = f"{target.get_name()}'s bank group configure is error, there's no weights bank for the engine."
    return False, msg

  output_channel_parallel = engine.output_channel_parallel
  k_oc, k_h, k_w, k_ic = kernel_shape
  weight_depth = k_w * k_h * math.ceil(k_ic * 1.0 / weights_bank.bank_width) * math.ceil(output_channel_parallel * 1.0 / weights_bank.bank_num)
  if weight_depth > weights_bank.bank_depth:
    msg = f"Weights({kernel_shape}) is too large to be loaded into parameter buffer. 'kernel_h * kernel_w  *  ⌈input_channel / weights_bank_width⌉ * ⌈output_channel_parallel / weights_bank_num⌉({weight_depth})' is supporsed to be less equal than {weights_bank.bank_depth}."
    return False, msg
  
  return True, msg
 
def check_dwconv_weights_bank_depth(target, engine, kernel_shape):
  msg = ""
  weight_bank_name = engine.weight_bank
  bank_groups = target.get_bank_group()
  weights_bank = None
  for bank_group in bank_groups:
    if bank_group.name  == weight_bank_name:
      weights_bank = bank_group

  if weights_bank is None:
    msg = f"{target.get_name()}'s bank group configure is error, there's no weights bank for the engine."
    return False, msg

  channel_parallel = engine.channel_parallel
  k_oc, k_h, k_w, k_ic = kernel_shape
  weight_depth = k_w * k_h  * math.ceil(channel_parallel * 1.0 / weights_bank.bank_width)
  if weight_depth > weights_bank.bank_depth:
    msg = f"Weights({kernel_shape}) is too large to be loaded into parameter buffer. 'kernel_h * kernel_w  *  input_channel' is supporsed to be less equal than {weights_bank.bank_depth * weights_bank.bank_width}."
    return False, msg

  return True, msg

def check_transposed_kernel(kernel, stride, limit):
  msg = ""
  if not (kernel // stride  in limit and (kernel % stride == 0 or (kernel // stride + 1) in limit)):
    msg = f"'kernel / stride'({kernel} / {stride}) is not in DPU supported range{limit}."
    return False, msg
  return True, msg

def check_pool_engine(target):
  msg = ""
  if not (DPUTargetHelper.has_pool_engine(target) or DPUTargetHelper.has_alu_engine(target)):
    msg = f"{DPUTargetHelper.get_name(target)} does not have pool-engine."
    return False, msg
  return True, msg

def check_dwconv_engine(target):
  msg = ""
  if not (DPUTargetHelper.has_dwconv_engine(target) or DPUTargetHelper.has_alu_engine(target)):
    msg = f"{DPUTargetHelper.get_name(target)} does not have depthwise-conv-engine."
    return False, msg
  return True, msg

def check_eltwise_engine(target):
  msg = ""
  if not DPUTargetHelper.has_eltwise_engine(target):
    msg = f"{DPUTargetHelper.get_name(target)} does not have eltwise-engine"
    return False, msg
  return True, msg



def filter_conv2d(node, target):
  msg = ""
  ksize = node.node_attr(node.op.AttrName.KERNEL)
  strides = node.node_attr(node.op.AttrName.STRIDE)
  dilation = node.node_attr(node.op.AttrName.DILATION)
  padding = node.node_attr(node.op.AttrName.PAD)

  conv_engine = DPUTargetHelper.get_conv_engine(target)
  channel_parallel = conv_engine.input_channel_parallel
  ic = node.in_tensors[0].shape[3]
  oc = node.out_tensors[0].shape[3]
  dilated_ksize = list(ksize)
  for i in range(len(dilated_ksize)):
    dilated_ksize[i] = (ksize[i] - 1) * dilation[i] + 1
  
  kernel_limit = DPUTargetHelper.parse_range("1-16")

  if DPUTargetHelper.has_attr(conv_engine, "conv_limit") and conv_engine.conv_limit.kernel_size:
    kernel_limit = DPUTargetHelper.parse_range(conv_engine.conv_limit.kernel_size)
  
  ret, msg = check_kernel(ksize, kernel_limit)

  if not ret:
    return ret, msg
  
  ret, msg = check_conv_weights_bank_depth(target, conv_engine, node.op.get_param(node.op.ParamName.WEIGHTS).shape)

  if not ret:
    return ret, msg
  
  stride_limit = DPUTargetHelper.parse_range("1-4")
  if DPUTargetHelper.has_attr(conv_engine, "conv_limit") and conv_engine.conv_limit.stride:
    stride_limit = DPUTargetHelper.parse_range(conv_engine.conv_limit.stride)

  iw = node.in_tensors[0].shape[2]
  ih = node.in_tensors[0].shape[1]

  if iw != ksize[0] or ih != ksize[1]:
    ret, msg = check_stride(strides, stride_limit)
  
  if not ret:
    return ret, msg

  ret, msg = check_load_jump_write(ic, channel_parallel, dilation)

  if not ret:
    return ret, msg
  
  ret, msg = check_pad(padding, dilated_ksize)

  if not ret:
    return ret, msg

  # ret, msg = check_nonlinear(conv_engine, node)

  # if not ret:
  #   return ret, msg

  return True, msg




def filter_depthwise_conv2d(node, target):
  msg = ""
  ret, msg = check_dwconv_engine(target)
  if not ret:
    return ret, msg
  ksize = node.node_attr(node.op.AttrName.KERNEL)
  strides = node.node_attr(node.op.AttrName.STRIDE)
  dilation = node.node_attr(node.op.AttrName.DILATION)
  padding = node.node_attr(node.op.AttrName.PAD)
  ic = node.in_tensors[0].shape[3]
  oc = node.out_tensors[0].shape[3]
  dilated_ksize = list(ksize)
  for i in range(len(dilated_ksize)):
    dilated_ksize[i] = (ksize[i] - 1) * dilation[i] + 1
  
  kernel_limit = DPUTargetHelper.parse_range("1-16")
  stride_limit = DPUTargetHelper.parse_range("1-4")
  pad_limit = {}
  if DPUTargetHelper.has_alu_engine(target):
    alu_engine = DPUTargetHelper.get_alu_engine(target)
    channel_parallel = alu_engine.channel_parallel
    if DPUTargetHelper.has_attr(alu_engine, "alu_limit"):
      alu_limit = alu_engine.alu_limit
      if alu_limit.kernel_size:
        kernel_limit = DPUTargetHelper.parse_range(alu_limit.kernel_size)
      if alu_limit.stride:
        stride_limit = DPUTargetHelper.parse_range(alu_limit.stride)

    if DPUTargetHelper.has_attr(alu_engine, "pad_limit"):
      alu_pad_limit = alu_engine.pad_limit
      if alu_pad_limit.pad_left:
        pad_limit["pad_left"] = DPUTargetHelper.parse_range(alu_pad_limit.pad_left)
      if alu_pad_limit.pad_right:
        pad_limit["pad_right"] = DPUTargetHelper.parse_range(alu_pad_limit.pad_right)
      
      if alu_pad_limit.pad_top:
        pad_limit["pad_top"] = DPUTargetHelper.parse_range(alu_pad_limit.pad_top)

      if alu_pad_limit.pad_bottom:
        pad_limit["pad_bottom"] = DPUTargetHelper.parse_range(alu_pad_limit.pad_bottom)
  else:
    dwconv_engine = DPUTargetHelper.get_dwconv_engine(target)
    channel_parallel = dwconv_engine.channel_parallel
    if DPUTargetHelper.has_attr(dwconv_engine, "dwconv_limit"):
      dwconv_limit = dwconv_engine.dwconv_limit
      if dwconv_limit.kernel_size:
        kernel_limit = DPUTargetHelper.parse_range(dwconv_limit.kernel_size)
      if dwconv_limit.stride:
        stride_limit = DPUTargetHelper.parse_range(dwconv_limit.stride)
    
    if DPUTargetHelper.get_type(target) == "DPUCAHX8H":
      if strides[0] > ksize[0]:
        msg =  f"The stride_w({strides[0]}) > kernel_w({ksize[0]}), but {DPUTargetHelper.get_name(target)} only support stride_w <= kernel_w."
        return False, msg

  ret, msg = check_kernel(ksize, kernel_limit)
  if not ret:
    return ret, msg
  
  iw = node.in_tensors[0].shape[2]
  ih = node.in_tensors[0].shape[1]

  if not(iw == ksize[0] and ih == ksize[1]):
    ret, msg = check_stride(strides, stride_limit)
    if not ret:
      return ret, msg
  ret, msg = check_load_jump_write(ic, channel_parallel, dilation)
  if not ret:
    return ret, msg
  
  ret, msg = check_save_jump_read(oc, channel_parallel)
  if not ret:
    return ret, msg

  if DPUTargetHelper.has_alu_engine(target):
    ret, msg = check_dwconv_weights_bank_depth(target, DPUTargetHelper.get_alu_engine(target), node.op.get_param(node.op.ParamName.WEIGHTS).shape)
    if not ret:
      return ret, msg
  else:
    ret, msg = check_dwconv_weights_bank_depth(target, DPUTargetHelper.get_dwconv_engine(target), node.op.get_param(node.op.ParamName.WEIGHTS).shape)
    if not ret:
      return ret, msg

  if pad_limit:
    ret, msg = check_pad_with_limit(padding, dilated_ksize, pad_limit)
    if not ret:
      return ret, msg
  else:
    ret, msg = check_pad(padding, dilated_ksize)
    if not ret:
      return ret, msg

  # if DPUTargetHelper.has_alu_engine(target):
  #   ret, msg = check_nonlinear(DPUTargetHelper.get_alu_engine(target), node)
  #   if not ret:
  #     return ret, msg
  # else:
  #   ret, msg = check_nonlinear(DPUTargetHelper.get_dwconv_engine(target), node)
  #   if not ret:
  #     return ret, msg
  return True, msg

def filter_transpose_conv2d(node, target):
  msg = ""
  ksize = node.node_attr(node.op.AttrName.KERNEL)
  strides = node.node_attr(node.op.AttrName.STRIDE)
  dilation = node.node_attr(node.op.AttrName.DILATION)
  padding = node.node_attr(node.op.AttrName.PAD)

  output_padding = node.node_config("output_padding")
  if any([pad != 0 for pad in output_padding]):
    msg = "DPU does not support output_padding."
    return False, msg

  conv_engine = DPUTargetHelper.get_conv_engine(target)
  channel_parallel = conv_engine.input_channel_parallel

  ic = node.in_tensors[0].shape[3]
  oc = node.out_tensors[0].shape[3]

  kernel_limit = DPUTargetHelper.parse_range("1-16")
  if DPUTargetHelper.has_attr(conv_engine, "conv_limit") and conv_engine.conv_limit.kernel_size:
    kernel_limit = DPUTargetHelper.parse_range(conv_engine.conv_limit.kernel_size)

  ret, msg = check_transposed_kernel(ksize[0], strides[0], kernel_limit)
  if not ret:
    return ret, msg
  ret, msg = check_transposed_kernel(ksize[1], strides[1], kernel_limit)
  if not ret:
    return ret, msg

  ret, msg = check_conv_weights_bank_depth(target, conv_engine, node.op.get_param(node.op.ParamName.WEIGHTS).shape)

  if not ret:
    return ret, msg

  ret, msg = check_load_jump_write(ic, channel_parallel, dilation)
  if not ret:
    return ret, msg

  ret, msg = check_save_jump_read(oc, channel_parallel)
  if not ret:
    return ret, msg

  ret, msg = check_pad(padding, ksize)
  if not ret:
    return ret, msg

  # ret, msg = check_nonlinear(conv_engine, node)
  # if not ret:
  #   return ret, msg

  return True, msg


def filter_transpose_depthwise_conv2d(node, target):
  msg = ""
  ret, msg = check_dwconv_engine(target)
  if not ret:
    return ret, msg
  ksize = node.node_attr(node.op.AttrName.KERNEL)
  strides = node.node_attr(node.op.AttrName.STRIDE)
  padding = node.node_attr(node.op.AttrName.PAD)
  dilation = node.node_attr(node.op.AttrName.DILATION)
  ic = node.in_tensors[0].shape[3]
  oc = node.out_tensors[0].shape[3]

  output_padding = node.node_config("output_padding")
  if any([pad != 0 for pad in output_padding]):
    msg = "DPU does not support output_padding."
    return False, msg
 
  kernel_limit = DPUTargetHelper.parse_range("1-16")
  stride_limit = DPUTargetHelper.parse_range("1-4")
  pad_limit = {}
  if DPUTargetHelper.has_alu_engine(target):
    alu_engine = DPUTargetHelper.get_alu_engine(target)
    channel_parallel = alu_engine.channel_parallel
    if DPUTargetHelper.has_attr(alu_engine, "alu_limit"):
      alu_limit = alu_engine.alu_limit
      if alu_limit.kernel_size:
        kernel_limit = DPUTargetHelper.parse_range(alu_limit.kernel_size)
      if alu_limit.stride:
        stride_limit = DPUTargetHelper.parse_range(alu_limit.stride)

    if DPUTargetHelper.has_attr(alu_engine, "pad_limit"):
      alu_pad_limit = alu_engine.pad_limit
      if alu_pad_limit.pad_left:
        pad_limit["pad_left"] = DPUTargetHelper.parse_range(alu_pad_limit.pad_left)
      if alu_pad_limit.pad_right:
        pad_limit["pad_right"] = DPUTargetHelper.parse_range(alu_pad_limit.pad_right)
      
      if alu_pad_limit.pad_top:
        pad_limit["pad_top"] = DPUTargetHelper.parse_range(alu_pad_limit.pad_top)

      if alu_pad_limit.pad_bottom:
        pad_limit["pad_bottom"] = DPUTargetHelper.parse_range(alu_pad_limit.pad_bottom)
  else:
    dwconv_engine = DPUTargetHelper.get_dwconv_engine(target)
    channel_parallel = dwconv_engine.channel_parallel
    if DPUTargetHelper.has_attr(dwconv_engine, "dwconv_limit"):
      dwconv_limit = dwconv_engine.dwconv_limit
      if dwconv_limit.kernel_size:
        kernel_limit = DPUTargetHelper.parse_range(dwconv_limit.kernel_size)
      if dwconv_limit.stride:
        stride_limit = DPUTargetHelper.parse_range(dwconv_limit.stride)
  ret, msg = check_transposed_kernel(ksize[0], strides[0], kernel_limit)
  if not ret:
    return ret, msg
  
  ret, msg = check_transposed_kernel(ksize[1], strides[1], kernel_limit)
  if not ret:
    return ret, msg
  ret, msg = check_stride([1, 1], stride_limit)
  if not ret:
    return ret, msg

  ret, msg = check_load_jump_write(ic, channel_parallel, dilation)
  if not ret:
    return ret, msg

  ret, msg = check_save_jump_read(oc, channel_parallel)

  if pad_limit:
    ret, msg = check_pad_with_limit(padding, ksize, pad_limit)
    if not ret:
      return ret, msg
  else:
    ret, msg = check_pad(padding, ksize)
    if not ret:
      return ret, msg

  # if DPUTargetHelper.has_alu_engine(target):
  #   ret, msg = check_nonlinear(DPUTargetHelper.get_alu_engine(target), node)
  #   if not ret:
  #     return ret, msg
  # else:
  #   ret, msg = check_nonlinear(DPUTargetHelper.get_dwconv_engine(target), node)
  #   if not ret:
  #     return ret, msg 
  return True, msg

def filter_conv3d(node, target):
  msg = ""
  if DPUTargetHelper.get_type(target) != "DPUCVDX8G":
    msg = f"{DPUTargetHelper.get_name(target)} does not support {node.op.type}. Only DPUCVDX8G support this."
    return False, msg
  return True, msg

def filter_depthwise_conv3d(node, target):
  msg = ""
  if DPUTargetHelper.get_type(target) != "DPUCVDX8G":
    msg = f"{DPUTargetHelper.get_name(target)} does not support {node.op.type}. Only DPUCVDX8G support this."
    return False, msg
  return True, msg


def filter_transpose_conv3d(node, target):
  msg = ""
  if DPUTargetHelper.get_type(target) != "DPUCVDX8G":
    msg = f"{DPUTargetHelper.get_name(target)} does not support {node.op.type}. Only DPUCVDX8G support this."
    return False, msg

  output_padding = node.node_config("output_padding")
  if any([pad != 0 for pad in output_padding]):
    msg = "DPU does not support output_padding."
    return False, msg

  return True, msg


def filter_transpose_depthwise_conv3d(node, target):
  msg = ""
  if DPUTargetHelper.get_type(target) != "DPUCVDX8G":
    msg = f"{DPUTargetHelper.get_name(target)} does not support {node.op.type}. Only DPUCVDX8G support this."
    return False, msg
  
  output_padding = node.node_config("output_padding")
  if any([pad != 0 for pad in output_padding]):
    msg = "DPU does not support output_padding."
    return False, msg
  
  return True, msg



def filter_pool(node, target):
  msg = ""
  ret, msg = check_pool_engine(target)
  if not ret:
    return ret, msg

  avg_pool_type = [NNDCT_OP.AVG_POOL, NNDCT_OP.ADAPTIVEAVGPOOL2D]
  max_pool_type = [NNDCT_OP.MAX_POOL]

  ksize = node.node_attr(node.op.AttrName.KERNEL)
  strides = node.node_attr(node.op.AttrName.STRIDE)
  padding = node.node_attr(node.op.AttrName.PAD)

  if DPUTargetHelper.has_alu_engine(target):
    alu_engine = DPUTargetHelper.get_alu_engine(target)
    support_list = alu_engine.alu_type
    has_max = any([t == alu_engine.max_pool for t in support_list])
    has_avg = any([t == alu_engine.avg_pool for t in support_list])
    has_max_reduce = any([t == alu_engine.max_reduce for t in support_list])
  else:
    pool_engine = DPUTargetHelper.get_pool_engine(target)
    support_list = pool_engine.pool_type
    has_max = any([t == pool_engine.max for t in support_list])
    has_avg = any([t == pool_engine.avg for t in support_list])
    has_max_reduce = any([t == pool_engine.max_reduce for t in support_list])

  if not ((node.op.type in max_pool_type and (has_max or has_max_reduce)) or (node.op.type in avg_pool_type and has_avg)):
    msg = f"{DPUTargetHelper.get_name(target)} does not support {node.op.type}."
    return False, msg
  
  kernel_limit = DPUTargetHelper.parse_range("1-8")
  stride_limit = DPUTargetHelper.parse_range("1-8")
  pad_limit = {}
  if DPUTargetHelper.has_alu_engine(target):
    alu_engine = DPUTargetHelper.get_alu_engine(target)
    if DPUTargetHelper.has_attr(alu_engine, "alu_limit"):
      alu_limit = alu_engine.alu_limit
      if alu_limit.kernel_size:
        kernel_limit = DPUTargetHelper.parse_range(alu_limit.kernel_size)
      if alu_limit.stride:
        stride_limit = DPUTargetHelper.parse_range(alu_limit.stride)

    if DPUTargetHelper.has_attr(alu_engine, "pad_limit"):
      alu_pad_limit = alu_engine.pad_limit
      if alu_pad_limit.pad_left:
        pad_limit["pad_left"] = DPUTargetHelper.parse_range(alu_pad_limit.pad_left)
      if alu_pad_limit.pad_right:
        pad_limit["pad_right"] = DPUTargetHelper.parse_range(alu_pad_limit.pad_right)
      if alu_pad_limit.pad_top:
        pad_limit["pad_top"] = DPUTargetHelper.parse_range(alu_pad_limit.pad_top)
      if alu_pad_limit.pad_bottom:
        pad_limit["pad_bottom"] = DPUTargetHelper.parse_range(alu_pad_limit.pad_bottom)
  elif node.op.type in avg_pool_type:
    if ksize[0] != ksize[1]:
      msg = f"DPU only supports avgpool with square kernel, but this op has kernel {ksize[0]} x {ksize[1]}."
      return False, msg

    pool_engine = DPUTargetHelper.get_pool_engine(target)
    if DPUTargetHelper.has_attr(pool_engine, "avg_limit"):
      avg_limit = pool_engine.avg_limit
      if avg_limit.kernel_size:
        kernel_limit = DPUTargetHelper.parse_range(avg_limit.kernel_size)
      if avg_limit.stride:
        stride_limit = DPUTargetHelper.parse_range(avg_limit.stride)
  elif node.op.type in max_pool_type:
    pool_engine = DPUTargetHelper.get_pool_engine(target)
    if DPUTargetHelper.has_attr(pool_engine, "max_limit"):
      max_limit = pool_engine.max_limit
      if max_limit.kernel_size:
        kernel_limit = DPUTargetHelper.parse_range(max_limit.kernel_size)
      if max_limit.stride:
        stride_limit = DPUTargetHelper.parse_range(max_limit.stride)
  
  if node.op.type in max_pool_type and has_max_reduce and ksize[0] not in kernel_limit:
    if ksize[0] > 100:
      msg = f"'kernel_width'({ksize[0]}) is not in DPU supported range [1, 100]"
      return False, msg
    if ksize[1] < 1 or ksizse[1] > 2:
      msg = f"'kernel_height'({ksize[1]}) is not in DPU supported range [1, 2]"
      return False, msg
  else:
    ret, msg = check_kernel(ksize, kernel_limit)
    if not ret:
      return ret, msg
    iw = node.in_tensors[0].shape[2]
    ih = node.in_tensors[0].shape[1]
    if iw != ksize[0] or ih != ksize[1]:
      ret, msg = check_stride(strides, stride_limit)
      if not ret:
        return ret, msg
    
    if pad_limit:
      ret, msg = check_pad_with_limit(padding, ksize, pad_limit)
      if not ret:
        return ret, msg
    else:
      ret, msg = check_pad(padding, ksize)
      if not ret:
        return ret, msg
  return True, msg


def filter_eltwise(node, target):
  msg = ""
  if node.op.type == NNDCT_OP.MULTIPLY and node.in_tensors[1].node.op.type in [NNDCT_OP.CONST, NNDCT_OP.TENSOR]:
    prefix_msg = "Try to convert mul to DepthwiseConv2d failed."
    ret, check_msg = check_dim_of_inputs_of_mul(node)
    if not ret:
      msg = prefix_msg + check_msg
    else: 
      dwconv2d = create_dwconv2d_from_mul(node)
      mul_op = node.op
      node.op = dwconv2d
      input_tensor = node.in_tensors[0]  
      out_tensor = node.out_tensors[0]
      old_shape = input_tensor.shape
      new_shape = list(old_shape)
      for i in range(4 - len(old_shape)):
        new_shape.insert(0, 1)
      input_tensor.shape = new_shape
      out_tensor.shape = new_shape
      ret, check_msg = filter_depthwise_conv2d(node, target)
      node.op = mul_op
      input_tensor.shape = old_shape
      out_tensor.shape = old_shape
      if not ret:
        msg = prefix_msg + check_msg
      else:
        return True, msg
  
  ret, check_msg = check_eltwise_engine(target)
  if not ret:
    return ret, msg + check_msg
  eltwise_engine = DPUTargetHelper.get_eltwise_engine(target)
  support_list = eltwise_engine.elew_type
  if node.op.type == NNDCT_OP.ADD:
    has_add = any([t == eltwise_engine.add for t in support_list])
    if not has_add:
      msg = f"{DPUTargetHelper.get_name(target)} does not support eltwise ADD."
      return False, msg
  elif node.op.type == NNDCT_OP.MULTIPLY:
    has_mul = any([t == eltwise_engine.mult for t in support_list])
    if not has_mul:
      msg += f"{DPUTargetHelper.get_name(target)} does not support eltwise MUL."
      return False, msg
  else:
    msg = f"{DPUTargetHelper.get_name(target)} does not support {node.op.type}." 
    return False, msg
  
  return True, msg

def filter_concat(node, target):
  ret = True
  msg = ""
  if any([not pn.target_device or pn.target_device.get_device_type() == DeviceType.CPU for pn in node.owning_graph.parents(node)]):
    msg += "The input of concat is not in DPU subgraph."
    dimension = node.out_tensors[0].ndim
    if dimension != 4:
      msg += "And output dimension is not 4."
      ret = False
    else:
      if node.node_attr(node.op.AttrName.AXIS) != 3:
        msg += "And it's not a channel-wise concatenation."
        ret = False

  if DPUTargetHelper.get_name(target) == "DPUCADF8H":
    dimension = node.out_tensors[0].ndim
    if dimension != 4:
      msg += "Output dimension is not 4."
      ret = False
    else:
      if node.node_attr(node.op.AttrName.AXIS) != 3:
        msg += "It's not a channel-wise concatenation."
        ret = False

  return ret, msg




def filter_upsample(node, target):
  msg = ""
  if node.node_attr(node.op.AttrName.MODE) == "BILINEAR":
    prefix_msg = "Try to convert BlinearUpsamle2d to transpose depthwise conv2d failed."
    ret, check_msg = check_bilinear_upsample_scale(node)
    if not ret:
      msg = prefix_msg + check_msg
    else:
      transpose_dwconv2d = create_transpose_dwconv2d_from_bilinear_upsample(node)
      upsample = node.op
      node.op = transpose_dwconv2d
      ret, check_msg = filter_transpose_depthwise_conv2d(node, target)
      node.op = upsample
      if not ret:
        msg = prefix_msg + check_msg
      else:
        return True, msg
      
  align_corners = node.node_attr(node.op.AttrName.ALIGN_CORNERS)
  if align_corners:
    msg = "DPU does not support align_corners = True"
    return False, msg

  mode = node.node_attr(node.op.AttrName.MODE)
  if mode == "BILINEAR":
    msg += f"DPU does not support {mode} mode.(only support NEAREST mode)."
    return ret, msg 
  load_engine = DPUTargetHelper.get_load_engine(target)
  channel_parallel = load_engine.channel_parallel
  ic = node.in_tensors[0].shape[3]
  ret, msg = check_load_jump_write(ic, channel_parallel)
  if not ret:
    return ret, msg
  return True, msg

def filter_reshape(node, target):
  msg = ""
  if DPUTargetHelper.get_type(target) == "DPUCADF8H":
    return False, msg
  
  input_node = node.owning_graph.parents(node)[0]
  if not (input_node.target_device and input_node.target_device.get_device_type() == DeviceType.DPU):
    return False, msg
  
  return True, msg
  


def filter_pad(node, target):
  msg = ""
  mode = node.node_attr(node.op.AttrName.MODE)
  if mode not in [0, 2]: #DPU only support CONSTANT / SYMMETRIC mode
    msg = f"DPU only support CONSTANT or SYMMETRIC mode."
    return False, msg
  
  load_engine = DPUTargetHelper.get_load_engine(target)
  channel_parallel = load_engine.channel_parallel
  ic = node.in_tensors[0].shape[3]
  ret, msg = check_load_jump_write(ic, channel_parallel)
  if not ret:
    return ret, msg

  return True, msg

def filter_hard_sigmoid(node, target):
  msg = ""
  if not DPUTargetHelper.has_alu_engine(target):
    msg = "This target does not support single hard-sigmoid."
    return False, msg
  return True, msg


def filter_leaky_relu(node, target):
  msg = ""
  alpha = node.node_attr(node.op.AttrName.ALPHA)
  dpu_alpha = 26.0 / 256
  if alpha != dpu_alpha:
    msg = f"Its alpa is {alpha}, but DPU only support {dpu_alpha}."
    return False, msg
  return True, msg


filters = {
  NNDCT_OP.AVG_POOL: filter_pool,
  NNDCT_OP.ADAPTIVEAVGPOOL2D: filter_pool,
  NNDCT_OP.CONVTRANSPOSE2D: filter_transpose_conv2d,
  NNDCT_OP.CONV2D: filter_conv2d,
  # NNDCT_OP.CONCAT: filter_concat,  # relax
  NNDCT_OP.DEPTHWISE_CONV2D: filter_depthwise_conv2d,
  NNDCT_OP.ADD: filter_eltwise,
  NNDCT_OP.MULTIPLY: filter_eltwise,
  NNDCT_OP.MAX_POOL: filter_pool,
  NNDCT_OP.PAD: filter_pad,
  NNDCT_OP.CONV3D: filter_conv3d,
  NNDCT_OP.DEPTHWISE_CONV3D: filter_depthwise_conv3d,

  NNDCT_OP.RESIZE: filter_upsample,
  # NNDCT_OP.RESIZE_3D: filter_upsample, # relax
  NNDCT_OP.CONVTRANSPOSE3D: filter_transpose_conv3d,
  NNDCT_OP.HSIGMOID: filter_hard_sigmoid,
  NNDCT_OP.HSWISH: filter_hard_sigmoid,
  NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D: filter_transpose_depthwise_conv2d,
  NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D: filter_transpose_depthwise_conv3d,
  NNDCT_OP.LEAKY_RELU: filter_leaky_relu
  # append here

}


def merge_permute_to_matmul(graph, target):
  def handler(*args, **kwargs):
    _, node_set = args
    permute_node = node_set[0]
    dense_node = node_set[-1]
    if permute_node.target_device and permute_node.target_device.get_device_type() == DeviceType.DPU:
      return
    if permute_node.node_attr(permute_node.op.AttrName.ORDER) == [0, 3, 1, 2] and dense_node.target_device:
      permute_node.target_device = DeviceInfo(dense_node.target_device.get_device_type())
      permute_node.target_device.clear_filter_message()

  graph_searcher = GraphSearcher(graph)
  _ = graph_searcher.find_nodes_from_type(
        [PatternType(pattern=[NNDCT_OP.PERMUTE, NNDCT_OP.FLATTEN, NNDCT_OP.DENSE],
                     action=handler), 
         PatternType(pattern=[NNDCT_OP.PERMUTE, NNDCT_OP.RESHAPE, NNDCT_OP.DENSE], 
                     action=handler),
        ])
  
def filter_dpu_interface_concat(graph, target):
  def handler(*args, **kwargs):
    _, node_set = args
    concat = node_set[0]
    if concat.target_device and concat.target_device.get_device_type() == DeviceType.DPU:
      return
    ret, msg = filter_concat(concat, target)
    if ret:
      concat.target_device = DeviceInfo(DeviceType.DPU)
      concat.target_device.clear_filter_message()
    else:
      concat.target_device = DeviceInfo(DeviceType.CPU)
      concat.target_device.set_filter_message(msg)

  graph_searcher = GraphSearcher(graph)
  _ = graph_searcher.find_nodes_from_type(
        [PatternType(pattern=[NNDCT_OP.CONCAT],
                     action=handler)])

def filter_dpu_interface_reshape(graph, target):
  def handler(*args, **kwargs):
    _, node_set = args
    reshape = node_set[0]
    if not (reshape.target_device and reshape.target_device.get_device_type() == DeviceType.DPU):
      return
    input_node = reshape.owning_graph.parents(reshape)[0]
    if input_node.target_device and input_node.target_device.get_device_type() == DeviceType.DPU \
    and all([cn.target_device and cn.target_device.get_device_type() == DeviceType.DPU for cn in reshape.owning_graph.children(reshape)]):
      # an internal reshape, do nothing
      pass
    else:
      if input_node.out_tensors[0].shape[0] != reshape.out_tensors[0].shape[0]:
        msg = "First dimension is changed."
        reshape.target_device = DeviceInfo(DeviceType.CPU)
        reshape.target_device.set_filter_message(msg)

  graph_searcher = GraphSearcher(graph)
  _ = graph_searcher.find_nodes_from_type(
        [PatternType(pattern=[NNDCT_OP.RESHAPE],
                     action=handler),
        PatternType(pattern=[NNDCT_OP.FLATTEN],
                     action=handler),
                     ])
  
def check_dim_of_inputs_of_mul(node):
  msg = ""
  if_replaceabel = True
  input_shape = node.in_tensors[0].shape
  const_node = node.in_tensors[1].node
  const_data = const_node.node_attr(const_node.op.AttrName.DATA)
  if not isinstance(const_data, list):
    const_data = [const_data]
  const_data = np.array(const_data)
  const_shape = const_data.shape

  if_replaceabel = (len(input_shape) <= 4 and 
                    len(const_shape) == 1 and
                    (input_shape[-1] == const_shape[0] or
                    const_shape[0] == 1))
  
  if not if_replaceabel:
    msg = f"mul's input  has the tensor dimension {input_shape} and weights has the tensor dimenstion {const_shape}."
    return False, msg
  return True, msg

def create_dwconv2d_from_mul(node):
  dwconv2d = TorchConv2d(NNDCT_OP.DEPTHWISE_CONV2D)
  dwconv2d.set_config('kernel_size', [1, 1])
  dwconv2d.set_config('stride', [1, 1])
  dwconv2d.set_config('padding', [0, 0])
  input_channel = node.in_tensors[0].shape[-1]
  weight_tensor = Tensor("weight")
  weight_tensor.from_ndarray(np.random.randn(1, 1, 1, input_channel))
  dwconv2d.set_param(dwconv2d.ParamName.WEIGHTS, weight_tensor)

  return dwconv2d

pattern_filters = [merge_permute_to_matmul, filter_dpu_interface_reshape]




