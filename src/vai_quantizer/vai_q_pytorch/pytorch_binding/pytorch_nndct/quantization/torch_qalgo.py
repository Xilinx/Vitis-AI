

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

import sys
import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import entropy
from scipy import stats
import torch
from collections import Counter
import pytorch_nndct as py_nndct
from pytorch_nndct.nn.modules.fix_ops import diffs_fix_pos
from nndct_shared.utils import NndctOption, NndctScreenLogger, QError, QWarning
from nndct_shared.base import NNDCT_OP
from pytorch_nndct.nn import fake_quantize_per_tensor, fake_quantize_per_channel
from pytorch_nndct.nn import fake_quantize_per_tensor_tensorrt, fake_quantize_per_channel_tensorrt

_CONV_LINEAR_TYPES = [NNDCT_OP.CONV2D, NNDCT_OP.DEPTHWISE_CONV2D, \
                      NNDCT_OP.CONV3D, NNDCT_OP.DEPTHWISE_CONV3D, \
                      NNDCT_OP.DENSE, NNDCT_OP.LINEAR]

_CONV_TRANSPOSE_TYPES = [NNDCT_OP.CONVTRANSPOSE2D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D,\
                        NNDCT_OP.CONVTRANSPOSE3D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D]

def create_quant_algo(tensor_type, quant_strategy_info, node):
  algo_config = quant_strategy_info
  quant_algo = None
  scale_type = algo_config.get("scale_type")
  granularity = algo_config.get("granularity")
  
  if granularity == "per_channel":
    if (int(torch.__version__.split('.')[1]) < 5) and (int(torch.__version__.split('.')[0]) <= 1):
      NndctScreenLogger().error2user(QError.TORCH_VERSION, f"Torch should uptate to 1.5.0 or higher version if per_channel quantization.")
      exit(2)
  op_type = node.op.type
  if scale_type == "float":
    if granularity == "per_channel":
      axis = None
      #group = node.node_attr[node.op.AttrName.GROUP]
      if op_type in _CONV_LINEAR_TYPES:
        axis = 0
      elif op_type in _CONV_TRANSPOSE_TYPES:
        axis = 1
      quant_algo = FloatQuantPerChannelAlgo(algo_config, axis)
    elif granularity == "per_tensor":
      method = algo_config.get("method")
      if method == "maxmin":
        quant_algo = MaxMinQuantPerTensorAlgo(algo_config)
      elif method == "percentile":
        quant_algo = PercentileQuantPerTensorAlgo(algo_config)
      elif method == "mse":
        quant_algo = MSEQuantPerTensorAlgo(algo_config)
      elif method == "entropy":
        quant_algo = EntropyQuantPerTensorAlgo(algo_config)
  elif scale_type == "poweroftwo":
    if granularity == "per_tensor":
      quant_algo = PowerofTwoQuantPerTensorAlgo(algo_config, op_type)
    elif granularity == "per_channel":
      axis = None
      #group = node.node_attr[node.op.AttrName.GROUP]
      if op_type in _CONV_LINEAR_TYPES:
        axis = 0
      elif op_type in _CONV_TRANSPOSE_TYPES:
        axis = 1
      quant_algo = PowerofTwoQuantPerChannelAlgo(algo_config, axis, op_type)
  return quant_algo

class UniformQuantAlgo(ABC):
  def __init__(self, config):
    self._config = config
    self._quant_max = config.get("quant_max")
    self._quant_min = config.get("quant_min")
    self._round_method = config.get("round_method")
    self._symmetric_mode = config.get("symmetric_mode")
    self._signed = config.get("signed", True)
    self._range_stat_method = config.get("calib_statistic_method", None)
    self._bitwidth =  config.get("bit_width")
    self._narrow_range = config.get("narrow_range")
    self._method = config.get("method")
    self._scale = None
    self._zero_point = None
    self._float_max = None
    self._float_min = None
    self._calib_cnt = 0
    self._statistic_local = NndctOption.nndct_calibration_local.value
  
  @abstractmethod
  def fake_quantize(self, input, inplace):
    pass
  
  @abstractmethod
  def calib_global_statis(self):
    pass
  
  @abstractmethod
  def act_scale_stats(self, param_array):
        pass

  @abstractmethod
  def calibrate(self, *args, **kwargs):
    pass
  
  def calib_or_not(self, tensor_type):
    if tensor_type != 'param':
      return True
    else:
      if self._statistic_local:
        if self._scale is None or self._zero_point is None:
          return True
        else:
          return False
      else:
        if self._calib_cnt > 0:
          return False
        else:
          return True

  @property
  def bitwidth(self):
    return self._bitwidth

  @property
  def scale(self):
    return self._scale
  
  @property
  def zero_point(self):
    return self._zero_point 
  
  @property
  def float_max(self):
    return self._float_max
  
  @property
  def float_min(self):
    return self._float_min

  @property
  def quant_max(self):
    return self._quant_max
  
  @property
  def quant_min(self):
    return self._quant_min
  
  @property
  def statistic_local(self):
    return self._statistic_local
  
  @bitwidth.setter
  def bitwidth(self, value):
    self._bitwidth = value
  
  @scale.setter
  def scale(self, value):
    self._scale = value
  
  @zero_point.setter
  def zero_point(self, value):
    self._zero_point = value

  @float_max.setter
  def float_max(self, value):
    self._float_max = value
  
  @float_min.setter
  def float_min(self, value):
    self._float_min = value
    
  @statistic_local.setter
  def statistic_local(self, value):
    self._statistic_local = value

class PerChannelQuantAlgo(UniformQuantAlgo):
  def __init__(self, config, axis):
      super().__init__(config)
      self._axis = axis
      #self._groups = groups
  
  def fake_quantize(self, input, inplace):
    if self._round_method == "half_even":
      if NndctOption.nndct_tensorrt_quant_algo.value and self._symmetric_mode == "symmetric":
        return fake_quantize_per_channel_tensorrt(input, self._float_max, self._quant_min, self._quant_max, self._axis)
      else:
        if (int(torch.__version__.split('.')[1]) > 9) and (int(torch.__version__.split('.')[0]) > 0):
          self._zero_point = self._zero_point.to(torch.int32)
        else:
          self._zero_point = self._zero_point.to(torch.long)
        return torch.fake_quantize_per_channel_affine(input, self._scale, self._zero_point, 
                                                      self._axis, self._quant_min, self._quant_max)
              
    else:
      if self._round_method == "half_up":
        method = 2
      elif self._round_method == "half_down":
        method = 6
      elif self._round_method == "std_round":
        method = 3
      return fake_quantize_per_channel(input, 1.0/self._scale, self._zero_point,
                                       self._axis, self._quant_min, self._quant_max, method, inplace)

  @abstractmethod
  def calibrate(self, *args, **kwargs):
    pass

  def act_scale_stats(self, param_array):
    pass
    
  @abstractmethod
  def calib_global_statis(self):
    pass
  
  @abstractmethod
  def calibrate(self, *args, **kwargs):
    pass
    
  @property
  def scale(self):
    return self._scale.cpu().detach().numpy().tolist()
  
  @property
  def zero_point(self):
    return self._zero_point.cpu().detach().numpy().tolist()
  
  @property
  def float_max(self):
    return self._float_max.cpu().detach().numpy().tolist()
  
  @property
  def float_min(self):
    return self._float_min.cpu().detach().numpy().tolist()
  
  @scale.setter
  def scale(self, value):
    self._scale = value
  
  @zero_point.setter
  def zero_point(self, value):
    self._zero_point = value
    
  @float_max.setter
  def float_max(self, value):
    self._float_max = value
  
  @float_min.setter
  def float_min(self, value):
    self._float_min = value

class FloatQuantPerChannelAlgo(PerChannelQuantAlgo):
  def __init__(self, config, axis):
      super().__init__(config, axis)
  
  def calibrate(self, tensor):
    with torch.no_grad():
      #tensor = tensor.abs()
      axis = self._axis if isinstance(self._axis, (list, tuple)) else [self._axis]
      reduce_axis = []
      for i in range(tensor.dim()):
        if not i in axis:
          reduce_axis.append(i)
      #local_max = self.reduce_amax(tensor,axis=reduce_axis).detach()
      local_max, local_min = self.reduce_min_max(tensor,axis=reduce_axis)
      local_max = torch.squeeze(local_max)
      local_min = torch.squeeze(local_min)
      if self._symmetric_mode == "symmetric": 
        local_max, local_min = self._get_float_max_min(local_max, local_min)
      
      if self._float_max is None or self._statistic_local:
        self._float_max = local_max
      else:
        if local_max.shape != self._float_max.shape:
          raise RuntimeError("max shape changed!")
        self._float_max.copy_(torch.max(self._float_max, local_max).data)
      
      if self._float_min is None or self._statistic_local:
        self._float_min = local_min
      else:
        if local_min.shape != self._float_min.shape:
          raise RuntimeError("mix shape changed!")
        self._float_min.copy_(torch.min(self._float_min, local_min).data)
      
      if self._statistic_local:
        self.calib_scale()
  
  def calib_scale(self):
    if self._symmetric_mode == "symmetric":
      self._scale = self._float_max/self._quant_max
      self._zero_point = torch.zeros(self._scale.shape[0], device=self._scale.device).long()
    else:
      self._scale = (self._float_max-self._float_min)/(self._quant_max-self._quant_min)
      self._zero_point = (-self._float_min/self._scale).round().long()
  
  def calib_global_statis(self):
    self.calib_scale()
  
  @staticmethod
  def reduce_min_max(input, axis=None, keepdims=True):
    with torch.no_grad():
      o_max = input.detach().clone()
      o_min = input.detach().clone()
      if axis is None:
        o_max = torch.max(o_max)
        o_min = torch.min(o_min)
      else:
        if isinstance(axis, int):
          o_max, _ = torch.max(o_max, dim=axis, keepdim=keepdims)
          o_min, _ = torch.min(o_min, dim=axis, keepdim=keepdims)
        else:
          if isinstance(axis, tuple) and len(axis) > input.dim():
            raise ValueError("Cannot reduce more axes than tensor's dim.")
          for i in axis:
            o_max, _ = torch.max(o_max, dim=i, keepdim=True)
            o_min, _ = torch.min(o_min, dim=i, keepdim=True)
          if not keepdims or o_max.numel() == 1:
            o_max.squeeze_()
            o_min.squeeze_()
      return o_max, o_min
  
  def _get_float_max_min(self, float_max, float_min):
    if self._signed:
      float_max = float_max.abs()
      float_min = float_min.abs()
      float_max_min = torch.stack((float_max, float_min), dim=0)
      stack_max, _ = torch.max(float_max_min, dim=0)
      stack_min = -stack_max
      return stack_max, stack_min
    else:
      amin = float_min.min()
      if amin >= 0:
        return 0, float_max
      else:
        self._quant_max = int(2 ** (self._bitwidth - 1)) - 1
        if not self._narrow_range:
          self._quant_min = -int(2 ** (self._bitwidth - 1))
        else:
          self._quant_min = -int(2 ** (self._bitwidth - 1)) + 1
      float_max = float_max.abs()
      float_min = float_min.abs()
      float_max_min = torch.stack((float_max, float_min), dim=0)
      stack_max, _ = torch.max(float_max_min, dim=0)
      stack_min = -stack_max
      self._signed = True
      return stack_max, stack_min
    
class PowerofTwoQuantPerChannelAlgo(PerChannelQuantAlgo):
  def __init__(self, config, axis, node_type):
    super().__init__(config, axis)
    self._fix_pos = None
    self._node_type = node_type
    self._statistic_local = True
  
  def calibrate(self, tensor):
    scope = 5
    with torch.no_grad():
      if self._method == "diffs":
          scope = 5
      elif self._method == "maxmin":
        scope = 1
        
      if (self._node_type in [NNDCT_OP.INPUT, NNDCT_OP.QUANT_STUB]):
        scope = 1
        
      if self._round_method == "half_up":
        mth = 2
      elif self._round_method == "half_down":
        mth = 6
      elif self._round_method == "std_round":
        mth = 3
      elif self._round_method == "half_even":
        mth = -1
      
      device = tensor.device
      out_num = tensor.shape[self._axis]
      Tbuffer = torch.empty_like(tensor).to(device)
      self._fix_pos = torch.ones(out_num, dtype=tensor.dtype).to(device)
        
      self._calib_cnt = self._calib_cnt + 1
      
      # py_nndct.nn.NndctDiffsFixPosChannel(
      #       Tinput = tensor,
      #       Tbuffer = Tbuffer,
      #       Tfixpos = self._fix_pos,
      #       axis = self._axis,
      #       bit_width = self.bitwidth,
      #       scope = scope,
      #       method = mth)
      input_split = torch.split(tensor, 1, dim=self._axis)
      buffer_split = torch.split(Tbuffer, 1, dim=self._axis)
      # TODO(@kewang): The split is a tensor view operation. Is it neccessary to clone tensor before calib and test ? 
      for i in range(len(input_split)):
        self._fix_pos[i] = diffs_fix_pos(
                              input=input_split[i], 
                              bit_width=self.bitwidth, 
                              scope=scope, 
                              method=mth)
      if self._bitwidth <= 8:
        max_fp = NndctOption.nndct_max_fix_position.value
        #self._fix_pos = min(max_fp, self._fix_pos)
        self._fix_pos = torch.where(self._fix_pos.long()>max_fp, max_fp, self._fix_pos.long())
      else:
        #self._fix_pos = min(15, self._fix_pos)
        self._fix_pos = torch.where(self._fix_pos.long()>15, 15, self._fix_pos.long())
      self.calib_scale()
      # if self._statistic_local:
      #   self.calib_scale()
        
  def calib_scale(self):
    try:
      #self._scale = 1.0/2**self._fix_pos if self._fix_pos > 0 else 2**(-self._fix_pos)
      self._scale = torch.where(self._fix_pos.to(torch.float32)>0, 1.0/2**self._fix_pos.to(torch.float32), 2**(-self._fix_pos.to(torch.float32)))
      #self._scale = torch.where(self._fix_pos>0, 1.0/2**self._fix_pos, 2**(-self._fix_pos))
    except OverflowError as e:
      print("{}".format(repr(e)))
    self._zero_point = torch.zeros(self._scale.shape[0], device=self._scale.device).long()
    self._float_max = self._scale*self._quant_max
  
  def calib_global_statis(self):
    #self.calib_scale()
    pass

class PerTensorQuantAlgo(UniformQuantAlgo):
  def __init__(self, config):
    super().__init__(config)
  
  def fake_quantize(self, input, inplace):
    if self._round_method == "half_even":
      if NndctOption.nndct_tensorrt_quant_algo.value and self._symmetric_mode == "symmetric":
        return fake_quantize_per_tensor_tensorrt(input, self._float_max, self._quant_min, self._quant_max)
      else:
        return torch.fake_quantize_per_tensor_affine(input, self._scale, self._zero_point, 
                                                    self._quant_min, self._quant_max)
    else:
      if self._round_method == "half_up":
        method = 2
      elif self._round_method == "half_down":
        method = 6
      elif self._round_method == "std_round":
        method = 3
      return fake_quantize_per_tensor(input, 1.0/self._scale, self._zero_point,
                                      self._quant_min, self._quant_max, method, inplace)
  
  def _get_float_max_min(self, float_max, float_min):
    if self._signed:
      float_max = float_max.abs()
      float_min = float_min.abs()
      stack_max = torch.max(float_max, float_min).item()
      stack_min = -stack_max
      return stack_max, stack_min
    else:
      amin = float_min.min()
      if amin >= 0:
        return float_max.item(), 0
      else:
        raise TypeError("Negative values encountered in unsigned quantization.")
  
  def act_scale_stats(self, param_array):
    scale = None
    zero_point = None
    float_max = None
    if self._range_stat_method == 'max':
      scale = param_array[0].max()
    elif self._range_stat_method == 'mean':
      scale = param_array[0].mean()
    elif self._range_stat_method == 'median':
      scale = np.median(param_array[0])
    elif self._range_stat_method == 'modal':
      scale = stats.mode(param_array[0])[0][0]
    self._scale = scale

    zero_point = stats.mode(param_array[1])[0][0]
    zero_point = zero_point.astype(np.int32).tolist()
    self._zero_point = zero_point
    
    if self._range_stat_method == 'max':
      float_max = param_array[2].max()
    elif self._range_stat_method == 'mean':
      float_max = param_array[2].mean()
    elif self._range_stat_method == 'median':
      float_max = np.median(param_array[2])
    elif self._range_stat_method == 'modal':
      float_max = stats.mode(param_array[2])[0][0]
    self._float_max = float_max
    
    return scale, zero_point, float_max
  
  @abstractmethod
  def calib_global_statis(self):
    pass
  
  @abstractmethod
  def calibrate(self, *args, **kwargs):
    pass
  
class MaxMinQuantPerTensorAlgo(PerTensorQuantAlgo):
  def __init__(self, config):
    super().__init__(config)

  def calibrate(self, tensor):
    self._calib_cnt = self._calib_cnt + 1
    tensor_max = torch.max(tensor)
    tensor_min = torch.min(tensor)
    if self._symmetric_mode =="symmetric":
      new_max, new_min = self._get_float_max_min(tensor_max, tensor_min)
    else:
      new_max = tensor_max.item()
      new_min = tensor_min.item()
    if self._float_max is None or self._statistic_local:
      self._float_max = new_max
    else:
      self._float_max = new_max if new_max > self._float_max else self._float_max
    if self._float_min is None or self._statistic_local:
      self._float_min = new_min
    else:
      self._float_min = new_min if new_min < self._float_min else self._float_min
    if self._statistic_local:
      self.calib_scale()
  
  def calib_scale(self):
    if self._symmetric_mode == "symmetric" :
      self._scale = self._float_max/self._quant_max
      self._zero_point = 0
    else:
      self._scale = (self._float_max-self._float_min)/(self._quant_max-self._quant_min)
      self._zero_point = round(-self._float_min/self._scale)
  
  def calib_global_statis(self):
    #self._scale = (self._float_max-self._float_min)/(self._quant_max-self._quant_min)
    self.calib_scale()

class HistogramQuantPerTensorAlgo(PerTensorQuantAlgo):
  def __init__(self, config):
      super().__init__(config)
      self._calib_hist = None
      self._calib_bin_edges = None
      self._num_bins = NndctOption.nndct_calib_histogram_bins.value

  def calibrate(self, tensor):
    self._calib_cnt = self._calib_cnt + 1
    device = tensor.device
    self._get_histogram(tensor, self._symmetric_mode)
    if self._statistic_local:
      calib_hist = self._calib_hist.long().cpu().numpy()
      calib_bin_edges = self._calib_bin_edges.cpu().numpy()
      self.calib_scale(calib_hist, calib_bin_edges)
  
  def _get_histogram(self, tensor, symmetric_mode):
    if torch.min(tensor) < 0:
      tensor = tensor.abs()
    tensor = tensor.float()
    with torch.no_grad():
      tensor_max, _ = self._get_float_max_min(torch.max(tensor), torch.min(tensor))
      tensor_min = 0.0
      
      if (self._calib_bin_edges is None and self._calib_hist is None) or self._statistic_local:
        self._calib_hist = torch.histc(tensor, bins=self._num_bins, min=tensor_min, max=tensor_max)
        self._calib_bin_edges = torch.linspace(tensor_min, tensor_max, self._num_bins+1, device=tensor.device)
      else:
        if tensor_max > self._calib_bin_edges[-1]:
          width = self._calib_bin_edges[1] - self._calib_bin_edges[0]
          self._num_bins = int((tensor_max/width).ceil().item())
          self._calib_bin_edges = torch.arange(0, tensor_max+width, width, device=tensor.device)
        
        hist = torch.histc(tensor, bins=self._num_bins, min=0, max=self._calib_bin_edges[-1])
        hist[:self._calib_hist.numel()] += self._calib_hist
        self._calib_hist = hist
    
    return self._calib_hist, self._calib_bin_edges
  
  def calib_global_statis(self):
    calib_hist = self._calib_hist.long().cpu().numpy()
    calib_bin_edges = self._calib_bin_edges.cpu().numpy()
    self.calib_scale(calib_hist, calib_bin_edges)

  @abstractmethod
  def calib_scale(self, calib_hist, calib_bin_edges):
    pass
  
class PercentileQuantPerTensorAlgo(HistogramQuantPerTensorAlgo):
  def __init__(self, config):
    super().__init__(config)

  def calib_scale(self, calib_hist, calib_bin_edges):
    percentage = self._config["percentage"]
    if percentage < 0 or percentage > 100:
      raise ValueError("Invalid percentage, Must be in range 0 <= percentage <= 100")

    total = calib_hist.sum()
    cdf = np.cumsum(calib_hist / total)
    idx = np.searchsorted(cdf, percentage / 100)
    calib_amax = calib_bin_edges[idx]
    self._scale = calib_amax/self._quant_max
    self._zero_point = 0
    self._float_max = calib_amax

    # return calib_amax
    
    # total = calib_hist.sum()
    # cdf = torch.cumsum(calib_hist/total, dim=0)
    # idx = torch.searchsorted(cdf, percentage/100)
    # calib_max = calib_bin_edges[idx].item()
    # #calib_min = -calib_max
    # #self._scale = (calib_max-calib_min)/(self._quant_max-self._quant_min)
    # self._scale = calib_max/self._quant_max
    # self._zero_point = 0
    # self._float_max = calib_max

class MSEQuantPerTensorAlgo(HistogramQuantPerTensorAlgo):
  def __init__(self, config):
    super().__init__(config)

  def calib_scale(self, calib_hist, calib_bin_edges):
    start_bin = NndctOption.nndct_mse_start_bin.value
    stride = NndctOption.nndct_mse_stride.value  
    
    counts = torch.from_numpy(calib_hist[:]).float()
    edges = torch.from_numpy(calib_bin_edges[:]).float()
    centers = (edges[1:] + edges[:-1]) / 2
    # counts = calib_hist.float()
    # edges = calib_bin_edges.float()
    # centers = (edges[1:] + edges[:-1]) / 2

    mses = []
    arguments = []
    for i in range(start_bin, len(centers), stride):
      amax = centers[i]
      #amin = -amax
      #scale = (amax-amin)/(self._quant_max-self._quant_min)
      scale = amax/self._quant_max
      zero_point = 0
      if self._round_method == "half_even":
        # if NndctOption.nndct_tensorrt_quant_algo.value:
        #   quant_centers = fake_quantize_per_tensor_tensorrt(centers, amax,
        #                                                     self._quant_min, self._quant_max)
        # else:
        #   quant_centers = torch.fake_quantize_per_tensor_affine(centers, scale, zero_point, 
        #                                                         self._quant_min, self._quant_max)
        quant_centers = fake_quantize_per_tensor_tensorrt(centers, amax,
                                                          self._quant_min, self._quant_max)
      else:
        if self._round_method == "half_up":
          method = 2
        elif self._round_method == "half_down":
          method = 6
        elif self._round_method == "std_round":
          method = 3
        quant_centers = fake_quantize_per_tensor(centers, 1.0/scale, zero_point,
                                                 self._quant_min, self._quant_max, method, True)
      mse = ((quant_centers - centers)**2 * counts).mean().cpu().detach().item()
      mses.append(mse)
      arguments.append(i)
      
    argmin = np.argmin(mses)
    calib_max = centers[arguments[argmin]].cpu().detach().item()
    #calib_min = -calib_max
    #self._scale = (calib_max-calib_min)/(self._quant_max-self._quant_min)
    self._scale = calib_max/self._quant_max
    self._zero_point = 0
    self._float_max = calib_max
      
class EntropyQuantPerTensorAlgo(HistogramQuantPerTensorAlgo):
  def __init__(self, config):
    super().__init__(config)
    
  def calib_scale(self, calib_hist, calib_bin_edges):
    start_bin = NndctOption.nndct_entropy_start_bin.value
    stride = NndctOption.nndct_entropy_stride.value    
    
    #calib_hist = calib_hist.int().cpu().numpy()
    # calib_hist = calib_hist.long().cpu().numpy()
    # calib_bin_edges = calib_bin_edges.cpu().numpy()
    def _normalize_distr(distr):
      summ = np.sum(distr)
      if summ != 0:
        distr = distr / summ

    bins = calib_hist[:]
    bins[0] = bins[1]

    total_data = np.sum(bins)

    divergences = []
    arguments = []
    unsigned_bit = 0 if self._signed else 1
    nbins = 1 << (self.bitwidth - 1 + unsigned_bit)
    starting = start_bin
    stop = len(bins)

    new_density_counts = np.zeros(nbins, dtype=np.float64)

    for i in range(starting, stop + 1, stride):
      new_density_counts.fill(0)
      space = np.linspace(0, i, num=nbins + 1)
      digitized_space = np.digitize(range(i), space) - 1

      digitized_space[bins[:i] == 0] = -1

      for idx, digitized in enumerate(digitized_space):
        if digitized != -1:
          new_density_counts[digitized] += bins[idx]

      counter = Counter(digitized_space)
      for key, val in counter.items():
        if key != -1:
          new_density_counts[key] = new_density_counts[key] / val

      new_density = np.zeros(i, dtype=np.float64)
      for idx, digitized in enumerate(digitized_space):
        if digitized != -1:
          new_density[idx] = new_density_counts[digitized]

      total_counts_new = np.sum(new_density) + np.sum(bins[i:])
      _normalize_distr(new_density)
      
      reference_density = np.array(bins[:len(digitized_space)], dtype=np.int64)
      #reference_density = np.array(bins[:len(digitized_space)])
      reference_density[-1] += np.sum(bins[i:])

      total_counts_old = np.sum(reference_density)
      if round(total_counts_new) != total_data or round(total_counts_old) != total_data:
        raise RuntimeError("Count mismatch! total_counts_new={}, total_counts_old={}, total_data={}".format(
          total_counts_new, total_counts_old, total_data))
        
      _normalize_distr(reference_density)
            
      ent = entropy(reference_density, new_density)
      divergences.append(ent)
      arguments.append(i)

    divergences = np.array(divergences)
    last_argmin = len(divergences) - 1 - np.argmin(divergences[::-1])
    calib_max = calib_bin_edges[last_argmin * stride + starting]
    #calib_min = -calib_max
    #self._scale = (calib_max-calib_min)/(self._quant_max-self._quant_min)
    self._scale = calib_max/self._quant_max
    self._zero_point = 0
    self._float_max = calib_max

class PowerofTwoQuantPerTensorAlgo(PerTensorQuantAlgo):
  def __init__(self, config, node_type):
    super().__init__(config)
    self._fix_pos = None
    self._node_type = node_type
    self._statistic_local = True

  def calibrate(self, tensor):
    scope = 5
    if self._method == "diffs":
      scope = 5
    elif self._method == "maxmin":
      scope = 1
      
    if (self._node_type in [NNDCT_OP.INPUT, NNDCT_OP.QUANT_STUB]):
      scope = 1
      
    if self._round_method == "half_up":
      mth = 2
    elif self._round_method == "half_down":
      mth = 6
    elif self._round_method == "std_round":
      mth = 3
    elif self._round_method == "half_even":
      mth = -1
    
    device = tensor.device
    Tbuffer = torch.empty_like(tensor).to(device)
    Tfixpos = torch.tensor([1], dtype=tensor.dtype).to(device)
      
    self._calib_cnt = self._calib_cnt + 1
    
    # py_nndct.nn.NndctDiffsFixPos(
    #       Tinput = tensor,
    #       Tbuffer = Tbuffer,
    #       Tfixpos = Tfixpos,
    #       bit_width = self.bitwidth,
    #       range = scope,
    #       method = mth)
    Tfixpos = diffs_fix_pos(
            input=tensor,
            bit_width=self.bitwidth,
            scope=scope,
            method=mth)
    
    self._fix_pos = (int)(Tfixpos.item())
    if self._bitwidth <= 8:
      max_fp = NndctOption.nndct_max_fix_position.value
      self._fix_pos = min(max_fp, self._fix_pos)
    else:
      self._fix_pos = min(15, self._fix_pos)
    self.calib_scale()
    # if self._statistic_local:
    #   self.calib_scale()
  
  def calib_scale(self):
    try:
      self._scale = 1.0/2**self._fix_pos if self._fix_pos > 0 else 2**(-self._fix_pos)
    except OverflowError as e:
      print("{}".format(repr(e)))
    self._zero_point = 0
    self._float_max = self._scale*self._quant_max
    
  def calib_global_statis(self):
    #self._scale = (self._float_max-self._float_min)/(self._quant_max-self._quant_min)
    #self.calib_scale()
    pass
