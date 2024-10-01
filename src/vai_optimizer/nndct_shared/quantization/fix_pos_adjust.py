
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

from abc import ABC, abstractmethod
import threading
from nndct_shared.base import NNDCT_OP
from nndct_shared.utils import NndctScreenLogger, QWarning, NndctOption
from nndct_shared.utils import NNDCT_KEYS, GLOBAL_MAP

class FixPosChecker(ABC):
 
  def __init__(self, ctx):
    self.ctx = ctx
    # aligning with output OPs
    
    self.adjust_relative_and_absolute_fn_map = {
      # hardsigmoid: output_fp >=7 and input_fp > 0 
      NNDCT_OP.HSIGMOID: [self.output_must_larger_than_7, self.input_must_larger_than_0],
      # hardswish: input_fp > 0 and input_fp + 7 >= output_fp
      NNDCT_OP.HSWISH: [self.input_must_larger_than_0, self.input_output_relative_check_for_hswish],  
      # shift_cut/shift_bias for convlike
      NNDCT_OP.HSWISH: [self.input_must_larger_than_0, self.input_output_relative_check_for_hswish],  
      NNDCT_OP.CONV2D: [self.shift_cut_bias], 
      NNDCT_OP.CONV3D: [self.shift_cut_bias],
      NNDCT_OP.CONV1D: [self.shift_cut_bias],
      NNDCT_OP.CONVTRANSPOSE2D: [self.shift_cut_bias],
      NNDCT_OP.CONVTRANSPOSE3D: [self.shift_cut_bias],
      NNDCT_OP.DEPTHWISE_CONV2D: [self.shift_cut_bias],
      NNDCT_OP.DEPTHWISE_CONV3D: [self.shift_cut_bias],
      NNDCT_OP.DENSE: [self.shift_cut_bias],
      NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D: [self.shift_cut_bias],
      NNDCT_OP.LAYER_NORM: [self.shift_normlization_weight_bias],
    }
  
  @abstractmethod
  def __call__(self, graph):
    pass

  @abstractmethod
  def get_quant_input_config(self, node, offset):
    pass

  @abstractmethod
  def get_quant_output_config(self, node):
    pass
  
  @abstractmethod
  def get_quant_param_config(self, name):
    pass  
  
  @abstractmethod
  def set_quant_output_config(self, node):
    pass
  
  @abstractmethod
  def set_quant_param_config(self, name):
    pass

  @abstractmethod
  def get_quant_output_name(self, node):
    pass
  
  def output_must_larger_than_7(self, node, filter_fn):
    if filter_fn(self.ctx, node):
      bnfp = self.get_quant_output_config(node)
      if bnfp[1] < 7:
        bnfp[1] = 7
      self.set_quant_output_config(node, bnfp)
  
  def input_must_larger_than_0(self, node, filter_fn):
    if filter_fn(self.ctx, node):
      graph = node.owning_graph
      for idx, in_node in enumerate(node.in_nodes):
        bnfp = self.get_quant_input_config(node, idx)
        if bnfp[1] < 0:
          bnfp[1] = 0
        self.set_quant_output_config(graph.node(in_node), bnfp)

  def input_output_relative_check_for_hswish(self, node, filter_fn):
    if filter_fn(self.ctx, node):
      bnfp_o = self.get_quant_output_config(node)
      bnfp_i = self.get_quant_input_config(node, 0)
      #print(f'Before change pool fix of {node.name} {node.op.type} : {self.get_quant_config(node, False)}')
      if bnfp_i[1] + 7 < bnfp_o[1]:
        bnfp_o[1] = bnfp_i[1] + 7
      self.set_quant_output_config(node, bnfp_o)

  def shift_cut_bias(self, node, filter_fn):
    if filter_fn(self.ctx, node):
      fix_pos_i = self.get_quant_input_config(node, 0)
      fix_pos_o = self.get_quant_output_config(node)
      fix_pos_w = self.get_quant_param_config(node.op.param['weights'].name)
      if fix_pos_i[-1] is None:
        NndctScreenLogger().warning2user(QWarning.TENSOR_NOT_QUANTIZED, f"Input tensor of node {node.name} is not quantized.")
        return
      if fix_pos_o[-1] is None:
        NndctScreenLogger().warning2user(QWarning.TENSOR_NOT_QUANTIZED, f"Output tensor of node {node.name} is not quantized.")
        return
      if fix_pos_w[-1] is None:
        NndctScreenLogger().warning2user(QWarning.TENSOR_NOT_QUANTIZED, f"Weights tensor {node.op.param['weights'].name} is not quantized.")
        return
      # handle shift_cut
      shift_cut = fix_pos_w[-1] + fix_pos_i[-1] - fix_pos_o[-1]
      shift_cut_min = 0
      # Flor FLEXML device, use shift_cut_min specified in config file
      config = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_CONFIG)
      if config and config['target_device'] == "FLEXML":
        shift_cut_min = -1
      shift_cut_max = 16
      if shift_cut < shift_cut_min:
        NndctScreenLogger().warning2user(QWarning.SHIFT_CHECK, "output {} value is too small, so adjust the fix position from {} to {}"
                .format(self.get_quant_output_name(node), fix_pos_o[-1], fix_pos_o[-1] + shift_cut - shift_cut_min))
        fix_pos_o[-1] = fix_pos_o[-1] + shift_cut - shift_cut_min
        self.set_quant_output_config(node, fix_pos_o)
      elif shift_cut > shift_cut_max:
        NndctScreenLogger().warning2user(QWarning.SHIFT_CHECK, "weight {} value is too small, so adjust the fix position from {} to {}"
                .format(node.op.param['weights'].name, fix_pos_w[-1], fix_pos_w[-1] - shift_cut + shift_cut_max))
        fix_pos_w[-1] = fix_pos_w[-1] - shift_cut + shift_cut_max
        self.set_quant_param_config(node.op.param['weights'].name, fix_pos_w)

      # handle shift_bias
      if node.op.ParamName.BIAS in node.op.params:
        fix_pos_b = self.get_quant_param_config(node.op.param['bias'].name)
        shift_bias = fix_pos_w[-1] + fix_pos_i[-1] - fix_pos_b[-1]
        shift_bias_min = min(0, -(24 - (8 + shift_cut)))
	# For FLEXML device use shift_bias specified in config file
        if config and config['target_device'] == "FLEXML":
          shift_bias_min = 0
        shift_bias_max = 16
        if shift_bias < shift_bias_min:
          NndctScreenLogger().warning2user(QWarning.SHIFT_CHECK, "bias {} value is too small, so adjust the fix position from {} to {}"
                  .format(node.op.param['bias'].name, fix_pos_b[-1], fix_pos_b[-1] + shift_bias - shift_bias_min))
          fix_pos_b[-1] = fix_pos_b[-1] + shift_bias - shift_bias_min;
          self.set_quant_param_config(node.op.param['bias'].name, fix_pos_b)
        elif shift_bias > shift_bias_max:
          NndctScreenLogger().warning2user(QWarning.SHIFT_CHECK, "weight {} value is too small, so adjust the fix position from {} to {}"
                  .format(node.op.param['weights'].name, fix_pos_w[-1], fix_pos_w[-1] - shift_bias + shift_bias_max))
          fix_pos_w[-1] = fix_pos_w[-1] - shift_bias + shift_bias_max
          self.set_quant_param_config(node.op.param['weights'].name, fix_pos_w)
  
  def shift_normlization_weight_bias(self, node, filter_fn):
    if filter_fn(self.ctx, node):
      min_position = 0
      max_position = 15
      # weight shift
      if node.op.ParamName.GAMMA in node.op.params:
        wfp = self.get_quant_param_config(node.op.params[node.op.ParamName.GAMMA].name)
        if wfp[-1] < min_position:
          wfp[-1] = min_position
          self.set_quant_param_config(node.op.params[node.op.ParamName.GAMMA].name, wfp)
          NndctScreenLogger().warning2user(QWarning.SHIFT_CHECK, f"The fix position of weight in node {node.name} is less than {min_position} and it is adjusted to {min_position}.")
        elif wfp[-1] > max_position:
          wfp[-1] = max_position
          self.set_quant_param_config(node.op.params[node.op.ParamName.GAMMA].name, wfp)
          NndctScreenLogger().warning2user(QWarning.SHIFT_CHECK, f"The fix position of weight in node {node.name} is larger than {max_position} and it is adjusted to {max_position}.")

      # bias shift
      if node.op.ParamName.BETA in node.op.params:
        bfp = self.get_quant_param_config(node.op.params[node.op.ParamName.BETA].name)
        if bfp[-1] < min_position:
          bfp[-1] = min_position
          self.set_quant_param_config(node.op.params[node.op.ParamName.BETA].name, bfp)
          NndctScreenLogger().warning2user(QWarning.SHIFT_CHECK, f"The fix position of bias in node {node.name} is less than {min_position} and it is adjusted to {min_position}.")
        elif bfp[-1] > max_position:
          bfp[-1] = max_position
          self.set_quant_param_config(node.op.params[node.op.ParamName.BETA].name, bfp)
          NndctScreenLogger().warning2user(QWarning.SHIFT_CHECK, f"The fix position of bias in node {node.name} is larger than {max_position} and it is adjusted to {max_position}.")

class FixPosInserter(object):
  _instance_lock = threading.Lock()
  
  def __init__(self, ctx):
    self.ctx = ctx
    self.insert_ouput_fn_map_with_filter = {
      NNDCT_OP.CONCAT: [(self.insert_before_concat, lambda *args: NndctOption.nndct_insert_concat_input_fix.value)],
      NNDCT_OP.HSWISH: [self.insert_after_hswish]
    }
    self.__call_time = 0
    
  def __new__(cls, *args, **kwargs):
    if not hasattr(FixPosInserter, "_instance"):
      with FixPosInserter._instance_lock:
        if not hasattr(FixPosInserter, "_instance"):
          FixPosInserter._instance = object.__new__(cls)  
    return FixPosInserter._instance
    
  def insert_before_concat(self, node, filter_fn):
    if filter_fn(self.ctx, node):
      graph = node.owning_graph
      children = graph.children(node)[0]
      if self.ctx.configer.is_node_quantizable(children, self.ctx.lstm):
        bnfp = self.get_quant_output_config(children)
        self.insert_local_quant_config(node, bnfp)
  
  def insert_after_hswish(self, node):
    if self.get_quant_output_config(node) is None:
      end_node = self.ctx.configer.quant_output(node)
      bnfp = self.get_quant_output_config(end_node)
      if bnfp is not None:
        self.insert_local_quant_config(node, bnfp)
      
  def __call__(self, graph):
    if self.__call_time > 0:
      return
    self.__call_time = self.__call_time + 1
    
    # from head to tail for aligning with input OPs
    for node in graph.all_nodes():
      # if not self.ctx.configer.is_node_quantizable(node, self.ctx.lstm) \
      #   or not self.ctx.configer.is_concat_input(node.name):
      #   continue
      if self.ctx.configer.is_concat_input(node.name):
        fn_list = self.insert_ouput_fn_map_with_filter.get(NNDCT_OP.CONCAT, None)
        if fn_list is not None:
          for adjust_fn, filter_fn in fn_list:
            adjust_fn(node, filter_fn)
      if node.op.type == NNDCT_OP.HSWISH:
        fn_list = self.insert_ouput_fn_map_with_filter.get(NNDCT_OP.HSWISH, None)
        if fn_list is not None:
          for adjust_fn in fn_list:
            adjust_fn(node)

  def get_quant_output_config(self, node):
    if self.ctx.quant_config and node.name in self.ctx.quant_config['output']:
      return self.ctx.get_quant_config(node.name, False)
    else:
      return None
  
  def insert_local_quant_config(self, node, bnfp):
    self.ctx.insert_local_quant_config(node.name, bnfp)
