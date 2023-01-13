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
import os
import sys
import math
import json
import copy
import numpy as np
from scipy import stats
import torch
from os import environ
from torch.autograd import Variable
import pdb
import pathlib
from typing import Dict, Union
from pytorch_nndct.version import __version__
import pytorch_nndct as py_nndct
from pytorch_nndct.nn import fake_quantize_per_tensor
from pytorch_nndct.nn.modules.fix_ops import diffs_fix_pos
from pytorch_nndct.parse.parse_utils import ValueDeviceInfo
from pytorch_nndct.qproc.utils import quant_model_inferenced
from nndct_shared.base import key_names, NNDCT_KEYS, NNDCT_DEBUG_LVL, GLOBAL_MAP, NNDCT_OP
from nndct_shared.base.key_names import FrameworkType
from nndct_shared.quantization import OriginBaseQuantizer
from nndct_shared.utils.log import nndct_debug_print
from nndct_shared.utils import NndctOption, tensor_util, NndctScreenLogger, NndctDebugLogger, QError, QWarning, QNote
from nndct_shared.quantization import DPUQstrategy, LstmQstrategy, TQTStrategy

import nndct_shared.utils as nndct_utils
import nndct_shared.quantization as nndct_quant

global_snr_inv = 0.0
def has_inf_nan():
  return hasattr(torch, "isinf") and hasattr(torch, "isnan")

def is_valid_tensor_for_quantizer(tensor):
  if has_inf_nan():
    inf = torch.isinf(tensor)
    nan = torch.isnan(tensor)
    if inf.sum() > 0 or nan.sum() > 0:
        return False
  return True

class TORCHQuantizer(OriginBaseQuantizer):

  def __init__(self,
               quant_mode: int,
               output_dir: str,
               quant_config: Dict[str, Union[str, int, bool]],
               is_lstm = False):
    super().__init__(quant_mode,
                     output_dir,
                     quant_config,
                     is_lstm)
    if NndctOption.nndct_param_corr.value > 0:
      if self.quant_mode == 2:
        path = pathlib.Path(self.bias_corr_file)
        if not (path.exists() and path.is_file()):
          NndctScreenLogger().error2user(QError.BIAS_CORRECTION, f"Bias correction result file does not exist. \
Please check calibration with bias correction is done or not.")
          exit(2)
        self.bias_corr = torch.load(self.bias_corr_file)
        self._bias_corr_loaded = True
    self.exporting = False
    self.inplace = True
    self.output_dir = output_dir
    self._scripts = []
    
    mix_bit = quant_config['mix_bit']
    if is_lstm:
      self.quant_strategy = LstmQstrategy(quant_config)
    else:
      if mix_bit:
        self.quant_strategy = TQTStrategy(quant_config)
      else:
        self.quant_strategy = DPUQstrategy(quant_config)

  def add_script(self, script):
    if script not in self._scripts:
      self._scripts.append(script)
 
  @property
  def scripts(self):
    return self._scripts
  
  def get_model_type(self):
    return FrameworkType.TORCH

  def features_check(self):
    if self.fast_finetuned and not self._finetuned_para_loaded:
      NndctScreenLogger().warning2user(QWarning.FAST_FINETUNE, f'Fast finetuned parameters are not loaded. \
Call load_ft_param to load them.')
    if self.bias_corrected and not self._bias_corr_loaded:
      NndctScreenLogger().warning2user(QWarning.BIAS_CORRECTION, f'Bias correction file is not loaded. Set \
command line option \"--nndct_param_corr\" to load it.')

  def reset_status_for_exporting(self):
    def _reset_param_quantized(model):
      for mod in model.modules():
        if hasattr(mod, "param_quantized"):
          setattr(mod, "param_quantized", False)
  
    self.exporting = True
    self.inplace = False
    if isinstance(self._quant_model, list):
      for q_model in self._quant_model:
        _reset_param_quantized(q_model)
    else:
      _reset_param_quantized(self._quant_model)

  def do_scan(self,
              res,
              name,
              node=None,
              tensor_type='input',
              idx=0,
              method=None):
    # keep quantization steps after fast finetune
    if self.keep_fp:
      return self.do_quantize(res, name, node, tensor_type)

    # Don't support quantizing multi-output
    # if isinstance(res,  (list, tuple)):
    #   return res
      
    # forward quant graph but not quantize parameter and activation
    if NndctOption.nndct_quant_off.value:
      if self.inplace:
        return res
      else:
        return res.clone().detach()

    res_save = None
    if isinstance(res.values, torch.Tensor):
      if res.values.data.numel() == 0:
        if self.inplace:
          return res
        else:
          return copy.deepcopy(res)
      res_save = res
      res = res.values.data
    else:
      if res.data.numel() == 0:
        if self.inplace:
          return res
        else:
          return res.clone().detach()
      
    if res.dtype != torch.float32 and res.dtype != torch.double and res.dtype != torch.float16:
      NndctScreenLogger().warning2user_once(QWarning.TENSOR_TYPE_NOT_QUANTIZABLE, f'The tensor type of {node.name} is {str(res.dtype)}. Only support float32/double/float16 quantization.')
      return res_save if res_save is not None else res

    if not is_valid_tensor_for_quantizer(res):
      NndctScreenLogger().warning2user_once(QWarning.TENSOR_VALUE_INVALID, f'The tensor type of {node.name} have "inf" or "nan" value.The quantization for this tensor is ignored.Please check it.')
      return res_save if res_save is not None else res

    # get fixed position
    bnfp = self.get_quant_config(name, False, tensor_type, idx)
    if bnfp is None:
      return res_save if res_save is not None else res

    if method is not None:
      mth = method
    else:
      # hardware cut method
      mth = 4 if self.lstm else 2
      
      if NndctOption.nndct_use_torch_quantizer.value is True:
        mth = -1
      elif tensor_type == 'param':
        mth = 3
      elif self.lstm and NndctOption.nndct_ip_asr.value is True:
        mth = 3

    scope = 5 if NndctOption.nndct_diffs_mode.value == "mse" else 1
    # set fix pos scanning scope to 1 for some type of tensors
    if (node.op.type in [NNDCT_OP.INPUT, NNDCT_OP.QUANT_STUB]):
      scope = 1
    if (self.lstm and tensor_type == 'input'):
      scope = 1
      res = res.detach().clone()

    # activation always calculate fix pos
    # calcualte fix pos if it is None
    # always calculate fis pos in finetune mode
    
    if tensor_type != 'param' or bnfp[1] is None or self.quant_mode == 3:
      Tfixpos = diffs_fix_pos(
          input = res,
          bit_width = bnfp[0],
          scope = scope,
          method = mth)
      if has_inf_nan() and (torch.isinf(Tfixpos) or torch.isnan(Tfixpos)):
        return res_save if res_save is not None else res
      bnfp[1] = (int)(Tfixpos.item())
      # limit max fix pos to 12 if bit width <= 8, others limit to 15
      if bnfp[0] <= 8 or self.lstm:
        max_fp = NndctOption.nndct_max_fix_position.value
        bnfp[1] = min(max_fp, bnfp[1])
      else:
        bnfp[1] = min(15, bnfp[1])
      # record fix pos of activation
      if tensor_type != 'param':
        self.config_history[tensor_type][name].append(bnfp[1])
        if (NndctOption.nndct_stat.value > 1):
          print(f'---- fp history: {stats.mode(np.array(self.config_history[tensor_type][name]))}')
        data = np.array(self.config_history[tensor_type][name])
        bnfp[1] = stats.mode(data)[0][0]
        bnfp[1] = bnfp[1].astype(np.int32).tolist()
      self.set_quant_config(name, bnfp, tensor_type, idx)
      if (NndctOption.nndct_stat.value > 1):
        print('---- quant %s tensor: %s with bw = %d and fp = %g' % (
            tensor_type, name, bnfp[0], bnfp[1]))

      # get 2^bit_width and 2^fracpos
      bnfp = self.get_quant_config(name, True, tensor_type, idx)

      if (NndctOption.nndct_stat.value > 2):
        quant_data = nndct_quant.QuantizeData(name, res.cpu().detach().numpy())

      # do quantization for parameter or activation
      res = fake_quantize_per_tensor(res, bnfp[1], 0, -bnfp[0], bnfp[0]-1, mth, self.inplace)
        
      if (NndctOption.nndct_stat.value > 2):
        #quant_data.all_close(res.cpu().detach().numpy())
        global global_snr_inv
        quant_efficiency, sqnr = quant_data.quant_efficiency(res.cpu().detach().numpy(), math.log2(bnfp[0]))
        global_snr_inv += 1 / sqnr
        if quant_efficiency < 3.0:
          print(f"quant_efficiency={quant_efficiency}, {quant_data._name}\n")
          print('Statistic [Min, Max, Mean, Std]:')
          print('[{}, {}, {}, {}]'.format( res.min(), res.max(), res.mean(), res.std() ))
          print('histogram: {}'.format( res.histc(bins = 10).cpu().detach().numpy() ))
          t = res
          if tensor_type != 'param':
            t = res.transpose(0, 1)
          print('Channel number:{}'.format(t.shape[0]))
          print('Channel-wise statistic [Min, Max, Mean, Std]:')
          for c in range(t.shape[0]):
            print('[{}, {}, {}, {}]'.format( t[c].min(), t[c].max(), t[c].mean(), t[c].std() ))
            print('histogram: {}'.format( t[c].histc(bins = 10).cpu().detach().numpy() ))
      
    if res_save is not None:
      res_save.values.data = res
      res = res_save

    return res

  def do_quantize(self, blob, name, node=None, tensor_type='input', idx=0, method=None):

    # if isinstance(blob, (list, tuple)):
    #   return blob

    # forward quant graph but not quantize parameter and activation
    if NndctOption.nndct_quant_off.value:
      if self.inplace:
        return blob
      else:
        return blob.clone().detach()
    
    blob_save = None
    if isinstance(blob.values, torch.Tensor):
      blob_save = blob
      blob = blob.values.data
    
    if blob.dtype != torch.float32 and blob.dtype != torch.double and blob.dtype != torch.float16:
      NndctScreenLogger().warning2user_once(QWarning.TENSOR_TYPE_NOT_QUANTIZABLE, f'The tensor type of {node.name} is {str(blob.dtype)}. Only support float32/double/float16 quantization.')
      return blob_save if blob_save is not None else blob

    if not is_valid_tensor_for_quantizer(blob):
      NndctScreenLogger().warning2user_once(QWarning.TENSOR_VALUE_INVALID, f'The tensor type of {node.name} have "inf" or "nan" value. The quantization is ignored. Please check it.')
      return blob_save if blob_save is not None else blob
    
    if (NndctOption.nndct_stat.value > 2):
      quant_data = nndct_quant.QuantizeData(name, blob.cpu().detach().numpy())
    # quantize the tensor
    bnfp = self.get_quant_config(name, True, tensor_type, idx)
    if bnfp is None:
      return blob_save if blob_save is not None else blob

    if (NndctOption.nndct_stat.value > 1):
        print('---- quant %s tensor: %s with 1/step = %g' % (
            tensor_type, name, bnfp[1]))

    if method is not None:
      mth = method
    else:
      # hardware cut method
      mth = 4 if self.lstm else 2
      
      if NndctOption.nndct_use_torch_quantizer.value is True:
        mth = -1
      elif tensor_type == 'param':
        mth = 3
      elif self.lstm and NndctOption.nndct_ip_asr.value is True:
        mth = 3

    res = fake_quantize_per_tensor(blob, bnfp[1], 0, -bnfp[0], bnfp[0]-1, mth, self.inplace)

    if (NndctOption.nndct_stat.value > 2):
      global global_snr_inv
      quant_efficiency, sqnr = quant_data.quant_efficiency(res.cpu().detach().numpy(), 8)
      global_snr_inv += 1 / sqnr
      if quant_efficiency < 3.0:
        print(f"quant_efficiency={quant_efficiency}, global_snr_inv={global_snr_inv} {quant_data._name}\n")
        print('Network input channel-wise statistic [Min, Max, Mean, Std]:')
        print('[{}, {}, {}, {}]'.format( res.min(), res.max(), res.mean(), res.std() ))
        print('histogram: {}'.format( res.histc(bins = 10).cpu().detach().numpy() ))
        t = res
        if tensor_type != 'param':
          t = res.transpose(0, 1)
        print('Channel number:{}'.format(t.shape[0]))
        print('Channel-wise statistic [Min, Max, Mean, Std]:')
        for c in range(t.shape[0]):
          print('[{}, {}, {}, {}]'.format( t[c].min(), t[c].max(), t[c].mean(), t[c].std() ))
          print('histogram: {}'.format( t[c].histc(bins = 10).cpu().detach().numpy() ))

    # update param to nndct graph
    if tensor_type == 'param' and not self.exporting:
      self.update_param_to_nndct(node, name, res.cpu().detach().numpy())
    
    if tensor_type == 'param':
      self.graph.param_tensor(name).device = ValueDeviceInfo('cpu') if res.device == torch.device('cpu') \
        else ValueDeviceInfo('cuda')
    elif tensor_type == 'output':
      node.out_tensors[idx].device = ValueDeviceInfo('cpu') if res.device == torch.device('cpu') \
        else ValueDeviceInfo('cuda')
    else:
      node.in_tensors[idx].device = ValueDeviceInfo('cpu') if res.device == torch.device('cpu') \
        else ValueDeviceInfo('cuda')
    
    if blob_save is not None:
      blob_save.values.data = res
      res = blob_save

    return res

  def _check_calibration_completion_for_target(self):
    def _check(tensor_type):
      ret = True
      for item in self._QuantInfo[tensor_type]:
        for idx in range(self.get_quant_len(item, tensor_type)):
          bnfp = self._QuantInfo[tensor_type][item][idx]
          if bnfp is None:
            continue
          if bnfp[1] is None:
            NndctScreenLogger().warning2user(QWarning.TENSOR_NOT_QUANTIZED, f'{tensor_type.capitalize()} tensor is not quantized: {item}')
            ret = False
      return ret
    return all([_check(tensor_type) for tensor_type in ["output", "input", "param"]])
    
    
  def _check_calibration_completion(self):
    if hasattr(self.configer, "_device_allocator"):
      return self._check_calibration_completion_for_target()
    
    ret = True
    # Check node output tensors
    for node in self.Nndctgraph.all_nodes():
      if self.configer.is_node_quantizable(node, self.lstm):
        qout = self.configer.quant_output(node.name).name
        bnfp = self.get_quant_config(qout, False)
        if bnfp[1] is None:
          if node.op.type not in [NNDCT_OP.SIGMOID, NNDCT_OP.TANH]:
            NndctScreenLogger().warning2user(QWarning.TENSOR_NOT_QUANTIZED,f'Node ouptut tensor is not quantized: {node.name} type: {node.op.type}')
            ret = False
    # Check node input tensors 
    for item in self.quant_config['input']:
      for idx in range(self.get_quant_len(item, 'input')):
        bnfp = self.get_quant_config(item, False, 'input', idx)
        if bnfp[1] is None:
          NndctScreenLogger().warning2user(QWarning.TENSOR_NOT_QUANTIZED, f'Input tensor is not quantized: {item}')
          ret = False
    # Check node parameters
    for item in self.quant_config['param']:
      for idx in range(self.get_quant_len(item, 'param')):
        bnfp = self.get_quant_config(item, False, 'param', idx)
        if bnfp[1] is None:
          NndctScreenLogger().warning2user(QWarning.TENSOR_NOT_QUANTIZED, f'Parameter tensor is not quantized: {item}. \
If this parameter is not a parameter embedded in torch operation, like weights of CONV, \
this kind of parameter will not be quantized. Please embed the parameter into CONV/Linear/BN.')
          ret = False

    return ret

  def _align_lstm_fix_pos_with_cell_output(self):
    if self.lstm:
      if self.rnn_front_end:
        for node in self.Nndctgraph.all_nodes():
          nextNode = self.configer.Nndctgraph.get_node_by_idx(idx=node.idx + 1)
          # find the last node of every cell
          if (nextNode is not None and nextNode.name.split('::')[-1] == 'input_0'
              or node.idx == len(list(self.configer.Nndctgraph.nodes)) - 1):
            #print('----find output h node %s' % node.name)
            for nid in range(node.idx-1, -1, -1):
              prevNode = self.configer.Nndctgraph.get_node_by_idx(idx=nid)
              # fragpos of sigmoid and tanh keep 15
              if prevNode.op.type in [NNDCT_OP.SIGMOID, NNDCT_OP.TANH]:
                bnfp = self.get_quant_config(prevNode.name, False)
                bnfp[1] = 15
                self.set_quant_config(prevNode.name, bnfp)
                continue
              if prevNode.name.split('::')[-1] == 'input_0':
                break
              bnfp = self.get_quant_config(node.name, False)
              #print('----set %s fix pos to %d' % (prevNode.name, bnfp[1]))
              target_name = self.configer.quant_output(prevNode.name).name
              # multiple output node is not grouped up with children,
              # for example, strided_slice is not grouped up
              if target_name in self.quant_config['output'].keys():
                self.set_quant_config(target_name, bnfp)
      else: # to handle dlrm
        # fragpos of sigmoid and tanh keep 15
        for node in self.Nndctgraph.all_nodes():
          if node.op.type in [NNDCT_OP.SIGMOID, NNDCT_OP.TANH]:
            bnfp = self.get_quant_config(node.name, False)
            bnfp[1] = 15
            self.set_quant_config(node.name, bnfp)

        # find the last non-sigmoid/tanh node
        node_num = len(list(self.configer.Nndctgraph.nodes))
        for idx in range(node_num-1, -1, -1):
          last_node = self.configer.Nndctgraph.get_node_by_idx(idx)
          if last_node.op.type not in [NNDCT_OP.SIGMOID, NNDCT_OP.TANH]:
              break
        last_bnfp = self.get_quant_config(last_node.name, False)

        # back propagate last_node's bnfp to previous nodes which are non-sigmoid/tanh
        for idx in range(last_node.idx-1, -1, -1):
          node = self.configer.Nndctgraph.get_node_by_idx(idx)
          if node.op.type in [NNDCT_OP.SIGMOID, NNDCT_OP.TANH]:
            continue
          elif node.name.split('::')[-1] == 'input_0':
            break
          else:
            self.set_quant_config(node.name, last_bnfp)


  def _node_relative_fix_pos_check(self):
    for node in self.Nndctgraph.all_nodes():
      # shift_bias and shift_cut check
      if (node.in_quant_part and self.configer.is_conv_like(node) and
          self.configer.is_node_quantizable(node, self.lstm)):
        # get i/w/b/o fix pos
        conv_quant_output = self.configer.quant_output(node.name).name
        fix_pos_i = self.get_in_node_config(node, 0, False)
        fix_pos_o = self.get_quant_config(conv_quant_output, False, 'output')
        fix_pos_w = self.get_quant_config(node.op.param['weights'].name, False, 'param')
        if fix_pos_i[-1] == None:
          NndctScreenLogger().warning2user(QWarning.TENSOR_NOT_QUANTIZED, f"Output tensor of node {node.in_nodes[0]} is not quantized.")
          break
	      # handle shift_cut
        shift_cut = fix_pos_w[-1] + fix_pos_i[-1] - fix_pos_o[-1]
        shift_cut_min = 0
        shift_cut_max = 16
        if shift_cut < shift_cut_min:
          NndctScreenLogger().warning2user(QWarning.SHIFT_CHECK, "output {} value is too small, so adjust the fix position from {} to {}"
                  .format(conv_quant_output, fix_pos_o[-1], fix_pos_o[-1] + shift_cut - shift_cut_min))
          fix_pos_o[-1] = fix_pos_o[-1] + shift_cut - shift_cut_min;
          self.set_quant_config(conv_quant_output, fix_pos_o)
        elif shift_cut > shift_cut_max:
          NndctScreenLogger().warning2user(QWarning.SHIFT_CHECK, "weight {} value is too small, so adjust the fix position from {} to {}"
                  .format(node.name, fix_pos_w[-1], fix_pos_w[-1] - shift_cut + shift_cut_max))
          fix_pos_w[-1] = fix_pos_w[-1] - shift_cut + shift_cut_max;
          self.set_quant_config(node.op.param['weights'].name, fix_pos_w, tensor_type = "param")

        # handle shift_bias
        if node.module.bias is not None:
          fix_pos_b = self.get_quant_config(node.op.param['bias'].name, False, 'param')
          shift_bias = fix_pos_w[-1] + fix_pos_i[-1] - fix_pos_b[-1]
          shift_bias_min = min(0, -(24-(8+shift_cut)))
          shift_bias_max = 16
          if shift_bias < shift_bias_min:
            NndctScreenLogger().warning2user(QWarning.SHIFT_CHECK, "bias {} value is too small, so adjust the fix position from {} to {}"
                    .format(node.op.param['bias'].name, fix_pos_b[-1], fix_pos_b[-1] + shift_bias - shift_bias_min))
            fix_pos_b[-1] = fix_pos_b[-1] + shift_bias - shift_bias_min;
            self.set_quant_config(node.op.param['bias'].name, fix_pos_b, tensor_type = "param")
          elif shift_bias > shift_bias_max:
            NndctScreenLogger().warning2user(QWarning.SHIFT_CHECK, "weight {} value is too small, so adjust the fix position from {} to {}"
                    .format(node.op.param['weights'].name, fix_pos_w[-1], fix_pos_w[-1] - shift_bias + shift_bias_max))
            fix_pos_w[-1] = fix_pos_w[-1] - shift_bias + shift_bias_max;
            self.set_quant_config(node.op.param['weights'].name, fix_pos_w, tensor_type = "param")

  def organize_quant_pos(self):
    # Transfer inplace operation fragpos forward,
    # to replace configerComannder in future
    if NndctOption.nndct_quant_off.value:
      return

    # check quantization calibration is performed completely
    if not self._check_calibration_completion():
      NndctScreenLogger().warning2user(QWarning.TENSOR_NOT_QUANTIZED,  "Some tensors are not quantized, please check their particularity.")

    # align lstm fix pos with cell output
    #self._align_lstm_fix_pos_with_cell_output()

    # from tail to head for aligning with output OPs
    for node in self.Nndctgraph.reverse_nodes:
      # change concat input nodes fix point to be the same as concat output node
      if (node.in_quant_part and
          node.op.type == NNDCT_OP.CONCAT and
          NndctOption.nndct_change_concat_input_fix.value):
        bnfp = self.get_quant_config(node, False)
        for in_node in node.in_nodes:
          self.set_quant_config(in_node, bnfp)

      # change add input nodes fix point to be the same as its output
      if (node.in_quant_part and
          node.op.type == NNDCT_OP.ADD and 
          NndctOption.nndct_change_add_input_fix.value):
        bnfp = self.get_quant_config(node, False)
        for in_node in node.in_nodes:
          #print(f'Before change add fix of {node.name} {node.op.type} : {self.get_quant_config(in_node, False)}')
          self.set_quant_config(in_node, bnfp)
          #print(f'After change : {self.get_quant_config(in_node, False)}')

      # change pooling input nodes fix point to be the same as its output node
      if (node.in_quant_part and
          node.op.type in [NNDCT_OP.MAX_POOL, 
                           NNDCT_OP.MAX_POOL1D,
                           NNDCT_OP.AVG_POOL, 
                           NNDCT_OP.ADAPTIVEAVGPOOL2D] 
          and NndctOption.nndct_change_pool_input_fix.value):
        bnfp = self.get_quant_config(node, False)
        for in_node in node.in_nodes:
          #print(f'Before change pool fix of {node.name} {node.op.type} : {self.get_quant_config(in_node, False)}')
          self.set_quant_config(in_node, bnfp)
          #print(f'After change : {self.get_quant_config(in_node, False)}')

    # from head to tail for aligning with input OPs
    for node in self.Nndctgraph.all_nodes():
      # align linear OP bias fix pos with output for lstm
      if not self.configer.is_node_quantizable(node, self.lstm):
        continue
      if self.lstm:
        if node.op.type == NNDCT_OP.DENSE:
          if len(node.op.params.values()) > 1:  
            params = [v.name for v in node.op.params.values()]
            bnfp = self.get_quant_config(node.name, False)
            self.set_quant_config(params[1], bnfp, 'param')
      # Strided_slice branches fix pos alignment
      if (node.in_quant_part and
          node.op.type == NNDCT_OP.STRIDED_SLICE):
        bnfp = None
        src_name = self.configer.quant_output(node.in_nodes[0]).name
        if self.need_quantize_tensor(src_name):
          bnfp = self.get_in_node_config(node, 0, False)
          self.quant_config['output'][node.name] = [[bnfp[0], bnfp[1]]]
          #print('Strided_Slice fix pos setting node: {} qout: {} pos: {}'.format(
          #    node.name, src_name, bnfp[1]))

      # zero padding output fix pos align with input
      if (node.in_quant_part and
          node.op.type == NNDCT_OP.PAD):
        out_name = self.configer.quant_output(node.name).name
        bnfp = self.get_in_node_config(node, 0, False)
        #print('---- set zero padding output %s fix pos to %s : %d' % (out_name, in_name, bnfp[1]))
        self.set_quant_config(out_name, bnfp)
        
      if (node.in_quant_part and node.op.type == NNDCT_OP.RESIZE and node.node_config('mode') == "'nearest'"):
        out_node = self.configer.quant_output(node.name)
        out_name = out_node.name
        if out_node.op.type != NNDCT_OP.CONCAT:
          bnfp = self.get_in_node_config(node, 0, False)
          #print('---- set nearest upsampling output %s fix pos to %s : %d' % (out_name, in_name, bnfp[1]))
          self.set_quant_config(out_name, bnfp)
    
      # limit hardsigmoid output fix pos to >= 7
      if (node.in_quant_part and
          node.op.type == NNDCT_OP.HSIGMOID):
        out_name = self.configer.quant_output(node.name).name
        bnfp = self.get_quant_config(out_name, False)
        #print('checking {}: {}'.format(out_name, bnfp))
        if bnfp[1] < 7:
          bnfp[1] = 7
        self.set_quant_config(out_name, bnfp)

      # clamp hsigmoid/hswish input node fix point to non-negative due to IP limitation
      if (node.in_quant_part and
          node.op.type in [NNDCT_OP.HSIGMOID, NNDCT_OP.HSWISH]):
        for idx, in_node in enumerate(node.in_nodes):
          #print(f'Before change pool fix of {node.name} {node.op.type} : {self.get_quant_config(in_node, False)}')
          bnfp = self.get_in_node_config(node, idx, False)
          if bnfp[1] < 0:
            bnfp[1] = 0
          self.set_quant_config(in_node, bnfp)
          #print(f'After change : {self.get_quant_config(in_node, False)}')

      # hardswish input_fp + 7 >= output_fp
      if (node.in_quant_part and
          node.op.type == NNDCT_OP.HSWISH):
          bnfp_o = self.get_quant_config(node, False)
          bnfp_i = self.get_in_node_config(node, 0, False)
          #print(f'Before change pool fix of {node.name} {node.op.type} : {self.get_quant_config(node, False)}')
          if bnfp_i[1] + 7 < bnfp_o[1]:
            bnfp_o[1] = bnfp_i[1] + 7
          self.set_quant_config(node, bnfp_o)
          #print(f'After change : {self.get_quant_config(node, False)}')

    # shift_bias and shift_cut check and adjustment
    if NndctOption.nndct_ip_asr.value is False:
      self._node_relative_fix_pos_check()

  def export_quant_config(self, export_file=None, adjust_pos=True, inference_check=True):
    if inference_check and quant_model_inferenced(self.quant_model) is False:
      NndctScreenLogger().error2user(QError.NO_CALIBRATION, "Quantization is not performed completely, check if module FORWARD function is called!\n    FORWARD function of torch_quantizer.quant_model needs to be called in user code explicitly.\n    Please refer to the example code at https://github.com/Xilinx/Vitis-AI/blob/master/src/Vitis-AI-Quantizer/vai_q_pytorch/example/resnet18_quant.py.")
      return
    if NndctOption.nndct_param_corr.value > 0:
      if self.quant_mode == 1:
        # gather bias correction, how to get nn module objec?
        for node in self.Nndctgraph.all_nodes():
          if node.op.type in [NNDCT_OP.CONV1D,
                              NNDCT_OP.CONV2D,
                              NNDCT_OP.CONVTRANSPOSE2D,
                              NNDCT_OP.DEPTHWISE_CONV2D,
                              NNDCT_OP.DENSE,
                              NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D]:
            if node.module.bias is not None:
              self.bias_corr[node.name] = node.module.bias_corr()

        # export bias correction
        torch.save(self.bias_corr, self.bias_corr_file)
        self.bias_corrected = True

    # export quant steps
    self.version = __version__
    file_name = export_file or self.export_file
    if isinstance(file_name, str):
        NndctScreenLogger().info(f"=>Exporting quant config.({file_name})")
        if adjust_pos:
          self.organize_quant_pos()
        with open(file_name, 'w') as f:
          f.write(nndct_utils.to_jsonstr(self.quant_config))

  def update_param_to_nndct(self, node, param_name, param_data):
    for param_type, tensor in node.op.params.items():
      if tensor.name == param_name:
        if node.op.type in [NNDCT_OP.CONVTRANSPOSE2D, NNDCT_OP.CONVTRANSPOSE3D] and param_type == node.op.ParamName.WEIGHTS:
            param_data = np.copy(param_data).swapaxes(1, 0)
            param_data = np.ascontiguousarray(param_data)
            
        if node.op.type in [NNDCT_OP.DEPTHWISE_CONV2D, NNDCT_OP.DEPTHWISE_CONV3D] and param_type == node.op.ParamName.WEIGHTS:
          in_channels = node.node_config("in_channels")
          out_channels = node.node_config("out_channels")
          kernel_size = node.node_config("kernel_size")
          channel_mutiplier = int(out_channels / in_channels)
          param_data = param_data.reshape((channel_mutiplier, in_channels, *kernel_size))
        
        if node.op.type in [NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D] and param_type == node.op.ParamName.WEIGHTS:
          in_channels = node.node_config("in_channels")
          out_channels = node.node_config("out_channels")
          kernel_size = node.node_config("kernel_size")
          channel_mutiplier = int(out_channels / in_channels)
          param_data = param_data.reshape((in_channels, channel_mutiplier, *kernel_size))
          param_data = np.copy(param_data).swapaxes(0, 1)
          param_data = np.ascontiguousarray(param_data)
        
        origin_shape = tensor.shape
        
        tensor.from_ndarray(param_data)
        tensor_util.convert_parameter_tensor_format(
            tensor, key_names.FrameworkType.TORCH,
            key_names.FrameworkType.NNDCT)
        
        NndctScreenLogger().check2user(QError.SHAPE_MISMATCH, f"The shape of data '{tensor.shape}' must be consistent with that of original data '{origin_shape}' for {tensor.name}", origin_shape == tensor.shape)
  
  def export_param(self):
    if self.quant_mode == 1:
      if isinstance(self.quant_model, (list, tuple)) and len(self.quant_model)>0:
        for i in range(len(self.quant_model)):
          param_file = self.param_file + str(i) + '.pth'
          NndctScreenLogger().info(f"=>Exporting quant model parameters.({param_file})")
          torch.save(self.quant_model[i].state_dict(), param_file)
      else:
        param_file = self.param_file + '.pth'
        NndctScreenLogger().info(f"=>Exporting quant model parameters.({param_file})")
        torch.save(self.quant_model.state_dict(), param_file)
      self.fast_finetuned = True

  def load_param(self):
    if self.quant_mode == 2:
      if isinstance(self.quant_model, (list, tuple)) and len(self.quant_model)>0:
        for i in range(len(self.quant_model)):
          param_file = self.param_file + str(i) + '.pth'
          NndctScreenLogger().info(f"=>Loading quant model parameters.({param_file})")
          path = pathlib.Path(param_file)
          if not (path.exists() and path.is_file()) or not self.fast_finetuned:
            NndctScreenLogger().error2user(QError.FAST_FINETINE, f"Fast finetuned parameter file does not exist. \
Please check calibration with fast finetune is done or not.")
            exit(2)
          self.quant_model[i].load_state_dict(torch.load(param_file))
      else:
        param_file = self.param_file + '.pth'
        NndctScreenLogger().info(f"=>Loading quant model parameters.({param_file})")
        path = pathlib.Path(param_file)
        if not (path.exists() and path.is_file()) or not self.fast_finetuned:
          NndctScreenLogger().error2user(QError.FAST_FINETINE, f"Fast finetuned parameter file does not exist. \
Please check calibration with fast finetune is done or not.")
          exit(2)
        self.quant_model.load_state_dict(torch.load(param_file))
      self._finetuned_para_loaded = True
      
  def export_float_param(self):
    if not os.path.exists(self.float_param_path):
      os.makedirs(self.float_param_path)
    if self.quant_mode == 1:
      if isinstance(self.quant_model, (list, tuple)) and len(self.quant_model)>0:
        for i in range(len(self.quant_model)):
          module_name = self.quant_model[i]._get_name()
          param_file = self.float_param_path + '/' + module_name + '.pth'
          NndctScreenLogger().info(f"=>Exporting float model parameters.({param_file})")
          torch.save(self.quant_model[i].state_dict(), param_file)
      else:
        module_name = self.quant_model._get_name()
        param_file = self.float_param_path + '/' + module_name + '.pth'
        NndctScreenLogger().info(f"=>Exporting float model parameters.({param_file})")
        torch.save(self.quant_model.state_dict(), param_file)
      #self.fast_finetuned = True

  def load_float_param(self):
    if isinstance(self.quant_model, (list, tuple)) and len(self.quant_model)>0:
      for i in range(len(self.quant_model)):
        module_name = self.quant_model[i]._get_name()
        param_file = self.float_param_path + '/' + module_name + '.pth'
        NndctScreenLogger().info(f"=>Loading float model parameters.({param_file})")
        path = pathlib.Path(param_file)
        if not (path.exists() and path.is_file()):
          NndctScreenLogger().error(f"Float model parameter file does not exist.")
          exit(2)
        self.quant_model[i].load_state_dict(torch.load(param_file))
    else:
      module_name = self.quant_model._get_name()
      param_file = self.float_param_path + '/' + module_name + '.pth'
      NndctScreenLogger().info(f"=>Loading float model parameters.({param_file})")
      path = pathlib.Path(param_file)
      if not (path.exists() and path.is_file()):
        NndctScreenLogger().error(f"Float model parameter file does not exist.")
        exit(2)
      self.quant_model.load_state_dict(torch.load(param_file))
      
  def contain_channel_quantize(self):
    return False

 
