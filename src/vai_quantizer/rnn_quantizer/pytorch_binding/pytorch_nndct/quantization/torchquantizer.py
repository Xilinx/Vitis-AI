
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
import json
import copy
import numpy as np
from scipy import stats
import torch
from os import environ
from torch.autograd import Variable
import pdb

import pytorch_nndct as py_nndct

from nndct_shared.base import key_names, NNDCT_KEYS, NNDCT_DEBUG_LVL, GLOBAL_MAP, NNDCT_OP
from nndct_shared.base.key_names import FrameworkType
from nndct_shared.quantization import BaseQuantizer
from nndct_shared.utils.log import nndct_debug_print
from nndct_shared.utils import NndctOption, tensor_util, NndctScreenLogger, NndctDebugLogger

import nndct_shared.utils as nndct_utils
import nndct_shared.quantization as nndct_quant

global_snr_inv = 0.0
class TORCHQuantizer(BaseQuantizer):

  def __init__(self,
               quant_mode: int,
               output_dir: str,
               bitwidth_w: int,
               bitwidth_a: int,
               mix_bit: bool):
    super().__init__(quant_mode,
                     output_dir,
                     bitwidth_w,
                     bitwidth_a,
                     mix_bit)
    self._quant_model = None
    if NndctOption.nndct_param_corr.value > 0:
      if self.quant_mode == 2:
        self.bias_corr = torch.load(self.bias_corr_file)

  def get_model_type(self):
    return FrameworkType.TORCH

  def do_scan(self,
              res,
              name,
              node=None,
              tensor_type='input'):
    # keep quantization steps after fast finetune
    if self.keep_fp:
      return self.do_quantize(res, name, node, tensor_type)

    # forward quant graph but not quantize parameter and activation
    if NndctOption.nndct_quant_off.value:
      return res

    res_save = None
    if isinstance(res.values, torch.Tensor):
      res_save = res
      res = res.values.data
      
    if res.dtype != torch.float32 and res.dtype != torch.double:
      NndctScreenLogger().warning_once(f'The tensor type of  {node.name} is {str(res.dtype)}. Only support float32/double quantization.')
      return res_save if res_save is not None else res
    
    quant_device = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_DEVICE)
    if res.device.type != quant_device.type:
      raise TypeError("Device of quantizer is {}, device of model and data should match device of quantizer".format(quant_device.type))

    # get fixed position
    bnfp = self.get_bnfp(name, False, tensor_type)

    # hardware cut method
    mth = 4 if self.lstm else 2
    if tensor_type == 'param':
      mth = 3

    scope = 5 if NndctOption.nndct_diffs_mode.value == "mse" else 1
    # set fix pos scanning scope to 1 for some type of tensors
    if (node.op.type in [NNDCT_OP.INPUT, NNDCT_OP.QUANT_STUB]):
      scope = 1
    if (self.lstm and tensor_type == 'input'):
      scope = 1
      res = res.detach().clone()

    '''
    if(quant_device.type == "cpu"):
      Tbuffer = torch.empty_like(res).to(torch.device("cpu"))
      Tfixpos = torch.tensor([1], dtype=torch.get_default_dtype()).to(torch.device("cpu"))
    else:
      Tbuffer = torch.empty_like(res).cuda()
      Tfixpos = torch.tensor([1], dtype=torch.get_default_dtype()).cuda()
    '''

    Tbuffer = torch.empty_like(res).to(quant_device)
    Tfixpos = torch.tensor([1], dtype=torch.get_default_dtype()).to(quant_device)

    # activation always calculate fix pos
    # calcualte fix pos if it is None
    # always calculate fis pos in finetune mode
    
      
    
    if tensor_type != 'param' or bnfp[1] is None or self.quant_mode == 3:
      py_nndct.nn.NndctDiffsFixPos(
          Tinput = res,
          Tbuffer = Tbuffer,
          Tfixpos = Tfixpos,
          bit_width = bnfp[0],
          range = scope,
          method = mth)
      bnfp[1] = (int)(Tfixpos.item())
      # limit max fix pos to 12 if bit width <= 8, others limit to 15
      if bnfp[0] <= 8 or self.lstm:
        max_fp = NndctOption.nndct_max_fix_position.value
        bnfp[1] = min(max_fp, bnfp[1])
      else:
        bnfp[1] = min(15, bnfp[1])
      # record fix pos of activation
      if tensor_type != 'param':
        self.fp_history[tensor_type][name].append(bnfp[1])
        if (NndctOption.nndct_stat.value > 1):
          print(f'---- fp history: {stats.mode(np.array(self.fp_history[tensor_type][name]))}')
        data = np.array(self.fp_history[tensor_type][name])
        bnfp[1] = stats.mode(data)[0][0]
        bnfp[1] = bnfp[1].astype(np.int32).tolist()
      self.set_bnfp(name, bnfp, tensor_type)
      if (NndctOption.nndct_stat.value > 1):
        print('---- quant %s tensor: %s with bw = %d and fp = %g' % (
            tensor_type, name, bnfp[0], bnfp[1]))

      # get 2^bit_width and 2^fracpos
      bnfp = self.get_bnfp(name, True, tensor_type)

      if (NndctOption.nndct_stat.value > 2):
        quant_data = nndct_quant.QuantizeData(name, res.cpu().detach().numpy())

      # do quantization for parameter or activation
      res = py_nndct.nn.NndctFixNeuron(res,
                                       res,
                                       maxamp = [bnfp[0], bnfp[1]],
                                       method = mth)

        
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

  def do_quantize(self, blob, name, node=None, tensor_type='input'):
    # forward quant graph but not quantize parameter and activation
    if NndctOption.nndct_quant_off.value:
      return blob
    
    blob_save = None
    if isinstance(blob.values, torch.Tensor):
      blob_save = blob
      blob = blob.values.data
    
    if blob.dtype != torch.float32 and blob.dtype != torch.double:
      NndctScreenLogger().warning_once(f'The tensor type of  {node.name} is {str(blob.dtype)}. Only support float32/double quantization.')
      return blob_save if blob_save is not None else blob
    
    quant_device = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_DEVICE)
    if blob.device.type != quant_device.type:
      raise TypeError("Device of quantizer is {}, device of model and data should match device of quantizer".format(quant_device.type))

    if (NndctOption.nndct_stat.value > 2):
      quant_data = nndct_quant.QuantizeData(name, blob.cpu().detach().numpy())
    # quantize the tensor
    bnfp = self.get_bnfp(name, True, tensor_type)
    if (NndctOption.nndct_stat.value > 1):
        print('---- quant %s tensor: %s with 1/step = %g' % (
            tensor_type, name, bnfp[1]))
    # hardware cut method
    mth = 4 if self.lstm else 2
    if tensor_type == 'param':
      mth = 3

    res = py_nndct.nn.NndctFixNeuron(blob,
                                     blob,
                                     maxamp = [bnfp[0], bnfp[1]],
                                     method = mth)

    if (NndctOption.nndct_stat.value > 2):
      global global_snr_inv
      quant_efficiency, sqnr = quant_data.quant_efficiency(blob.cpu().detach().numpy(), 8)
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
    if tensor_type == 'param':
      self.update_param_to_nndct(node, name, res.cpu().detach().numpy())
    
    if blob_save is not None:
      blob_save.values.data = blob
      blob = blob_save
      res = blob_save

    return res

  def _check_calibration_completion(self):
    ret = True
    # Check node output tensors
    for node in self.Nndctgraph.nodes:
      if self.configer.is_node_quantizable(node, self.lstm) and node.in_quant_part:
        qout = self.configer.quant_output(node.name).name
        bnfp = self.get_bnfp(qout, False)
        if bnfp[1] is None:
          if node.op.type not in [NNDCT_OP.SIGMOID, NNDCT_OP.TANH]:
            NndctScreenLogger().warning(f'Node ouptut tensor is not quantized: {node.name} type: {node.op.type}')
            ret = False
    # Check node input tensors 
    for item in self._QuantInfo['input']:
      bnfp = self._QuantInfo['input'][item]
      if bnfp[1] is None:
        NndctScreenLogger().warning(f'Input tensor is not quantized: {item}')
        ret = False
    # Check node parameters
    for item in self._QuantInfo['param']:
      bnfp = self._QuantInfo['param'][item]
      if bnfp[1] is None:
        NndctScreenLogger().warning(f'Parameter tensor is not quantized: {item}')
        ret = False

    return ret

  def _align_lstm_fix_pos_with_cell_output(self):
    if self.lstm:
      if self.rnn_front_end:
        for node in self.Nndctgraph.nodes:
          nextNode = self.configer.Nndctgraph.get_node_by_idx(idx=node.idx + 1)
          # find the last node of every cell
          if (nextNode is not None and nextNode.name.split('::')[-1] == 'input_0'
              or node.idx == len(list(self.configer.Nndctgraph.nodes)) - 1):
            #print('----find output h node %s' % node.name)
            for nid in range(node.idx-1, -1, -1):
              prevNode = self.configer.Nndctgraph.get_node_by_idx(idx=nid)
              # fragpos of sigmoid and tanh keep 15
              if prevNode.op.type in [NNDCT_OP.SIGMOID, NNDCT_OP.TANH]:
                bnfp = self.get_bnfp(prevNode.name, False)
                bnfp[1] = 15
                self.set_bnfp(prevNode.name, bnfp)
                continue
              if prevNode.name.split('::')[-1] == 'input_0':
                break;
              bnfp = self.get_bnfp(node.name, False)
              #print('----set %s fix pos to %d' % (prevNode.name, bnfp[1]))
              target_name = self.configer.quant_output(prevNode.name).name
              # multiple output node is not grouped up with children,
              # for example, strided_slice is not grouped up
              if target_name in self.quant_config['output'].keys():
                self.set_bnfp(target_name, bnfp)
      else: # to handle dlrm
        # fragpos of sigmoid and tanh keep 15
        for node in self.Nndctgraph.nodes:
          if node.op.type in [NNDCT_OP.SIGMOID, NNDCT_OP.TANH]:
            bnfp = self.get_bnfp(node.name, False)
            bnfp[1] = 15
            self.set_bnfp(node.name, bnfp)

        # find the last non-sigmoid/tanh node
        node_num = len(list(self.configer.Nndctgraph.nodes))
        for idx in range(node_num-1, -1, -1):
          last_node = self.configer.Nndctgraph.get_node_by_idx(idx)
          if last_node.op.type not in [NNDCT_OP.SIGMOID, NNDCT_OP.TANH]:
              break
        last_bnfp = self.get_bnfp(last_node.name, False)

        # back propagate last_node's bnfp to previous nodes which are non-sigmoid/tanh
        for idx in range(last_node.idx-1, -1, -1):
          node = self.configer.Nndctgraph.get_node_by_idx(idx)
          if node.op.type in [NNDCT_OP.SIGMOID, NNDCT_OP.TANH]:
            continue
          elif node.name.split('::')[-1] == 'input_0':
            break;
          else:
            self.set_bnfp(node.name, last_bnfp)

  def _node_relative_fix_pos_check(self):
    for node in self.Nndctgraph.nodes:
      # shift_bias and shift_cut check
      if (node.in_quant_part and self.configer.is_conv_like(node)):
        # get i/w/b/o fix pos
        conv_quant_output = self.configer.quant_output(node.name).name
        fix_pos_i = self.get_bnfp(node.in_nodes[0], False, 'output')
        fix_pos_o = self.get_bnfp(conv_quant_output, False, 'output')
        fix_pos_w = self.get_bnfp(node.op.param['weights'].name, False, 'param')
        if fix_pos_i[-1] == None:
          NndctScreenLogger().warning("Unsupported op type of input node: {}".format(node.in_nodes[0]))
          break
	# handle shift_cut
        shift_cut = fix_pos_w[-1] + fix_pos_i[-1] - fix_pos_o[-1]
        shift_cut_min = 0
        shift_cut_max = 16
        if shift_cut < shift_cut_min:
          NndctScreenLogger().warning("output {} value is too small, so adjust the fix position from {} to {}"
                  .format(conv_quant_output, fix_pos_o[-1], fix_pos_o[-1] + shift_cut - shift_cut_min))
          fix_pos_o[-1] = fix_pos_o[-1] + shift_cut - shift_cut_min;
          self.set_bnfp(conv_quant_output, fix_pos_o)
        elif shift_cut > shift_cut_max:
          NndctScreenLogger().warning("weight {} value is too small, so adjust the fix position from {} to {}"
                  .format(node.name, fix_pos_w[-1], fix_pos_w[-1] - shift_cut + shift_cut_max))
          fix_pos_w[-1] = fix_pos_w[-1] - shift_cut + shift_cut_max;
          self.set_bnfp(node.op.param['weights'].name, fix_pos_w, tensor_type = "param")

        # handle shift_bias
        if node.module.bias is not None:
          fix_pos_b = self.get_bnfp(node.op.param['bias'].name, False, 'param')
          shift_bias = fix_pos_w[-1] + fix_pos_i[-1] - fix_pos_b[-1]
          shift_bias_min = min(0, -(24-(8+shift_cut)))
          shift_bias_max = 16
          if shift_bias < shift_bias_min:
            NndctScreenLogger().warning("bias {} value is too small, so adjust the fix position from {} to {}"
                    .format(node.op.param['bias'].name, fix_pos_b[-1], fix_pos_b[-1] + shift_bias - shift_bias_min))
            fix_pos_b[-1] = fix_pos_b[-1] + shift_bias - shift_bias_min;
            self.set_bnfp(node.op.param['bias'].name, fix_pos_b, tensor_type = "param")
          elif shift_bias > shift_bias_max:
            NndctScreenLogger().warning("weight {} value is too small, so adjust the fix position from {} to {}"
                    .format(node.op.param['weights'].name, fix_pos_w[-1], fix_pos_w[-1] - shift_bias + shift_bias_max))
            fix_pos_w[-1] = fix_pos_w[-1] - shift_bias + shift_bias_max;
            self.set_bnfp(node.op.param['weights'].name, fix_pos_w, tensor_type = "param")

  def organize_quant_pos(self):
    # Transfer inplace operation fragpos forward,
    # to replace configerComannder in future
    if NndctOption.nndct_quant_off.value:
      return

    # check quantization calibration is performed completely
    if not self._check_calibration_completion():
      NndctScreenLogger().warning("Quantization is not performed completely, check if model inference function is called!!!")
      return

    # align lstm fix pos with cell output
    self._align_lstm_fix_pos_with_cell_output()

    for node in self.Nndctgraph.nodes:
      # align linear OP bias fix pos with output for lstm
      if self.lstm:
        if node.op.type == NNDCT_OP.DENSE:
          if len(node.op.params.values()) > 1:
            params = [v.name for v in node.op.params.values()]
            bnfp = self.get_bnfp(node.name, False)
            self.set_bnfp(params[1], bnfp, 'param')
      # Strided_slice branches fix pos alignment
      if (node.in_quant_part and
          node.op.type == NNDCT_OP.STRIDED_SLICE):
        bnfp = None
        src_name = self.configer.quant_output(node.in_nodes[0]).name
        if self.need_quantize_tensor(src_name):
          bnfp = self.get_bnfp(src_name, False)
          self.quant_config['output'][node.name] = [bnfp[0], bnfp[1]]
          #print('Strided_Slice fix pos setting node: {} qout: {} pos: {}'.format(
          #    node.name, src_name, bnfp[1]))

      # zero padding output fix pos align with input
      if (node.in_quant_part and
          node.op.type == NNDCT_OP.PAD):
        in_name = self.configer.quant_output(node.in_nodes[0]).name
        out_name = self.configer.quant_output(node.name).name
        bnfp = self.get_bnfp(in_name, False)
        #print('---- set zero padding output %s fix pos to %s : %d' % (out_name, in_name, bnfp[1]))
        self.set_bnfp(out_name, bnfp)
        
      if (node.in_quant_part and node.op.type == NNDCT_OP.RESIZE and node.node_config('mode') == "'nearest'"):
        in_name = self.configer.quant_output(node.in_nodes[0]).name
        out_node = self.configer.quant_output(node.name)
        out_name = out_node.name
        if out_node.op.type != NNDCT_OP.CONCAT:
          bnfp = self.get_bnfp(in_name, False)
        #print('---- set nearest upsampling output %s fix pos to %s : %d' % (out_name, in_name, bnfp[1]))
          self.set_bnfp(out_name, bnfp)
    
      # limit hardsigmoid output fix pos to >= 7
      if (node.in_quant_part and
          node.op.type == NNDCT_OP.HSIGMOID):
        out_name = self.configer.quant_output(node.name).name
        bnfp = self.get_bnfp(out_name, False)
        #print('checking {}: {}'.format(out_name, bnfp))
        if bnfp[1] < 7:
          bnfp[1] = 7
          self.set_bnfp(out_name, bnfp)

    # shift_bias and shift_cut check and adjustment
    self._node_relative_fix_pos_check()

  def export_quant_config(self, export_file=None, adjust_pos=True):
    if NndctOption.nndct_param_corr.value > 0:
      if self.quant_mode == 1:
        # gather bias correction, how to get nn module objec?
        for node in self.Nndctgraph.nodes:
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

    # export quant steps
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
          
        tensor.from_ndarray(param_data)
        tensor_util.convert_parameter_tensor_format(
            tensor, key_names.FrameworkType.TORCH,
            key_names.FrameworkType.NNDCT)

  def export_param(self):
    if self.quant_mode == 1:
      NndctScreenLogger().info(f"=>Exporting quant model parameters.({self.param_file})")
      torch.save(self.quant_model.state_dict(), self.param_file)

  def load_param(self):
    if self.quant_mode == 2:
      NndctScreenLogger().info(f"=>Loading quant model parameters.({self.param_file})")
      self.quant_model.load_state_dict(torch.load(self.param_file))

  @property
  def quant_model(self):
    return self._quant_model

  @quant_model.setter
  def quant_model(self, quant_model):
    self._quant_model = quant_model

