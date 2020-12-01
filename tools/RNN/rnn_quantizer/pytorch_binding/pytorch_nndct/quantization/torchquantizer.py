

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
    # forward quant graph but not quantize parameter and activation
    if NndctOption.nndct_quant_off.value:
      return res
    
    res_save = res
    if isinstance(res.values, torch.Tensor):
      res = res.values
    
    quant_device = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_DEVICE)
    if res.device.type != quant_device.type:
      raise TypeError("Device of quantizer is {}, device of model and data should match device of quantizer".format(quant_device.type))
    
    # hardware cut method
    mth = 4 if self.lstm else 2
    if tensor_type == 'param':
      mth = 3

    range = 5
    # set fix pos scanning range to 1 for some type of tensors
    if ((node.op.type in [NNDCT_OP.INPUT, NNDCT_OP.QUANT_STUB])
        or (self.lstm and tensor_type == 'input')) :
      range = 1

    # get fixed position
    bnfp = self.get_bnfp(name, False, tensor_type)
    #if(res.device == torch.device("cpu")):
    if(quant_device.type == "cpu"):
      Tbuffer = torch.empty_like(res).to(torch.device("cpu"))
      Tfixpos = torch.tensor([1], dtype=torch.get_default_dtype()).to(torch.device("cpu"))
    else:
      Tbuffer = torch.empty_like(res).cuda()
      Tfixpos = torch.tensor([1], dtype=torch.get_default_dtype()).cuda()

    # activation always calculate fix pos
    # calcualte fix pos if it is None
    # always calculate fis pos in finetune mode
    if tensor_type != 'param' or bnfp[1] is None or self.quant_mode == 3:
      py_nndct.nn.NndctDiffsFixPos(
          Tinput = res,
          Tbuffer = Tbuffer,
          Tfixpos = Tfixpos,
          bit_width = bnfp[0],
          range = range,
          method = mth)
      bnfp[1] = (int)(Tfixpos.item())
      # record fix pos of activation
      if tensor_type != 'param':
        self.fp_history[tensor_type][name].append(bnfp[1])
        data = np.array(self.fp_history[tensor_type][name])
        bnfp[1] = stats.mode(data)[0][0]
        bnfp[1] = bnfp[1].astype(np.int32).tolist()
      self.set_bnfp(name, bnfp, tensor_type)
      #print('---- quant %s with bw = %d and fp = %g' % (name, bnfp[0], bnfp[1]))

      # get 2^bit_width and 2^fracpos
      bnfp = self.get_bnfp(name, True, tensor_type)

      if (NndctOption.nndct_quant_opt.value and 
          NndctOption.nndct_logging_level.value > 0):
        #if tensor_type == "param":
        quant_data = nndct_quant.QuantizeData(name, res.cpu().detach().numpy())
  
      #print('---- quant %s with bw = %d and 1/step = %g' % (name, bnfp[0], bnfp[1]))
      # do quantization for parameter or activation
      res = py_nndct.nn.NndctFixNeuron(res, 
                                       res, 
                                       maxamp = [bnfp[0], bnfp[1]], 
                                       method = mth)
      
      if (NndctOption.nndct_quant_opt.value and 
          NndctOption.nndct_logging_level.value > 0):
        #if tensor_type == "param":
        global global_snr_inv
        quant_efficiency, sqnr = quant_data.quant_efficiency(res.cpu().detach().numpy(), 8) 
        global_snr_inv += 1 / sqnr
        print(f"quant_efficiency={quant_efficiency}, {quant_data._name}\n")
        #print(f"quant_efficiency={quant_efficiency}, global_snr_inv={globacl_snr_inv} {quant_data._name}\n")
    res = res_save
    return res

  def do_quantize(self, blob, name, node=None, tensor_type='input'):
    # forward quant graph but not quantize parameter and activation
    if NndctOption.nndct_quant_off.value:
      return blob
    
    blob_save = blob
    if isinstance(blob.values, torch.Tensor):
      blob = blob.values
    
    quant_device = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_DEVICE)
    if blob.device.type != quant_device.type:
      raise TypeError("Device of quantizer is {}, device of model and data should match device of quantizer".format(quant_device.type))

    if (NndctOption.nndct_quant_opt.value and 
        NndctOption.nndct_logging_level.value > 0):
      quant_data = nndct_quant.QuantizeData(name, blob.cpu().detach().numpy())
    # quantize the tensor
    bnfp = self.get_bnfp(name, True, tensor_type)
    #print('---- quant %s with 1/step = %g' % (name, bnfp[1]))
    # hardware cut method
    mth = 4 if self.lstm else 2
    if tensor_type == 'param':
      mth = 3
    
    res = py_nndct.nn.NndctFixNeuron(blob, 
                                     blob, 
                                     maxamp = [bnfp[0], bnfp[1]], 
                                     method = mth)

    if (NndctOption.nndct_quant_opt.value and 
        NndctOption.nndct_logging_level.value > 0):
      global global_snr_inv
      quant_efficiency, sqnr = quant_data.quant_efficiency(blob.cpu().detach().numpy(), 8) 
      global_snr_inv += 1 / sqnr
      print(f"quant_efficiency={quant_efficiency}, global_snr_inv={global_snr_inv} {quant_data._name}\n")

    # update param to nndct graph
    if tensor_type == 'param':
      self.update_param_to_nndct(node, name, res.cpu().detach().numpy())

    blob = blob_save
    res = blob_save
    
    return res

  def organize_quant_pos(self):
    # Transfer inplace operation fragpos forward,
    # to replace configerComannder in future
    if NndctOption.nndct_quant_off.value:
      return 

    # align lstm fix pos with cell output
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

    for node in self.Nndctgraph.nodes:
      # align linear OP bias fix pos with output for lstm
      if self.lstm:
        if node.op.type == NNDCT_OP.DENSE:
          if len(node.op.params.values()) > 1: 
            params = [v.name for v in node.op.params.values()]
            bnfp = self.get_bnfp(node.name, False)
            self.set_bnfp(params[1], bnfp, 'param')
      # node with multiple ouputs 
      if (node.in_quant_part and 
          node.op.type == NNDCT_OP.STRIDED_SLICE): 
        bnfp = None
        src_name = self.configer.quant_output(node.name).name
        bnfp = self.get_bnfp(src_name, False)
        self.quant_config['output'][node.name] = [bnfp[0], bnfp[1]]
      # concat inputs fix pos align with output
      if (node.in_quant_part and 
          node.op.type == NNDCT_OP.CONCAT):
        out_name = self.configer.quant_output(node.name).name
        bnfp = self.get_bnfp(out_name, False)
        for i in node.in_nodes:
          in_name = self.configer.quant_output(i).name
          #print('---- set concat input %s fix pos to %d' % (in_name, bnfp[1]))
          self.set_bnfp(in_name, bnfp)
      # zero padding output fix pos align with input
      if (node.in_quant_part and 
          node.op.type == NNDCT_OP.PAD):
        in_name = self.configer.quant_output(node.in_nodes[0]).name
        out_name = self.configer.quant_output(node.name).name
        bnfp = self.get_bnfp(in_name, False)
        #print('---- set zero padding output %s fix pos to %s : %d' % (out_name, in_name, bnfp[1]))
        self.set_bnfp(out_name, bnfp)
      # do shift_bias and shift_cut check
      if (node.in_quant_part and node.op.type == NNDCT_OP.CONV2D):
        # get i/w/b/o fix pos
        conv_quant_output = self.configer.quant_output(node.name).name
        fix_pos_i = self.get_bnfp(node.in_nodes[0], False, 'output')
        fix_pos_o = self.get_bnfp(conv_quant_output, False, 'output')
        fix_pos_w = self.get_bnfp(node.op.param['weights'].name, False, 'param')
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
                    .format(node.op.param['weights'].name, fix_pos_w[-1], fix_pos_w[-1] - shift_bias - shift_bias_max))
            fix_pos_w[-1] = fix_pos_w[-1] - shift_bias - shift_bias_max;
            self.set_bnfp(node.op.param['weights'].name, fix_pos_w, tensor_type = "param")

  def export_quant_config(self, export_file=None):
    if NndctOption.nndct_param_corr.value > 0:
      if self.quant_mode == 1:
        # gather bias correction, how to get nn module objec?
        for node in self.Nndctgraph.nodes:
          if node.op.type in [NNDCT_OP.CONV2D,
                              NNDCT_OP.CONVTRANSPOSE2D,
                              NNDCT_OP.DEPTHWISE_CONV2D,
                              NNDCT_OP.DENSE]:
            if node.module.bias is not None:
              self.bias_corr[node.name] = node.module.bias_corr()

        # export bias correction
        torch.save(self.bias_corr, self.bias_corr_file)

    # export quant steps 
    file_name = export_file or self.export_file
    if isinstance(file_name, str):
        NndctScreenLogger().info(f"=>Exporting quant config.({file_name})")
        self.organize_quant_pos()
        with open(file_name, 'w') as f:
          f.write(nndct_utils.to_jsonstr(self.quant_config))

  def update_param_to_nndct(self, node, param_name, param_data):
    for param_type, tensor in node.op.params.items():
      if tensor.name == param_name:
        if node.op.type == NNDCT_OP.CONVTRANSPOSE2D:
          if param_type == node.op.ParamName.WEIGHTS:
            param_data = np.copy(param_data).transpose(1, 0, 2, 3)
            
        if node.op.type == NNDCT_OP.DEPTHWISE_CONV2D and param_type == node.op.ParamName.WEIGHTS:
            in_channels = node.node_config("in_channels")
            out_channels = node.node_config("out_channels")
            kernel_size = node.node_config("kernel_size")
            channel_mutiplier = int(out_channels / in_channels)
            param_data = param_data.reshape((channel_mutiplier, in_channels, *kernel_size))
            
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
    
