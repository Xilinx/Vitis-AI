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
from nndct_shared.utils import NndctOption, tensor_util, NndctScreenLogger

import nndct_shared.utils as nndct_utils
import nndct_shared.quantization as nndct_quant

class TORCHQuantizer(BaseQuantizer):

  def __init__(self, 
               quant_mode: int, 
               output_dir: str,
               bitwidth_w: int,
               bitwidth_a: int):
    super().__init__(quant_mode, 
                     output_dir,
                     bitwidth_w, 
                     bitwidth_a)
    self._quant_model = None

  def get_model_type(self):
    return FrameworkType.TORCH

  def do_scan(self,
              res,
              max_tensor,
              min_tensor,
              name,
              node=None,
              tensor_type='input'):
    # forward quant graph but not quantize parameter and activation
    if NndctOption.nndct_quant_off.value:
      return res

    # hardware cut method
    mth = 4 if self.lstm else 2
    if tensor_type == 'param':
      mth = 3

    # get fixed position
    bnfp = self.get_bnfp(name, False)
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
          range = 5,
          method = mth)
      bnfp[1] = (int)(Tfixpos.item())
      # record fix pos of activation
      if tensor_type != 'param':
        self.maxmins[name].append(bnfp[1])
        data = np.array(self.maxmins[name])
        bnfp[1] = stats.mode(data)[0][0]
        bnfp[1] = bnfp[1].astype(np.int32).tolist()
      self.set_bnfp(name, bnfp)

      # get 2^bit_width and 2^fracpos
      bnfp = self.get_bnfp(name)
      #print('---- quant %s with 1/step = %g' % (name, bnfp[1]))
      # do quantization for parameter or activation
      res = py_nndct.nn.NndctFixNeuron(res, 
                                       res, 
                                       maxamp = [bnfp[0], bnfp[1]], 
                                       method = mth)
    return res

  def do_quantize(self, blob, name, node=None, tensor_type='input'):
    # forward quant graph but not quantize parameter and activation
    if NndctOption.nndct_quant_off.value:
      return blob

    bnfp = self.get_bnfp(name)
    #print('---- quant %s with 1/step = %g' % (name, bnfp[1]))
    # hardware cut method
    mth = 4 if self.lstm else 2
    if tensor_type == 'param':
      mth = 3
    res = py_nndct.nn.NndctFixNeuron(blob, 
                                     blob, 
                                     maxamp = [bnfp[0], bnfp[1]], 
                                     method = mth)
    # update param to nndct graph
    if tensor_type == 'param':
      self.update_param_to_nndct(node, name, res.cpu().detach().numpy())

    return res


  # def add_module_info(self, model):
  #   self._quant_model = model
  #   if model.__class__.__name__ in ['DataParallel', 'DistributedDataParallel']:
  #     model = model.module
  #   for name, module in model.named_children():
  #     # module_type = module._get_name().split('_')[-1]
  #     #print("---- Initialize module info, %s : %s\n" % (module_type, module._get_name()))
  #     # if module_type != module._get_name():
  #     #   module.name = name
  #     self.set_module(module)

  # def set_module(self, module):
  #   if self.quant_mode > 0:
  #     idx = int(module.name.split('_')[-1])
  #     node = self.configer.get_Nndctnode(idx=idx)
  #     params = self.configer.quant_node_params(node)
  #     if node.op.type in [NNDCT_OP.CONV2D, NNDCT_OP.DENSE]:
  #       # handle by ModuleHooker
  #       # module.node,\  
  #       _, module.valid_inputs,\
  #       module.valid_output = nndct_quant.get_flows_and_info(
  #                               self.quant_mode,
  #                               self,
  #                               node_name = node.name,
  #                               params = params,
  #                               inputs = node.in_nodes)
  #     else:
  #       # handle by ModuleHooker
  #       # module.node,\
  #       module.quant_info,\
  #       module.valid_inputs,\
  #       module.valid_output = nndct_quant.get_flows_and_info(
  #                               self.quant_mode,
  #                               self,
  #                               node_name = node.name,
  #                               params = params,
  #                               inputs = node.in_nodes)

  #     # module.params_name = node.op.params
      
  #     # handle by ModuleHooker
  #     # module.params_name = [v.name for v in node.op.params.values()]


  def organize_quant_pos(self):
    # Transfer inplace operation fragpos forward,
    # to replace configerComannder in future
    if NndctOption.nndct_quant_off.value:
      return 

    # align lstm fix pos with cell output
    if self.lstm:
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
              continue
            if prevNode.name.split('::')[-1] == 'input_0':
              break;
            bnfp = self.get_bnfp(node.name, False)
            #print('----set %s fix pos to %d' % (prevNode.name, bnfp[1]))
            self.set_bnfp(prevNode.name, bnfp)

    for node in self.Nndctgraph.nodes:
      # align linear OP bias fix pos with output for lstm
      if self.lstm:
        if node.op.type == NNDCT_OP.DENSE:
          if len(node.op.params.values()) > 1: 
            params = [v.name for v in node.op.params.values()]
            bnfp = self.get_bnfp(node.name, False)
            self.set_bnfp(params[1], bnfp)
      # node with multiple ouputs 
      if node.op.type == NNDCT_OP.STRIDED_SLICE:
        bnfp = None
        in_node = self.configer.get_Nndctnode(node.in_nodes[0])
        for o in in_node.out_nodes:
          if o in self.quant_config['blobs']:
            bnfp = self.get_bnfp(o, False)
            break
        for o in in_node.out_nodes:
          if o not in self.quant_config['blobs']:
            self.quant_config['blobs'][o] = [bnfp[0], bnfp[1]]
      # inplace operation
      if node.op.type in [NNDCT_OP.RELU, NNDCT_OP.CLAMP, NNDCT_OP.RELU6]:
        out_name = self.configer.quant_groups[node.name][-1]
        if self.quant_config['blobs'][out_name][1] is None:
          input_name = self.configer.quant_inputs(
              node, node.in_nodes[0], validate=False)[0]
          print('----set %s fix pos to %d' %
                (out_name, self.quant_config['blobs'][input_name][1]))
          self.quant_config['blobs'][out_name][1] = self.quant_config['blobs'][
              input_name][1]
      # concat inputs fix pos align with output
      if node.op.type == NNDCT_OP.CONCAT:
        out_name = self.configer.quant_groups[node.name][-1]
        bnfp = self.get_bnfp(out_name, False)
        for i in node.in_nodes:
          in_name = self.configer.quant_groups[i][-1]
          #print('---- set concat input %s fix pos to %d' % (in_name, bnfp[1]))
          self.set_bnfp(in_name, bnfp)


  def export_quant_config(self, export_file=None):
    if self.quant_mode in [1, 3]:
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

  @property
  def quant_model(self):
    return self._quant_model
 
  @quant_model.setter
  def quant_model(self, quant_model):
    self._quant_model = quant_model
    
