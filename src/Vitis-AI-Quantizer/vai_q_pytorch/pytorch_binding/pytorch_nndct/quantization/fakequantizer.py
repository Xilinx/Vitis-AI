

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
import torch
import pathlib
#from scipy import stats
from collections import namedtuple
from nndct_shared.quantization import NewBaseQuantizer
from nndct_shared.base import GLOBAL_MAP, NNDCT_KEYS, NNDCT_OP
from nndct_shared.base.key_names import FrameworkType
from nndct_shared.utils import NndctOption, tensor_util, NndctScreenLogger
import nndct_shared.utils as nndct_utils

from pytorch_nndct.quantization import NndctCGQstrategy, TensorRTCGQStrategy
from pytorch_nndct.quantization import PerChannelQuantAlgo

class FakeQuantizer(NewBaseQuantizer):
  
  def __init__(self, 
               quant_mode: int,
               output_dir: str,
               quant_config,
               is_lstm = False):
    super().__init__(quant_mode,
                     output_dir,
                     quant_config,
                     is_lstm)
    if NndctOption.nndct_param_corr.value > 0:
      if self.quant_mode == 2:
        path = pathlib.Path(self.bias_corr_file)
        if not (path.exists() and path.is_file()):
          NndctScreenLogger().error(f"Bias correction result file does not exist. \
Please check calibration with bias correction is done or not.")
          exit(2)
        self.bias_corr = torch.load(self.bias_corr_file)
        self._bias_corr_loaded = True

    self.exporting = False
    self.inplace = True
    self.serial = True
    #self._fast_finetuned = False
    self.output_dir = output_dir
    
    if NndctOption.nndct_tensorrt_strategy.value:
      self.quant_strategy = TensorRTCGQStrategy(quant_config)
    else:
      self.quant_strategy = NndctCGQstrategy(quant_config)
  
  def get_model_type(self):
    return FrameworkType.TORCH
  
  def reset_status_for_exporting(self):
    self.exporting = True
    self.inplace = False
    for mod in self._quant_model.modules():
      if hasattr(mod, "param_quantized"):
        setattr(mod, "param_quantized", False)

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
      if self.inplace:
        return res
      else:
        return res.clone().detach()
    
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
    q_config = self.get_quant_config(name, False, tensor_type)
    
    # turn off quantization if bit width is more than 32
    # if q_config[0] >= 32:
    #   if self.inplace:
    #     return res
    #   else:
    #     return res.clone().detach()
    
    q_algorithm = self.get_quant_algo(name, tensor_type)
    # get quant algorithm
    #if tensor_type != 'param' or q_config[1] is None or q_config[2] is None:
    if q_algorithm.calib_or_not(tensor_type):
      #q_algorithm = self.get_quant_algo(name, tensor_type)
      q_algorithm.calibrate(res)
      
      if q_algorithm.statistic_local:
        # quant_tensor = q_algorithm.fake_quantize(res, self.inplace)
        # if self.inplace:
        #   res.data = quant_tensor.data.clone()
        # else:
        #   res = quant_tensor
        
        q_config[1] = q_algorithm.scale
        q_config[2] = q_algorithm.zero_point
        q_config[3] = q_algorithm.float_max
        if tensor_type != 'param':
          self.config_history[tensor_type][name].append([q_config[1], q_config[2], q_config[3]])
          data = np.array(self.config_history[tensor_type][name]).transpose(1,0)
          q_config[1], q_config[2], q_config[3] = q_algorithm.act_scale_stats(data)
          #q_algorithm.scale, q_algorithm.zero_point, q_algorithm.float_max = q_config[1], q_config[2], q_config[3]
        self.set_quant_config(name, q_config, tensor_type)
        
        quant_tensor = q_algorithm.fake_quantize(res, self.inplace)
        if self.inplace:
          res.data = quant_tensor.data.clone()
        else:
          res = quant_tensor
    
    if res_save is not None:
      res_save.values.data = res
      res = res_save
    return res

  def do_quantize(self, blob, name, node=None, tensor_type='input'):
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
      
    if blob.dtype != torch.float32 and blob.dtype != torch.double:
      NndctScreenLogger().warning_once(f'The tensor type of  {node.name} is {str(blob.dtype)}. Only support float32/double quantization.')
      return blob_save if blob_save is not None else blob

    quant_device = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_DEVICE)
    if blob.device.type != quant_device.type:
      raise TypeError("Device of quantizer is {}, device of model and data should match device of quantizer".format(quant_device.type))

    q_config = self.get_quant_config(name, True, tensor_type)
    # turn off quantization ifbit width is more than 32
    # if q_config[0] >= 32:
    #   if self.inplace:
    #     return blob
    #   else:
    #     return blob.clone().detach()
    
    self._set_serial_or_not(node, tensor_type)
    
    q_algorithm = self.get_quant_algo(name, tensor_type)
    if q_algorithm is None:
      if node.op.type == NNDCT_OP.STRIDED_SLICE:
        if self.inplace:
          return blob
        else:
          return blob.clone().detach()
      else:
        raise ValueError("Quantization algorithm should not be none")
      
    q_algorithm.bitwidth = q_config[0]
    q_algorithm.zero_point = (q_config[2].to(blob.device) 
                              if isinstance(q_config[2], torch.Tensor) 
                              else torch.tensor(q_config[2], device=blob.device))
    #q_algorithm.zero_point = torch.tensor(q_config[2], device=blob.device)
    if isinstance(q_config[1], float):
      q_algorithm.scale = q_config[1]
    else:
      q_algorithm.scale = (q_config[1].to(blob.device) 
                           if isinstance(q_config[1], torch.Tensor) 
                           else torch.tensor(q_config[1], device=blob.device))
    
    if isinstance(q_config[3], float):
      q_algorithm.float_max = q_config[3]
    else:
      q_algorithm.float_max = (q_config[3].to(blob.device) 
                               if isinstance(q_config[3], torch.Tensor) 
                               else torch.tensor(q_config[3], device=blob.device))
    
    quant_tensor = q_algorithm.fake_quantize(blob, (self.inplace and self.serial))
    
    if self.inplace and self.serial:
      blob.data = quant_tensor.data.clone()
      output = blob
    else:
      output = quant_tensor

    # update param to nndct graph
    if tensor_type == 'param' and not self.exporting:
      self.update_param_to_nndct(node, name, blob.cpu().detach().numpy())

    if blob_save is not None:
      blob_save.values.data = output
      output = blob_save

    return output
  
  def _set_serial_or_not(self, node, tensor_type):
    self.serial = True
    if tensor_type == "param":
      return
    for name in node.in_nodes:
      in_node = self.Nndctgraph.node(name)
      if len(in_node.out_nodes) > 1:
        self.serial = False
        return

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
            tensor, FrameworkType.TORCH, FrameworkType.NNDCT)
        
        NndctScreenLogger().check(f"The shape of data '{tensor.shape}' must be consistent with that of original data \
          '{origin_shape}' for {tensor.name}", origin_shape == tensor.shape)

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
        self.bias_corrected = True

    # export quant steps
    file_name = export_file or self.export_file
    if isinstance(file_name, str):
        NndctScreenLogger().info(f"=>Exporting quant config.({file_name})")
        self.calib_global_param()
        if adjust_pos and (not NndctOption.nndct_tensorrt_strategy.value):
          self.organize_quant_pos()
        with open(file_name, 'w') as f:
          f.write(nndct_utils.to_jsonstr(self.quant_config))

  def calib_global_param(self):
    quant_device = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_DEVICE)
    for tensor_type, algo_dict in self._QuantAlgo.items():
      for name, algo in algo_dict.items():
        if not algo.statistic_local:
          q_config = self.get_quant_config(name, False, tensor_type)
          # if q_config[0] < 32:
          #   algo.calib_global_statis(quant_device)
          #   q_config[1], q_config[2], q_config[3] = algo.scale, algo.zero_point, algo.float_max
          algo.calib_global_statis(quant_device)
          q_config[1], q_config[2], q_config[3] = algo.scale, algo.zero_point, algo.float_max
          self.set_quant_config(name, q_config, tensor_type)
          #self.set_quant_algo(name, algo, tensor_type)
        
  def organize_quant_pos(self):
    # Transfer inplace operation fragpos forward,
    # to replace configerComannder in future
    if NndctOption.nndct_quant_off.value:
      return

    # check quantization calibration is performed completely
    if not self._check_calibration_completion():
      NndctScreenLogger().warning("Quantization is not performed completely, check if model inference function is called!!!")
      #return

    for node in self.Nndctgraph.nodes:
      # Strided_slice branches fix pos alignment
      if (node.in_quant_part and
          node.op.type == NNDCT_OP.STRIDED_SLICE):
        q_config = None
        src_name = self.configer.quant_output(node.in_nodes[0]).name
        if self.need_quantize_tensor(src_name):
          q_config = self.get_quant_config(src_name, False)
          self.quant_config['output'][node.name] = [q_config[0], q_config[1], q_config[2], q_config[3]]
          #print('Strided_Slice fix pos setting node: {} qout: {} pos: {}'.format(
          #    node.name, src_name, q_config[1]))

      # zero padding output fix pos align with input
      if (node.in_quant_part and
          node.op.type == NNDCT_OP.PAD):
        in_name = self.configer.quant_output(node.in_nodes[0]).name
        out_name = self.configer.quant_output(node.name).name
        q_config = self.get_quant_config(in_name, False)
        #print('---- set zero padding output %s fix pos to %s : %d' % (out_name, in_name, q_config[1]))
        self.set_quant_config(out_name, q_config)
        
      if (node.in_quant_part and node.op.type == NNDCT_OP.RESIZE and node.node_config('mode') == "'nearest'"):
        in_name = self.configer.quant_output(node.in_nodes[0]).name
        out_node = self.configer.quant_output(node.name)
        out_name = out_node.name
        if out_node.op.type != NNDCT_OP.CONCAT:
          q_config = self.get_quant_config(in_name, False)
        #print('---- set nearest upsampling output %s fix pos to %s : %d' % (out_name, in_name, q_config[1]))
          self.set_quant_config(out_name, q_config)
    
      # TODO:complete next
      # limit hardsigmoid output fix pos to >= 7
      # if (node.in_quant_part and
      #     node.op.type == NNDCT_OP.HSIGMOID):
      #   out_name = self.configer.quant_output(node.name).name
      #   q_config = self.get_quant_config(out_name, False)
      #   #print('checking {}: {}'.format(out_name, q_config))
      #   #fp = np.log2(1.0/q_config[1])
      #   if fp < 7:
      #     fp = 7
      #     bnfp[1] = 1.0 / (2**fp)
      #     self.set_quant_config(out_name, bnfp)
  

  def _check_calibration_completion(self):
    ret = True
    # Check node output tensors
    for node in self.Nndctgraph.nodes:
      if self.configer.is_node_quantizable(node, False) and node.in_quant_part:
        qout = self.configer.quant_output(node.name).name
        q_config = self.get_quant_config(qout, False)
        if q_config[1] is None:
          if node.op.type not in [NNDCT_OP.SIGMOID, NNDCT_OP.TANH]:
            NndctScreenLogger().warning(f'Node ouptut tensor is not quantized: {node.name} type: {node.op.type}')
            ret = False
    # Check node input tensors 
    for item in self._QuantInfo['input']:
      q_config = self._QuantInfo['input'][item]
      if q_config[1] is None:
        NndctScreenLogger().warning(f'Input tensor is not quantized: {item}')
        ret = False
    # Check node parameters
    for item in self._QuantInfo['param']:
      q_config = self._QuantInfo['param'][item]
      if q_config[1] is None:
        NndctScreenLogger().warning(f'Parameter tensor is not quantized: {item}')
        ret = False

    return ret
  
  def export_param(self):
    if self.quant_mode == 1:
      NndctScreenLogger().info(f"=>Exporting quant model parameters.({self.param_file})")
      torch.save(self.quant_model.state_dict(), self.param_file)
      self.fast_finetuned = True

  def load_param(self):
    if self.quant_mode == 2:
      NndctScreenLogger().info(f"=>Loading quant model parameters.({self.param_file})")
      path = pathlib.Path(self.param_file)
      if not (path.exists() and path.is_file()) or not self.fast_finetuned:
        NndctScreenLogger().error(f"Fast finetuned parameter file does not exist. \
                                  Please check calibration with fast finetune is done or not.")
        exit(2)
      self.quant_model.load_state_dict(torch.load(self.param_file))
      self._finetuned_para_loaded = True
      
  def features_check(self):
    if self.fast_finetuned and not self._finetuned_para_loaded:
      NndctScreenLogger().warning(f'Fast finetuned parameters are not loaded. \
                                  Call load_ft_param to load them.')
    if self.bias_corrected and not self._bias_corr_loaded:
      NndctScreenLogger().warning(f'Bias correction file is not loaded. Set \
                                  command line option \"--nndct_param_corr\" to load it.')
      
  def contain_channel_quantize(self):
    contain_channel_quantize_or_not = False
    for key, algo in self.quant_algo['param'].items():
      if isinstance(algo, PerChannelQuantAlgo):
        contain_channel_quantize_or_not = True
        break
    return contain_channel_quantize_or_not
