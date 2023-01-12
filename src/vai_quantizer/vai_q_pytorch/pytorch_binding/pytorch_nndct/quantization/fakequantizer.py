

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
import copy
import numpy as np
import torch
import pathlib
#from scipy import stats
from collections import namedtuple
from pytorch_nndct.version import __version__
from nndct_shared.quantization import NewBaseQuantizer
from nndct_shared.base import GLOBAL_MAP, NNDCT_KEYS, NNDCT_OP
from nndct_shared.base.key_names import FrameworkType
from nndct_shared.utils import NndctOption, tensor_util, NndctScreenLogger, QError, QWarning, QNote
import nndct_shared.utils as nndct_utils

from pytorch_nndct.quantization import NndctCGQstrategy, TensorRTCGQStrategy
from pytorch_nndct.quantization import PerChannelQuantAlgo
from pytorch_nndct.parse.parse_utils import ValueDeviceInfo
from pytorch_nndct.qproc.utils import quant_model_inferenced


def has_inf_nan():
  return hasattr(torch, "isinf") and hasattr(torch, "isnan")

def is_valid_tensor_for_quantizer(tensor):
  if has_inf_nan():
    inf = torch.isinf(tensor)
    nan = torch.isnan(tensor)
    if inf.sum() > 0 or nan.sum() > 0:
        return False
  return True

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
          NndctScreenLogger().error2user(QError.BIAS_CORRECTION, f"Bias correction result file does not exist. \
Please check calibration with bias correction is done or not.")
          exit(2)
        self.bias_corr = torch.load(self.bias_corr_file)
        self._bias_corr_loaded = True

    self.exporting = False
    self.inplace = True
    self.serial = True
    self.inferenced = False
    #self._fast_finetuned = False
    self.output_dir = output_dir
    self._scripts = []
    
    if NndctOption.nndct_tensorrt_strategy.value:
      self.quant_strategy = TensorRTCGQStrategy(quant_config, is_lstm)
    else:
      self.quant_strategy = NndctCGQstrategy(quant_config, is_lstm)
      
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

    # quant_device = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_DEVICE)
    # if res.device.type != quant_device.type:
    #   raise TypeError("Device of quantizer is {}, device of model and data should match device of quantizer".format(quant_device.type))

    # get fixed position
    q_config = self.get_quant_config(name, False, tensor_type, idx)
    if q_config is None:
      return res_save if res_save is not None else res
    
    q_algorithm = self.get_quant_algo(name, tensor_type, idx)
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
        self.set_quant_config(name, q_config, tensor_type, idx)
        
        quant_tensor = q_algorithm.fake_quantize(res, self.inplace)
        if self.inplace:
          res.data = quant_tensor.data.clone()
        else:
          res = quant_tensor
    
    if res_save is not None:
      res_save.values.data = res
      res = res_save
    return res

  def do_quantize(self, blob, name, node=None, tensor_type='input', idx=0, method=None):
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
      NndctScreenLogger().warning2user_once(QWarning.TENSOR_VALUE_INVALID, f'The tensor type of {node.name} have "inf" or "nan" value.The quantization is ignored. Please check it.')
      return blob_save if blob_save is not None else blob

    q_config = self.get_quant_config(name, True, tensor_type, idx)
    if q_config is None:
      return blob_save if blob_save is not None else blob
    
    self._set_serial_or_not(node, tensor_type)
    q_algorithm = self.get_quant_algo(name, tensor_type, idx)
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
      
    if tensor_type == 'param':
      self.graph.param_tensor(name).device = ValueDeviceInfo('cpu') if output.device == torch.device('cpu') \
        else ValueDeviceInfo('cuda')
    elif tensor_type == 'output':
      node.out_tensors[idx].device = ValueDeviceInfo('cpu') if output.device == torch.device('cpu') \
        else ValueDeviceInfo('cuda')
    else:
      node.in_tensors[idx].device = ValueDeviceInfo('cpu') if output.device == torch.device('cpu') \
        else ValueDeviceInfo('cuda')

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
        
        NndctScreenLogger().check2user(QError.SHAPE_MISMATCH, f"The shape of data '{tensor.shape}' must be consistent with that of original data \
          '{origin_shape}' for {tensor.name}", origin_shape == tensor.shape)

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
        self.calib_global_param()
        if adjust_pos and (not NndctOption.nndct_tensorrt_strategy.value):
          self.organize_quant_pos()
        with open(file_name, 'w') as f:
          f.write(nndct_utils.to_jsonstr(self.quant_config))

  def calib_global_param(self):
    #quant_device = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_DEVICE)
    for tensor_type, algo_dict in self._QuantAlgo.items():
      for name, algo in algo_dict.items():
        for idx in range(self.get_quant_len(name, tensor_type)):
          algo = algo[idx]
          if not algo.statistic_local:
            q_config = self.get_quant_config(name, False, tensor_type, idx)
            algo.calib_global_statis()
            q_config[1], q_config[2], q_config[3] = algo.scale, algo.zero_point, algo.float_max
            self.set_quant_config(name, q_config, tensor_type, idx)
          #self.set_quant_algo(name, algo, tensor_type)
        
  def organize_quant_pos(self):
    # Transfer inplace operation fragpos forward,
    # to replace configerComannder in future
    if NndctOption.nndct_quant_off.value:
      return

    # check quantization calibration is performed completely
    if not self._check_calibration_completion():
      NndctScreenLogger().warning2user(QWarning.TENSOR_NOT_QUANTIZED, f"Some tensors are not quantized, please check their particularity.")

    for node in self.Nndctgraph.all_nodes():
      # align linear OP bias fix pos with output for lstm
      if self.lstm:
        if node.op.type == NNDCT_OP.DENSE:
          if len(node.op.params.values()) > 1:
            params = [v.name for v in node.op.params.values()]
            q_config = self.get_quant_config(node.name, False)
            self.set_quant_config(params[1], q_config, 'param')
      
      # Strided_slice branches fix pos alignment
      if (node.in_quant_part and
          node.op.type == NNDCT_OP.STRIDED_SLICE):
        q_config = None
        src_name = self.configer.quant_output(node.in_nodes[0]).name
        if self.need_quantize_tensor(src_name):
          q_config = self.get_quant_config(src_name, False)
          self.quant_config['output'][node.name] = [[q_config[0], q_config[1], q_config[2], q_config[3]]]

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
          
      # change concat input nodes fix point to be the same as concat output node
      if (node.in_quant_part and
          node.op.type == NNDCT_OP.CONCAT and
          NndctOption.nndct_change_concat_input_fix.value):
        q_config = self.get_quant_config(node, False)
        for in_node in node.in_nodes:
          self.set_quant_config(in_node, q_config)

      # change add input nodes fix point to be the same as its output
      if (node.in_quant_part and
          node.op.type == NNDCT_OP.ADD and 
          NndctOption.nndct_change_add_input_fix.value):
        q_config = self.get_quant_config(node, False)
        for in_node in node.in_nodes:
          self.set_quant_config(in_node, q_config)

      # change pooling input nodes fix point to be the same as its output node
      if (node.in_quant_part and
          node.op.type in [NNDCT_OP.MAX_POOL, 
                           NNDCT_OP.MAX_POOL1D,
                           NNDCT_OP.AVG_POOL, 
                           NNDCT_OP.ADAPTIVEAVGPOOL2D] 
          and NndctOption.nndct_change_pool_input_fix.value):
        q_config = self.get_quant_config(node, False)
        for in_node in node.in_nodes:
          self.set_quant_config(in_node, q_config)
    
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
  
  def _check_calibration_completion_for_target(self):
    def _check(tensor_type):
      ret = True
      for item in self._QuantInfo[tensor_type]:
        for idx in range(self.get_quant_len(item, tensor_type)):
          qconfig = self._QuantInfo[tensor_type][item][idx]
          if qconfig is None:
            continue
          if qconfig[1] is None:
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
        q_config = self.get_quant_config(qout, False)
        if q_config[1] is None:
          if node.op.type not in [NNDCT_OP.SIGMOID, NNDCT_OP.TANH]:
            NndctScreenLogger().warning2user(QWarning.TENSOR_NOT_QUANTIZED, f'Node ouptut tensor is not quantized: {node.name} type: {node.op.type}')
            ret = False
    # Check node input tensors 
    for item in self.quant_config['input']:
      for idx in range(self.get_quant_len(item, 'input')):
        q_config = self.get_quant_config(item, False, 'input', idx)
        if q_config[1] is None:
          NndctScreenLogger().warning2user(QWarning.TENSOR_NOT_QUANTIZED, f'Input tensor is not quantized: {item}')
          ret = False
    # Check node parameters
    for item in self.quant_config['param']:
      for idx in range(self.get_quant_len(item, 'param')):
        q_config = self.get_quant_config(item, False, 'param', idx)
        if q_config[1] is None:
          NndctScreenLogger().warning2user(QWarning.TENSOR_NOT_QUANTIZED, f'Parameter tensor is not quantized: {item}. \
If this parameter is not a parameter embedded in torch operation, like weights of CONV, \
this kind of parameter will not be quantized. Please embed the parameter into CONV/Linear/BN.')
          ret = False

    return ret
  
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
    contain_channel_quantize_or_not = False
    for key, algo in self.quant_algo['param'].items():
      if isinstance(algo, PerChannelQuantAlgo):
        contain_channel_quantize_or_not = True
        break
    return contain_channel_quantize_or_not
