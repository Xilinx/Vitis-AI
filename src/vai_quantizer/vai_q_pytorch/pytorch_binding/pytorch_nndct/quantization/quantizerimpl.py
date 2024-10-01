
import copy
import numpy as np
from abc import ABC, abstractmethod
import torch
from nndct_shared.base import key_names, NNDCT_OP
from nndct_shared.utils import tensor_util, NndctOption, NndctScreenLogger, QWarning, QError
from pytorch_nndct.nn import fake_quantize_per_tensor

def has_inf_nan():
  return hasattr(torch, "isinf") and hasattr(torch, "isnan")

def is_valid_tensor_for_quantizer(tensor):
  if has_inf_nan():
    inf = torch.isinf(tensor.detach())
    nan = torch.isnan(tensor.detach())
    if inf.sum() > 0 or nan.sum() > 0:
      return False
  return True

def convert_datatype_to_pytorch_type(datatype):
  return {
  'bfloat16': torch.bfloat16,
  'float16': torch.float16,
  'float32': torch.float
  }.get(datatype, datatype)
  
def convert_datatype_to_index(datatype):
  return {
  'bfloat16': 1,
  'float16': 2,
  'float32': 3
  }.get(datatype, datatype)

class QuantizerImpl(ABC):
  
  def __init__(self):
    self.exporting = False
    self.inplace = True
    
  def calibrate(self, res, name, node=None, tensor_type='input', idx=0, method=None, datatype='int'):
    if datatype == 'int':
      return self.calibrate_int(res, name, node, tensor_type, idx, method)
    else:
      return self.calibrate_float(res, name, node, tensor_type, idx, datatype)

  def quantize(self, blob, name, node=None, tensor_type='input', idx=0, method=None, datatype='int'):
    if datatype == 'int':
      return self.quantize_int(blob, name, node, tensor_type, idx, method)
    else:
      return self.quantize_float(blob, name, node, tensor_type, idx, datatype)
  
  @abstractmethod
  def calibrate_int(self, res, name, node=None, tensor_type='input', idx=0, method=None):
    pass
  
  @abstractmethod
  def quantize_int(self, blob, name, node=None, tensor_type='input', idx=0, method=None):
    pass
  
  def calibrate_float(self, res, name, node=None, tensor_type='input', idx=0, datatype='bfloat16'):
    # Don't support non_tensor quantization
    if (not isinstance(res, torch.Tensor)) and ((not hasattr(res, "values")) or (not isinstance(res.values, torch.Tensor))):
      return res

    res_save = None
    if isinstance(res.values, torch.Tensor):
      if NndctOption.nndct_quant_off.value or res.values.data.numel() == 0:
        if self.inplace:
          return res
        else:
          return copy.deepcopy(res)
      res_save = res
      res = res.values.data
    else:
      if NndctOption.nndct_quant_off.value or res.data.numel() == 0:
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
    
    if convert_datatype_to_pytorch_type(datatype) == res.dtype:
      return res_save if res_save is not None else res
    
    origin_dtype = res.dtype
    convert_dtype = convert_datatype_to_pytorch_type(datatype)
    if convert_dtype != origin_dtype:
      res = res.to(convert_dtype)
      res = res.to(origin_dtype)
    
    if res_save is not None:
      res_save.values.data = res
      res = res_save

    return res
  
  def quantize_float(self, blob, name, node=None, tensor_type='input', idx=0, datatype='float16'):
    # Don't support non_tensor quantization
    if (not isinstance(blob, torch.Tensor)) and ((not hasattr(blob, "values")) or (not isinstance(blob.values, torch.Tensor))):
      return blob
    
    blob_save = None
    if isinstance(blob.values, torch.Tensor):
      if NndctOption.nndct_quant_off.value:
        if self.inplace:
          return blob
        else:
          return copy.deepcopy(blob)
      blob_save = blob
      blob = blob.values.data
    else:
      if NndctOption.nndct_quant_off.value:
        if self.inplace:
          return blob
        else:
          return blob.clone().detach()
    
    if blob.dtype != torch.float32 and blob.dtype != torch.double and blob.dtype != torch.float16:
      NndctScreenLogger().warning2user_once(QWarning.TENSOR_TYPE_NOT_QUANTIZABLE, f'The tensor type of {node.name} is {str(blob.dtype)}. Only support float32/double/float16 quantization.')
      return blob_save if blob_save is not None else blob

    if not is_valid_tensor_for_quantizer(blob):
      NndctScreenLogger().warning2user_once(QWarning.TENSOR_VALUE_INVALID, f'The tensor type of {node.name} have "inf" or "nan" value. The quantization is ignored. Please check it.')
      return blob_save if blob_save is not None else blob
    
    # dtype_index = convert_datatype_to_index(datatype)
    # blob = fake_quantize_per_tensor(blob, 1.0, 0, 0, 0, 0, self.inplace, dtype_index)
    
    if convert_datatype_to_pytorch_type(datatype) == blob.dtype:
      return blob_save if blob_save is not None else blob
    
    origin_dtype = blob.dtype
    convert_dtype = convert_datatype_to_pytorch_type(datatype)
    if convert_dtype != origin_dtype:
      blob = blob.to(convert_dtype)
      blob = blob.to(origin_dtype)
    
    # update param to nndct graph
    if tensor_type == 'param' and not self.exporting:
      self.update_param_to_nndct(node, name, blob.cpu().detach().numpy())
    
    if blob_save is not None:
      blob_save.values.data = blob
      blob = blob_save

    return blob
  
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
        if node.op.type != NNDCT_OP.LAYER_NORM:
          tensor_util.convert_parameter_tensor_format(
              tensor, key_names.FrameworkType.TORCH,
              key_names.FrameworkType.NNDCT)
        
        NndctScreenLogger().check2user(QError.SHAPE_MISMATCH, f"The shape of data '{tensor.shape}' must be consistent with that of original data '{origin_shape}' for {tensor.name}", origin_shape == tensor.shape)
