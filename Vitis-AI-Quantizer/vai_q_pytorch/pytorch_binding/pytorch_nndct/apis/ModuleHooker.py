import copy
from functools import partial

import numpy as np
import torch

import nndct_shared.quantization as nndct_quant
import pytorch_nndct.parse.torch_op_def as torch_op_def
import pytorch_nndct.utils.tensor_utils as py_tensor_utils
from nndct_shared.base import NNDCT_OP
from nndct_shared.utils import NndctScreenLogger


def _is_module_hooked(module):
  for m in module.children():
    if hasattr(m, "node"):
      return True
  return False


class ModuleHooker(object):
  _parameter_map = {
      torch_op_def.TorchConv2d.ParamName.WEIGHTS: "weight",
      torch_op_def.TorchConv2d.ParamName.BIAS: "bias",
      torch_op_def.TorchLinear.ParamName.WEIGHTS: "weight",
      torch_op_def.TorchLinear.ParamName.BIAS: "bias",
      torch_op_def.TorchBatchNorm.ParamName.GAMMA: "weight",
      torch_op_def.TorchBatchNorm.ParamName.BETA: "bias",
      torch_op_def.TorchBatchNorm.ParamName.MOVING_MEAN: "running_mean",
      torch_op_def.TorchBatchNorm.ParamName.MOVING_VAR: "running_var"
  }

  @staticmethod
  def name_modules(module):
    for name, op in module.named_modules():
      op.name = name

  @staticmethod
  def add_outputs(module, record_once):
    for op in module.modules():
      op.__outputs__ = None if record_once else []
      op.__enable_record__ = False

  @staticmethod
  def apply_to_children(module, func):
    for child in module.children():
      child.apply(func)
  
  @staticmethod
  def turn_on_record_outputs(module):
    for op in module.modules():
      op.__enable_record__ = True
  
  @staticmethod
  def turn_off_record_outputs(module):
    for op in module.modules():
      op.__enable_record__ = False
  
  @staticmethod
  def clear_record_outputs(module):
    for op in module.modules():
      if op.__outputs__ is not None:
        if isinstance(op.__outputs__, list):
          op.__outputs__.clear()
        else:
          op.__outputs__ = None  
        
  @classmethod
  def register_output_hook(cls, module, record_once=True):

    def _record_outputs(op, inputs, outputs):
      if op.__enable_record__ is True:
        if isinstance(outputs, torch.Tensor):
          output_data = outputs.cpu().detach()
        else:
          output_data = copy.deepcopy(outputs)
          
        if isinstance(op.__outputs__, list):
          op.__outputs__.append(output_data)
        else:
          op.__outputs__ = output_data

    def _output_hooker(op, record_func):
      op.register_forward_hook(record_func)

    if not hasattr(module, "__outputs__"):
      cls.add_outputs(module, record_once)

    partial_func = partial(_output_hooker, record_func=_record_outputs)
    cls.apply_to_children(module, partial_func)

  @classmethod
  def hook_module_with_node(cls, module, graph):

    def _add_node_on_module(op):
      idx = int(op.name.split('_')[-1])
      node = graph.get_node_by_idx(idx)
      op.node = node
      op.params_name = [v.name for v in node.op.params.values()]

    if not hasattr(module, "name"):
      cls.name_modules(module)

    # partial_func = partial(_add_node_on_module, graph=graph)
    cls.apply_to_children(module, _add_node_on_module)

  @classmethod
  def hook_module_with_quantizer(cls, module, quantizer):
    
    def _add_quant_info_on_module(op):
      node = op.node
      params = quantizer.configer.quant_node_params(node)
      if node.op.type in [NNDCT_OP.CONV2D, NNDCT_OP.DENSE]:
        _, op.valid_inputs, op.valid_output = nndct_quant.get_flows_and_info(quantizer.quant_mode, 
                                                                             quantizer, 
                                                                             node_name=node.name, 
                                                                             params=params, 
                                                                             inputs=node.in_nodes)
      else:
        op.quant_info, op.valid_inputs, op.valid_output = nndct_quant.get_flows_and_info(quantizer.quant_mode, 
                                                                                         quantizer,
                                                                                         node_name=node.name, 
                                                                                         params=params,
                                                                                         inputs=node.in_nodes)
                                                                                         
    if not _is_module_hooked(module):
      cls.hook_module_with_node(module, quantizer)
    
    cls.apply_to_children(module, _add_quant_info_on_module)
                                                                                                                                                                             
  @classmethod
  def update_parameters(cls, module, graph, graph2module):

    def _graph2module(op, param_map):
      node = getattr(op, "node", None)
      for param_name, tensor in node.op.params.items():
        py_tensor_utils.param_to_torch_format(tensor)

        data = np.copy(tensor.data)
        if node.op.type == NNDCT_OP.CONVTRANSPOSE2D and param_name == torch_op_def.TorchConv2d.ParamName.WEIGHTS:
          data = data.transpose(1, 0, 2, 3)
          data = np.ascontiguousarray(data)
        
        if node.op.type == NNDCT_OP.DEPTHWISE_CONV2D and param_name == torch_op_def.TorchConv2d.ParamName.WEIGHTS:
          out_channels = node.node_config("out_channels")
          kernel_size = node.node_config("kernel_size")
          data = data.reshape((out_channels, 1, *kernel_size)) 
          
        torch_tensor = torch.from_numpy(data)
    
        if hasattr(op, param_map[param_name]):
          if isinstance(getattr(op, param_map[param_name]), torch.Tensor):
            torch_tensor = torch_tensor.to(
                getattr(op, param_map[param_name]))
          else:
            torch_tensor = torch_tensor.to(
                getattr(op, param_map[param_name]).data)
            
          if param_map[param_name] in op._buffers:
            op._buffers[param_map[param_name]] = torch_tensor
          else:
            op._parameters[param_map[param_name]] = torch.nn.Parameter(torch_tensor)
        else:
          op.register_parameter(param_map[param_name],
                                torch.nn.Parameter(torch_tensor))

        py_tensor_utils.param_to_nndct_format(tensor)

    # No one will call it and will be removed later.
    def _module2graph(op, param_map):
      node = getattr(op, "node", None)
      for param_name, tensor in node.op.params.items():
        if hasattr(op, param_map[param_name]):
          if param_name in [torch_op_def.TorchBatchNorm.ParamName.MOVING_MEAN, 
                            torch_op_def.TorchBatchNorm.ParamName.MOVING_VAR]:
            torch_tensor = getattr(op, param_map[param_name])
          else:
            torch_tensor = getattr(op, param_map[param_name]).data
            
          torch_tensor_data = np.copy(torch_tensor.detach().cpu().numpy())

          if node.op.type == NNDCT_OP.CONVTRANSPOSE2D and param_name == torch_op_def.TorchConv2d.ParamName.WEIGHTS:
            torch_tensor_data = torch_tensor_data.transpose(1, 0, 2, 3)
            torch_tensor_data = np.ascontiguousarray(torch_tensor_data)
            
          if node.op.type == NNDCT_OP.DEPTHWISE_CONV2D and param_name == torch_op_def.TorchConv2d.ParamName.WEIGHTS:
            in_channels = node.node_config("in_channels")
            out_channels = node.node_config("out_channels")
            kernel_size = node.node_config("kernel_size")
            channel_mutiplier = int(out_channels / in_channels)
            torch_tensor_data = torch_tensor_data.reshape((channel_mutiplier, in_channels, *kernel_size))
            
          tensor.from_ndarray(torch_tensor_data)
          py_tensor_utils.param_to_nndct_format(tensor)

    if not _is_module_hooked(module):
      cls.hook_module_with_node(module, graph)

    partial_func = partial(
        _graph2module if graph2module else _module2graph,
        param_map=cls._parameter_map)
    cls.apply_to_children(module, partial_func)
    
  @staticmethod
  def _get_output_data(outptus, outptus_name):
    try:
        # output_data = outptus.cpu().detach().numpy()
        output_data = outptus.numpy()
    except AttributeError:
      NndctScreenLogger().warning(f"{outptus_name} is not tensor. It's data is ignored.")
      return None
    else:
      return output_data
    
  @classmethod
  def update_blobs_once(cls, module, graph, time_step=None):
      
    def _update_node_outputs(op):
      node = op.node
      one_step_outputs = op.__outputs__[
          time_step] if time_step is not None else op.__outputs__
      if len(node.out_tensors) > 1:
        for idx, tensor in enumerate(node.out_tensors):
          output_data = cls._get_output_data(one_step_outputs[idx], tensor.name)
          if output_data is not None:
            tensor.from_ndarray(output_data)
            py_tensor_utils.blob_to_nndct_format(tensor)
      else:
        tensor = node.out_tensors[0]
        output_data = cls._get_output_data(one_step_outputs, tensor.name)
        if output_data is not None:
          tensor.from_ndarray(output_data)
          py_tensor_utils.blob_to_nndct_format(tensor)

    if not _is_module_hooked(module):
      cls.hook_module_with_node(module, graph)

    cls.apply_to_children(module, _update_node_outputs)
