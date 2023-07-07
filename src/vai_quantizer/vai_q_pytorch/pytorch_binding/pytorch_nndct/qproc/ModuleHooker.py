
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

import copy
import functools
from functools import partial
import os
import numpy as np
import torch

import nndct_shared.utils as nndct_utils
from nndct_shared.quantization import quantize_data2int
import pytorch_nndct.parse.torch_op_def as torch_op_def
import pytorch_nndct.utils.tensor_util as py_tensor_util
from pytorch_nndct.utils.module_util import to_device, collect_input_devices, get_flattened_input_args
from nndct_shared.base import NNDCT_OP, GLOBAL_MAP, NNDCT_KEYS
from nndct_shared.nndct_graph import Graph

from nndct_shared.utils import NndctScreenLogger, NndctOption, permute_data, permute_axes, QError, QWarning, QNote
from pytorch_nndct.utils.module_util import get_module_name
from typing import Sequence


class TimeStepData(list):
  pass


def _is_module_hooked(module):
  for m in module.children():
    if hasattr(m, "node") and m.node:
      return True
  return False


class ModuleHooker(object):
  _parameter_map = {
      torch_op_def.TorchConv1d.ParamName.WEIGHTS: "weight",
      torch_op_def.TorchConv1d.ParamName.BIAS: "bias",
      torch_op_def.TorchConv2d.ParamName.WEIGHTS: "weight",
      torch_op_def.TorchConv2d.ParamName.BIAS: "bias",
      torch_op_def.TorchConv3d.ParamName.WEIGHTS: "weight",
      torch_op_def.TorchConv3d.ParamName.BIAS: "bias",
      torch_op_def.TorchLinear.ParamName.WEIGHTS: "weight",
      torch_op_def.TorchLinear.ParamName.BIAS: "bias",
      torch_op_def.TorchBatchNorm.ParamName.GAMMA: "weight",
      torch_op_def.TorchBatchNorm.ParamName.BETA: "bias",
      torch_op_def.TorchBatchNorm.ParamName.MOVING_MEAN: "running_mean",
      torch_op_def.TorchBatchNorm.ParamName.MOVING_VAR: "running_var",
      torch_op_def.TorchLayerNorm.ParamName.GAMMA: "weight",
      torch_op_def.TorchLayerNorm.ParamName.BETA: "bias",
      torch_op_def.TorchEmbedding.ParamName.WEIGHT: "weight",
      torch_op_def.TorchPReLU.ParamName.WEIGHT: "weight",
      torch_op_def.TorchInstanceNorm.ParamName.GAMMA: "weight",
      torch_op_def.TorchInstanceNorm.ParamName.BETA: "bias",
      torch_op_def.TorchGroupNorm.ParamName.GAMMA: "weight",
      torch_op_def.TorchGroupNorm.ParamName.BETA: "bias"

  }


  @staticmethod
  def name_modules(module):
    for name, op in module.named_modules():
      op.name = name

  @staticmethod
  def add_outputs(module, record_once):
    for op in module.modules():
      op.__outputs__ = None if record_once else TimeStepData()
      op.__enable_record__ = False

  @staticmethod
  def add_output_intime(module):
    for op in module.modules():
      op.__enable_output_intime_ = False
      op.__called_times_ = 0
      
  @staticmethod
  def add_input_dump_time(module):
    module.__enable_input_dump_ = False
    module.__input_called_times_ = 0

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
      if hasattr(op, "__outputs__") and op.__outputs__ is not None:
        if isinstance(op.__outputs__, TimeStepData):
          op.__outputs__.clear()
        else:
          op.__outputs__ = None
  
  @staticmethod
  def turn_on_output_intime(module):
    for op in module.modules():
      op.__enable_output_intime_ = True

  @staticmethod
  def turn_off_output_intime(module):
    for op in module.modules():
      op.__enable_output_intime_ = False

  @staticmethod
  def clear_op_called_times(module):
    for op in module.modules():
      op.__called_times_ = 0
      
  @staticmethod
  def turn_on_input_dump(module):
    module.__enable_input_dump_ = True

  @staticmethod
  def turn_off_input_dump(module):
    module.__enable_input_dump_ = False

  @staticmethod
  def clear_input_called_times(module):
    module.__input_called_times_ = 0

  @classmethod
  def register_state_dict_hook(cls, module):
    def _resume_state_dict_key(op, destination, prefix, local_meta_data):
      if op.node:
        # only handle parameters
        prefixes = [".".join(param_tensor.name.split(".")[:-1]) for param_tensor in op.node.op.params.values()]
        if len(prefixes) == 0:
          raise RuntimeError("The num of module params is not equal with graph")
        new_prefix = prefixes[0].split("::")[-1] + "."
        for name, data in op._parameters.items():
          if prefix + name in destination:
            destination[new_prefix + name] = data
            del destination[prefix + name]

    def _state_dict_hooker(op, hooker):
      if len(op._parameters):
        op._register_state_dict_hook(hooker)

    partial_func = partial(_state_dict_hooker, hooker=_resume_state_dict_key)
    cls.apply_to_children(module, partial_func)

  @classmethod
  def register_output_hook(cls, module, record_once=True):

    def _record_outputs(op, inputs, outputs):
      if op.__enable_record__ is True and hasattr(op, "node"):
        def get_outputs_value(outputs):
          if isinstance(outputs, torch.Tensor):
            return outputs.cpu().detach().clone()
          elif hasattr(outputs, "values") and hasattr(outputs, "indices"):
            return (outputs.values.cpu().detach().clone(), outputs.indices.cpu().detach().clone())
          elif isinstance(outputs, Sequence):
            return type(outputs)([get_outputs_value(op) for op in outputs])
          else:
            return outputs

        if isinstance(op.__outputs__, TimeStepData) or (op.__outputs__ is None) or \
        (NndctOption.nndct_record_slow_mode.value is True):

          output_data = get_outputs_value(outputs)

          if isinstance(op.__outputs__, TimeStepData):
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
  def register_input_dump_hook(cls, module):
    # dump input float data
    def dump_input_data(module, inputs):
      input_tensor_name = module._input_tensors_name
      inputs_list = []
      
      def inputs_flatten(tensors, tensors_flatten):
        if isinstance(tensors, torch.Tensor):
          tensors_flatten.append(tensors.cpu().detach().numpy())
        elif isinstance(tensors, (list, tuple)):
          for tensor in tensors:
            inputs_flatten(tensor, tensors_flatten)
        else:
          raise TypeError(f"Type of input_args should be tuple/list/torch.Tensor.")
      
      inputs_flatten(inputs, inputs_list)
      if len(input_tensor_name) != len(inputs_list):
        raise TypeError(f"Length of inputs should be the same as length of inputs name.")
      
      quantizer = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANTIZER)
      dir_name = os.path.join(quantizer.output_dir, 'deploy_check_data')
      input_path = os.path.join(dir_name, 'input')
      
      if module.__input_called_times_== 0:
        nndct_utils.create_work_dir(dir_name)
        nndct_utils.create_work_dir(os.path.join(dir_name, 'input'))
        
      for name, tensor in zip(input_tensor_name, inputs_list):
        name = name.replace('/', '_')
        current_input_path = os.path.join(input_path, name)
        if module.__input_called_times_== 0:
          nndct_utils.create_work_dir(current_input_path)
          shape_file_name = os.path.join(current_input_path, 'shape.txt')
          np.array(tensor.shape).tofile(shape_file_name, sep=' ')
        
        file_name = os.path.join(current_input_path, str(module.__input_called_times_))
        txt_file = '.'.join([file_name, 'txt'])
        tensor.flatten().tofile(txt_file, sep="\n", format="%g")
      
      module.__input_called_times_ += 1
    
    if not hasattr(module, "__input_called_times_"):
      cls.add_input_dump_time(module)
      
    module.register_forward_pre_hook(dump_input_data)

  @classmethod
  def register_output_intime_hook(cls, module):
    # only for jit script mode in rnnt quantization
    def _dump_intime(op, inputs, outputs):
      if op.__enable_output_intime_ is True and hasattr(op, "node") and op.node.in_quant_part:

        def get_outputs_value(outputs):
          if isinstance(outputs, torch.Tensor):
            return outputs.cpu().detach().numpy()
          elif hasattr(outputs, "values") and hasattr(outputs, "indices"):
            return (outputs.values.cpu().detach().numpy(), outputs.indices.cpu().detach().numpy())
          elif isinstance(outputs, Sequence):
            return type(outputs)([get_outputs_value(op) for op in outputs])
          else:
            return None

        dir_name = os.path.join(op.quantizer.output_dir, 'deploy_check_data')
        params_path = os.path.join(dir_name, 'params')
        output_path = os.path.join(dir_name, 'output')
        if op.__called_times_== 0:
          nndct_utils.create_work_dir(dir_name)
          nndct_utils.create_work_dir(os.path.join(dir_name, 'params'))
          nndct_utils.create_work_dir(os.path.join(dir_name, 'output'))
          shape_file_name = os.path.join(dir_name, 'shape.txt')
          
          # dump params
          for k, para in op.quantizer.configer.quant_node_params(op.node).items():
            bit_width = 16
            fix_point = None
            if para.name in op.quantizer.quant_config['param'].keys():
              #bit_width, fix_point = op.quantizer.quant_config['param'][para.name]
              bit_width, fix_point = op.quantizer.get_quant_config(para.name, False, 'param')
            data = para.data
            para_name = para.name.replace('/', '_')
            file_name = os.path.join(params_path, para_name)
            shape_file_name = '_'.join([file_name, 'shape.txt'])
            np.array(data.shape).tofile(shape_file_name, sep=' ')
            bin_file = '.'.join([file_name+'_fix', 'bin'])
            txt_file = '.'.join([file_name+'_fix', 'txt'])
            if fix_point is not None:
              quantize_data2int(data.flatten(), bit_width, fix_point).tofile(txt_file, sep="\n", format="%g")
              quantize_data2int(data.flatten(), bit_width, fix_point).tofile(bin_file)

        # dump output
        def dump_output_data(op, output_data, index=None):
          if output_data is not None:
            bit_width = 16
            fix_point = None
            end = op.quantizer.configer.quant_output(op.node.name).name
            #bit_width, fix_point = op.quantizer.quant_config['output'][end]
            node_name = op.node.name.replace('/', '_')
            if index is None:
              bit_width, fix_point = op.quantizer.get_quant_config(end, False, 'output')
              current_output_path = os.path.join(output_path, node_name + "_fix")
            else:
              bit_width, fix_point = op.quantizer.get_quant_config(end, False, 'param', index)
              index_str = str(index)
              current_output_path = os.path.join(output_path, node_name + "_fix_i" + index_str)
            if op.__called_times_== 0:
              nndct_utils.create_work_dir(current_output_path)
              shape_file_name = os.path.join(current_output_path, 'shape.txt')
              np.array(output_data.shape).tofile(shape_file_name, sep=' ')
            file_name = os.path.join(current_output_path, str(op.__called_times_))
            bin_file = '.'.join([file_name, 'bin'])
            txt_file = '.'.join([file_name, 'txt'])
            if output_data.ndim > 0:
              data = output_data[:]
            else:
              data = output_data
            if fix_point is not None:
              quantize_data2int(data.flatten(), bit_width, fix_point).tofile(txt_file, sep="\n", format="%g")
              quantize_data2int(data.flatten(), bit_width, fix_point).tofile(bin_file)

        def dump_output(op, output_data):
          index = 0
          if isinstance(output_data, Sequence):
            for output in output_data:
              if isinstance(output, Sequence):
                for i in output:
                  dump_output_data(op, i, index)
                  index += 1
              else:
                dump_output_data(op, output, index)
                index += 1
          else:
            dump_output_data(op, output_data, None)

        #if op.quantizer.configer.is_node_quantizable(op.node, True):
        if op.quantizer.have_quant_or_not(op.node.name):
          output_data = get_outputs_value(outputs)
          dump_output(op, output_data)
      op.__called_times_ += 1

    def _output_hooker(op, output_func):
      op.register_forward_hook(output_func)
    
    if not hasattr(module, "__called_times_"):
      cls.add_output_intime(module)

    partial_func = partial(_output_hooker, output_func=_dump_intime)
    cls.apply_to_children(module, partial_func)

  @classmethod
  def detach_node_from_module(cls, module):
    def _del_node(op):
      if hasattr(op, "node"):
        op.attached_node_name = op.node.name
        del op.node
  
      if hasattr(op, "params_name"):
        del op.params_name

    cls.apply_to_children(module, _del_node)

  @classmethod
  def hook_module_with_node(cls, module, graph):

    def _add_node_on_module(op):
      if not hasattr(op, "attached_node_name"):
        idx = int(op.name.split('_')[-1])
        node = graph.get_node_by_idx(idx)
      else:
        node = graph.node(op.attached_node_name)
        
      if node is not None:
        node.module = op
        op.node = node
        op.params_name = [v.name for v in node.op.params.values()]

    if not hasattr(module, "name"):
      cls.name_modules(module)

    if _is_module_hooked(module):
      cls.detach_node_from_module(module)

    # partial_func = partial(_add_node_on_module, graph=graph)
    cls.apply_to_children(module, _add_node_on_module)

  # @classmethod
  # def hook_module_with_quantizer(cls, module, quantizer):
  #   if not _is_module_hooked(module):
  #     cls.hook_module_with_node(module, quantizer.Nndctgraph)

  @classmethod
  def update_parameters(cls, module, graph, graph2module):
    def safe_torch_nn_Parameter(torch_tensor, requires_grad):      
      if requires_grad is not None:
        return torch.nn.Parameter(torch_tensor, requires_grad=requires_grad)
      return torch.nn.Parameter(torch_tensor)

    def _graph2module(op):
      node = getattr(op, "node", None)
      for param_type, tensor in node.op.params.items():
        if node.has_bound_params():
          py_tensor_util.param_to_torch_format(tensor)

        data = np.copy(tensor.data)
        if node.op.type in [NNDCT_OP.CONVTRANSPOSE2D, NNDCT_OP.CONVTRANSPOSE3D] and param_type == node.op.ParamName.WEIGHTS:
          # data = data.transpose(1, 0, 2, 3)
          data = data.swapaxes(0, 1)
          data = np.ascontiguousarray(data)

        if node.op.type in [NNDCT_OP.DEPTHWISE_CONV2D, NNDCT_OP.DEPTHWISE_CONV3D] and param_type == node.op.ParamName.WEIGHTS:
          out_channels = node.node_config("out_channels")
          kernel_size = node.node_config("kernel_size")
          data = data.reshape((out_channels, 1, *kernel_size))

        if node.op.type in [NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D] and param_type == node.op.ParamName.WEIGHTS:
          in_channels = node.node_config("in_channels")
          kernel_size = node.node_config("kernel_size")
          data = data.reshape((1, in_channels, *kernel_size))
          data = data.swapaxes(0, 1)
          data = np.ascontiguousarray(data)

        torch_tensor = torch.from_numpy(data)
        tensor_dtype = torch_tensor.dtype
        param_name = cls._parameter_map.get(param_type, param_type.value)
        if node.has_bound_params():
          if hasattr(op, param_name):
            if isinstance(getattr(op, param_name), torch.Tensor):
              torch_tensor = torch_tensor.to(
                  getattr(op, param_name)).to(tensor_dtype)
            else:
              torch_tensor = torch_tensor.to(
                  getattr(op, param_name).data).to(tensor_dtype)

            if param_name in op._buffers:
              op._buffers[param_name] = torch_tensor
            else:
              op._parameters[param_name] = safe_torch_nn_Parameter(torch_tensor, tensor.requires_grad)
          else:
            NndctScreenLogger().warning(f"new parameter: '{param_name}' is registered in {node.name}")
            op.register_parameter(param_name, safe_torch_nn_Parameter(torch_tensor, tensor.requires_grad))
        else:
          torch_tensor = torch_tensor.to(device=GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_DEVICE))
          module.register_parameter(param_name, safe_torch_nn_Parameter(torch_tensor, tensor.requires_grad))

        if node.has_bound_params():
          py_tensor_util.param_to_nndct_format(tensor)

    # No one will call it and will be removed later.
    def _module2graph(op):
      node = getattr(op, "node", None)
      for param_name, tensor in node.op.params.items():
        if hasattr(op, cls._parameter_map[param_name]):
          if param_name in [torch_op_def.TorchBatchNorm.ParamName.MOVING_MEAN,
                            torch_op_def.TorchBatchNorm.ParamName.MOVING_VAR]:
            torch_tensor = getattr(op, cls._parameter_map[param_name])
          else:
            torch_tensor = getattr(op, cls._parameter_map[param_name]).data

          torch_tensor_data = np.copy(torch_tensor.detach().cpu().numpy())

          if node.op.type == NNDCT_OP.CONVTRANSPOSE2D and param_name == node.op.ParamName.WEIGHTS:
            torch_tensor_data = torch_tensor_data.transpose(1, 0, 2, 3)
            torch_tensor_data = np.ascontiguousarray(torch_tensor_data)

          if node.op.type == NNDCT_OP.DEPTHWISE_CONV2D and param_name == node.op.ParamName.WEIGHTS:
            in_channels = node.node_config("in_channels")
            out_channels = node.node_config("out_channels")
            kernel_size = node.node_config("kernel_size")
            channel_mutiplier = int(out_channels / in_channels)
            torch_tensor_data = torch_tensor_data.reshape((channel_mutiplier, in_channels, *kernel_size))

          tensor.from_ndarray(torch_tensor_data)
          py_tensor_util.param_to_nndct_format(tensor)

    if not _is_module_hooked(module):
      cls.hook_module_with_node(module, graph)

    func = _graph2module if graph2module else _module2graph
    # partial_func = partial(
    #     _graph2module if graph2module else _module2graph,
    #     param_map=cls._parameter_map)
    cls.apply_to_children(module, func)

  @staticmethod
  def _get_output_data(outptus, outptus_name):
    try:
        # output_data = outptus.cpu().detach().numpy()
        output_data = outptus.numpy()
    except AttributeError:
      NndctScreenLogger().warning(f"{outptus_name} is not tensor.")
      output_data = outptus
    return output_data

  @staticmethod
  def _get_output_shape(outputs, outputs_name):
    try:
      output_shape = list(outputs.size())
    except AttributeError:
      NndctScreenLogger().warning(f"{outputs_name} is not tensor. It's shape is ignored.")
      return None
    else:
      return output_shape

  @classmethod
  def update_blobs_once(cls, module, graph=None, time_step=None, update_shape_only=False):
    def _updata_tensor_data(node, nndct_tensor, torch_tensor):
      output_data = cls._get_output_data(torch_tensor, nndct_tensor.name)
      if output_data is not None and isinstance(output_data, np.ndarray):
        output_data = permute_data(output_data, node.transpose_out_order)
        nndct_tensor.from_ndarray(output_data)   
      else:
        nndct_tensor.data = output_data
    
    def _updata_tensor_shape(node, nndct_tensor, torch_tensor):
      output_shape = cls._get_output_shape(torch_tensor, nndct_tensor.name)
      if output_shape is not None:
        output_shape = permute_axes(output_shape, node.transpose_out_order)
        nndct_tensor.shape = output_shape

    def _update_node_outputs(op):
      if hasattr(op, "node") and op.node is not None:
        node = op.node
        if not hasattr(op, "__outputs__") or op.__outputs__ is None or (isinstance(op.__outputs__, TimeStepData) and len(op.__outputs__) == 0):
          return
        one_step_outputs = op.__outputs__[
            time_step] if time_step is not None else op.__outputs__
        if len(node.out_tensors) > 1:
          for idx, tensor in enumerate(node.out_tensors):
            if not update_shape_only:
              _updata_tensor_data(node, tensor, one_step_outputs[idx])
            else:
              _updata_tensor_shape(node, tensor, one_step_outputs[idx])
          
        else:
          tensor = node.out_tensors[0]
          if not update_shape_only:
            _updata_tensor_data(node, tensor, one_step_outputs)
          else:
            _updata_tensor_shape(node, tensor, one_step_outputs)
        

    if not _is_module_hooked(module) and graph is not None:
      cls.hook_module_with_node(module, graph)

    cls.apply_to_children(module, _update_node_outputs)


  @classmethod
  def clone_quant_module(cls, quant_module, quant_graph):
    quantizer = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANTIZER)
  
    if _is_module_hooked(quant_module):
      cls.detach_node_from_module(quant_module)
      cls.hook_module_with_quantizer(quant_module, None)
      new_quant_module = copy.deepcopy(quant_module)
      cls.hook_module_with_node(quant_module, quant_graph)
      #cls.hook_module_with_node(quant_module, quantizer.graph)
      cls.hook_module_with_quantizer(quant_module, quantizer)
      new_graph = Graph(graph_name=quant_graph.name)
      new_graph.clone_from(quant_graph)
      #new_graph = Graph(graph_name=quantizer.graph.name)
      #new_graph.clone_from(quantizer.graph)
      cls.hook_module_with_node(new_quant_module, new_graph)
      cls.hook_module_with_quantizer(new_quant_module, quantizer)
    else:
      new_quant_module = copy.deepcopy(quant_module)

    return new_quant_module


  @classmethod
  def hook_module_with_quantizer(cls, module, quantizer):
    def _add_quantizer_on_module(op):
      if hasattr(op, "quantizer"):
        op.quantizer = quantizer
    cls.apply_to_children(module, _add_quantizer_on_module)

  @classmethod
  def hook_module_with_input_device_checker(cls, module, module_gen_from_script):

    def _check_input_args(op, input):
      op.nndct_inferenced = True
      quant_device = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_DEVICE)
      input_devices = collect_input_devices(input)
      if any([device != quant_device.type for device in input_devices]):
        NndctScreenLogger().warning2user_once(QWarning.DEVICE_MISMATCH, f"The Device of input args mismatch with quantizer device type({quant_device.type}).")
        _, input = to_device(None, input, device=quant_device)
      if module_gen_from_script:
        return input
      else:
        return get_flattened_input_args(input)

    module.register_forward_pre_hook(_check_input_args)

  @classmethod
  def register_tensor_dtype_and_shape_hook(cls, module):
    torch2nndct_dtype_mapper = lambda torch_dtype: {
      torch.float64: "float64",
      torch.float32: "float32",
      torch.float16: "float16",
      torch.int64: "int64",
      torch.int32: "int32",
      torch.bool: "bool",
    }.get(torch_dtype, torch_dtype)
    handlers = []

    def _record_dtype_and_shape(op, inputs, outputs):
      def _get_output_dtype_and_shape(output):
        if isinstance(output, torch.Tensor):
          return torch2nndct_dtype_mapper(output.dtype), tuple(output.shape)
        else:
          return type(output), None
      quantizer = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANTIZER)
      if hasattr(op, "node"):
        node = op.node
        if not isinstance(outputs, (list, tuple)):
          outputs = [outputs]
        for nndct_tensor, output in zip(node.out_tensors, outputs):
          dtype, tensor_shape = _get_output_dtype_and_shape(output)
          nndct_tensor.from_des(tensor_shape, dtype)

    def _hooker(op, record_func):
      handlers.append(op.register_forward_hook(record_func))

    partial_func = partial(_hooker, record_func=_record_dtype_and_shape)
    cls.apply_to_children(module, partial_func)
    return handlers
  


  
