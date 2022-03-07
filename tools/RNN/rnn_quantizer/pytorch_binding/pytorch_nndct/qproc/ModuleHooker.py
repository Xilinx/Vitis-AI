

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

from functools import partial

import numpy as np
import torch

import nndct_shared.quantization as nndct_quant
import pytorch_nndct.parse.torch_op_def as torch_op_def
import pytorch_nndct.utils.tensor_util as py_tensor_util
from nndct_shared.base import NNDCT_OP, GLOBAL_MAP, NNDCT_KEYS
from nndct_shared.utils import NndctScreenLogger, NndctOption, permute_data
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
      torch_op_def.TorchEmbedding.ParamName.WEIGHT: "weight"

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
            return None

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

    def _graph2module(op):
      node = getattr(op, "node", None)
      for param_type, tensor in node.op.params.items():
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
        param_name = cls._parameter_map.get(param_type, param_type.value)
        if node.has_bound_params():
          if hasattr(op, param_name):
            if isinstance(getattr(op, param_name), torch.Tensor):
              torch_tensor = torch_tensor.to(
                  getattr(op, param_name))
            else:
              torch_tensor = torch_tensor.to(
                  getattr(op, param_name).data)

            if param_name in op._buffers:
              op._buffers[param_name] = torch_tensor
            else:
              op._parameters[param_name] = torch.nn.Parameter(torch_tensor)
          else:
            NndctScreenLogger().warning(f"new parameter: '{param_name}' is registered in {node.name}")
            op.register_parameter(param_name,
                                  torch.nn.Parameter(torch_tensor))
        else:
          torch_tensor = torch_tensor.to(device=GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_DEVICE))
          module.register_parameter(param_name, torch.nn.Parameter(torch_tensor))

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
      NndctScreenLogger().warning(f"{outptus_name} is not tensor. It's data is ignored.")
      return None
    else:
      return output_data

  @classmethod
  def update_blobs_once(cls, module, graph, time_step=None):

    def _update_node_outputs(op):
      if hasattr(op, "node") and op.node is not None:
        node = op.node
        if not hasattr(op, "__outputs__") or op.__outputs__ is None or (isinstance(op.__outputs__, TimeStepData) and len(op.__outputs__) == 0):
          return
        one_step_outputs = op.__outputs__[
            time_step] if time_step is not None else op.__outputs__
        if len(node.out_tensors) > 1:
          for idx, tensor in enumerate(node.out_tensors):
            output_data = cls._get_output_data(one_step_outputs[idx], tensor.name)
            if output_data is not None:
              output_data = permute_data(output_data, node.transpose_order)
              tensor.from_ndarray(output_data)
              # py_tensor_util.blob_to_nndct_format(tensor)
        else:
          tensor = node.out_tensors[0]
          output_data = cls._get_output_data(one_step_outputs, tensor.name)
          if output_data is not None:
            output_data = permute_data(output_data, node.transpose_order)
            tensor.from_ndarray(output_data)
            # py_tensor_util.blob_to_nndct_format(tensor)

    if not _is_module_hooked(module):
      cls.hook_module_with_node(module, graph)

    cls.apply_to_children(module, _update_node_outputs)
