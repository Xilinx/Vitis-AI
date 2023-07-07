

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
from typing import NoReturn, Optional, Dict, List
import numpy as np
import re
from nndct_shared.nndct_graph import Graph
from nndct_shared.quantization import quantize_data2int
from nndct_shared.base import GLOBAL_MAP, NNDCT_KEYS, NNDCT_OP
from nndct_shared import utils as nndct_utils
from nndct_shared.utils import NndctScreenLogger

NndctQuantInfo = Dict[str, Dict[str, List[int]]]


class DeployChecker(object):

  def __init__(self, output_dir_name: str, data_format="bin"):
    if data_format not in ["bin", "txt"]:
      raise RuntimeError("The dump data file format should be txt or bin file.")
    self._data_format = data_format
    self._quant_off = nndct_utils.NndctOption.nndct_quant_off.value
    if self._quant_off:
      self._dump_folder = os.path.join(output_dir_name, NNDCT_KEYS.DEPLOY_CHECK_DATA_FOLDER + '_float')
    else:
      self._dump_folder = os.path.join(output_dir_name, NNDCT_KEYS.DEPLOY_CHECK_DATA_FOLDER + '_int')

    self._full_folder = self._dump_folder
    self.dump_file_suffix = ""
    self.dump_file_prefix = ""

  def update_dump_folder(self, sub_dir: str) -> NoReturn:
    pattern = re.compile(r'[^0-9A-Za-z\/]')
    formal_sub_dir = re.sub(pattern, "_", sub_dir)
    self._full_folder = os.path.join(self._dump_folder, formal_sub_dir)

  def _dump_floating_model(self, nndct_graph, enable_dump_weight, round_method, select_batch) -> NoReturn:
    for node in nndct_graph.nodes:
      if node.has_custom_op():
        # print('-------------------custom op---------------------')
        self._dump_custom_op(node, round_method, select_batch)
        continue
    
      if enable_dump_weight:
        for param_type, param_tensor in node.op.params.items():
          data = param_tensor.data
          if node.op.type in [NNDCT_OP.CONVTRANSPOSE3D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D] and param_type == node.op.ParamName.WEIGHTS:
            data = np.flip(data.copy(), (1, 2, 3))
          elif node.op.type in [NNDCT_OP.CONVTRANSPOSE2D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D] and param_type == node.op.ParamName.WEIGHTS:
            data = np.flip(data.copy(), (1, 2))
            
          self.dump_tensor_to_file(
              param_tensor.name,
              param_tensor.data,
              select_batch=False,
              round_method=round_method)
      if len(node.out_tensors) > 1:
        NndctScreenLogger().warning(f"Only dump first output of multi-output node:'{node.name}({node.op.type})'.")

      for i, tensor in enumerate(node.out_tensors):
        if i > 0:
          break 
        self.dump_tensor_to_file(
            node.name,
            tensor.data,
            select_batch=select_batch,
            round_method=round_method)
        
  def _dump_custom_op(self, node, round_method, select_batch) -> NoReturn:
    for param_type, param_tensor in node.op.params.items():
      data = param_tensor.data
        
      self.dump_tensor_to_file(
          param_tensor.name,
          param_tensor.data,
          select_batch=False,
          round_method=round_method)

    for i, tensor in enumerate(node.out_tensors):
      if i > 0:
        break
      self.dump_tensor_to_file(
          node.name,
          tensor.data,
          select_batch=select_batch,
          round_method=round_method)
    
    for i, tensor in enumerate(node.in_tensors):
      if tensor.node:
        self.dump_tensor_to_file(
            tensor.node.name,
            tensor.data,
            select_batch=select_batch,
            round_method=round_method)

  def _dump_fixed_model(self, nndct_graph, quant_configs, enable_dump_weight, round_method, select_batch) -> NoReturn:
      for node in nndct_graph.nodes:
        if not node.in_quant_part:
          continue
        if enable_dump_weight:
          for param_type, param_tensor in node.op.params.items():
            if param_tensor.name in quant_configs['param']:
              bit_width, fix_point = quant_configs['param'][param_tensor.name]
              data = param_tensor.data
              if node.op.type in [NNDCT_OP.CONVTRANSPOSE3D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D] and param_type == node.op.ParamName.WEIGHTS:
                data = np.flip(data.copy(), (1, 2, 3))
              elif node.op.type in [NNDCT_OP.CONVTRANSPOSE2D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D] and param_type == node.op.ParamName.WEIGHTS:
                data = np.flip(data.copy(), (1, 2))
              self.dump_tensor_to_file(
                  param_tensor.name + NNDCT_KEYS.FIX_OP_SUFFIX,
                  data,
                  bit_width,
                  fix_point,
                  select_batch=False,
                  round_method=round_method)
        if len(node.out_tensors) > 1:
          NndctScreenLogger().warning(f"Only dump first output of multi-output node:'{node.name}({node.op.type})'.")
        
        if node.op.type == NNDCT_OP.QUANT_STUB:
          for i, tensor in enumerate(node.out_tensors):
            if i > 0:
              break
            if node.name not in quant_configs['output'].keys():
              quantizer = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANTIZER)
              end = quantizer.configer.quant_output(node).name
              bit_width, fix_point = quant_configs['output'][end]
              self.dump_tensor_to_file(
                  node.name + NNDCT_KEYS.FIX_OP_SUFFIX,
                  tensor.data,
                  bit_width,
                  fix_point,
                  select_batch=select_batch,
                  round_method=-1)

        if node.name in quant_configs['output']:
          for i, tensor in enumerate(node.out_tensors):
            if i > 0:
              break
            bit_width, fix_point = quant_configs['output'][node.name]
            self.dump_tensor_to_file(
                node.name + NNDCT_KEYS.FIX_OP_SUFFIX,
                tensor.data,
                bit_width,
                fix_point,
                select_batch=select_batch,
                round_method=round_method)
            
        if node.name in quant_configs['input']:
          bit_width, fix_point = quant_configs['input'][node.name]
          for idx, tensor in enumerate(node.in_tensors):
            self.dump_tensor_to_file(
                node.name + NNDCT_KEYS.PRE_FIX_OP_SUFFIX + f"_i{idx}",
                tensor.data,
                bit_width,
                fix_point,
                select_batch=select_batch,
                round_method=round_method)
            
  def _dump_graph_info(self, nndct_graph, quant_configs) -> NoReturn:
      # dump tensor shape information
      file_name = os.path.join(self._full_folder, "shape.txt")
      with open(file_name, "w") as file_obj:
        for node in nndct_graph.nodes:
          # if node.name in quant_configs['output']:
          for tensor in node.out_tensors:
            if tensor.shape is not None:
              file_obj.write("{}: {}\n".format(tensor.shape, node.name))
            
                  
  def dump_nodes_output(self, nndct_graph: Graph, quant_configs: NndctQuantInfo,
                        round_method: int, enable_dump_weight: bool = True, select_batch: bool = False) -> NoReturn:
    
    nndct_utils.create_work_dir(self._full_folder)
    if self._quant_off:
      self._dump_floating_model(nndct_graph, enable_dump_weight, round_method, select_batch)
    else:
      self._dump_floating_model(nndct_graph, enable_dump_weight, round_method, select_batch)
      self._dump_fixed_model(nndct_graph, quant_configs, enable_dump_weight, round_method, select_batch)
      self._dump_graph_info(nndct_graph, quant_configs)

  def dump_tensor_to_file(self,
                          file_name: str,
                          data: np.ndarray,
                          bit_width: Optional[int] = None,
                          fix_point: Optional[int] = None,
                          select_batch: bool = False,
                          round_method: int = 2) -> NoReturn:
    pattern = re.compile(r'[^0-9A-Za-z]')
    formal_file_name = re.sub(pattern, "_", file_name)
    formal_file_name = "".join([self.dump_file_prefix, formal_file_name, self.dump_file_suffix])
    dump_int_file = os.path.join(self._full_folder, ".".join([formal_file_name, self._data_format]))
    dump_float_file = os.path.join(self._full_folder, ".".join([formal_file_name, self._data_format]))
   
    if data is not None and isinstance(data, np.ndarray):
      if select_batch and data.ndim > 0:
        dump_data = data[:1]
      else:
        dump_data = data
      if bit_width is None or fix_point is None:
        if self._data_format == "bin":
          dump_data = np.copy(dump_data).astype("float32")
          dump_data.flatten().tofile(dump_float_file)
        else:
          dump_data.flatten().tofile(dump_float_file, sep="\n", format="%12g")
      else:
        if self._data_format == "bin":
          if bit_width >= 8:
            quantize_data2int(dump_data.flatten(), bit_width, fix_point,
                              round_method).tofile(dump_int_file)
          else:
            int_data = quantize_data2int(dump_data.flatten(), bit_width, fix_point, 
                                         round_method)
            self.compact_low_bit_data_to_8_bit(int_data, bit_width).tofile(dump_int_file)
        else:
          quantize_data2int(dump_data.flatten(), bit_width, fix_point,
                            round_method).tofile(dump_int_file, sep="\n", format="%g")

  @staticmethod
  def compact_low_bit_data_to_8_bit(data: np.ndarray, bit_width: int):
    if data.ndim != 1:
      raise RuntimeError("Flatten data before data compaction.")
    if bit_width == 4:
      high_4_bits_data = data[::2]
      low_4_bits_data = data[1::2]
      if len(high_4_bits_data) > len(low_4_bits_data):
        low_4_bits_data = np.append(low_4_bits_data, 0)
      high_4_bits_data = np.left_shift(high_4_bits_data, 4)
      low_4_bits_data = np.bitwise_and(low_4_bits_data, 15)
      data = np.bitwise_or(high_4_bits_data, low_4_bits_data)
    else:
      NotImplementedError("Only support compacting 4 bit data to 8 bit data.")
    
    return data
      
  @property
  def dump_folder(self):
    return self._full_folder
  
