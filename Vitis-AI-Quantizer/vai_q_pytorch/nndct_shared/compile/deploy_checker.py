import os
from abc import abstractmethod
from typing import NoReturn, Optional, Dict, List
import numpy as np
import re
from nndct_shared.nndct_graph import Graph
from nndct_shared.quantization import quantize_data2int
from nndct_shared.base import NNDCT_KEYS
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
    
  def dump_nodes_output(self, nndct_graph: Graph, quant_configs: NndctQuantInfo,
                        round_method: int, enable_dump_weight=True) -> NoReturn:

    def _dump_floating_model() -> NoReturn:
      for node in nndct_graph.nodes:
        if enable_dump_weight:
          for _, param_tensor in node.op.params.items():
            self.dump_tensor_to_file(
                param_tensor.name, param_tensor.data, round_method=round_method)
        if len(node.out_tensors) > 1:
          raise RuntimeError(
              "Don't support multi-output op:'{} {}' for deploying!".format(
                  node.name, node.op.type))
        for tensor in node.out_tensors:
          self.dump_tensor_to_file(
              node.name, tensor.data, round_method=round_method)

    def _dump_fixed_model() -> NoReturn:
      for node in nndct_graph.nodes:
        if enable_dump_weight:
          for _, param_tensor in node.op.params.items():
            if param_tensor.name in quant_configs['params']:
              bit_width, fix_point = quant_configs['params'][param_tensor.name]
              self.dump_tensor_to_file(
                  param_tensor.name + NNDCT_KEYS.FIX_OP_SUFFIX,
                  param_tensor.data,
                  bit_width,
                  fix_point,
                  round_method=round_method)
        if len(node.out_tensors) > 1:
          raise RuntimeError(
              "Don't support multi-output op:'{} {}' for deploying!".format(
                  node.name, node.op.type))
        if node.name in quant_configs['blobs']:
          for tensor in node.out_tensors:
            bit_width, fix_point = quant_configs['blobs'][node.name]
            self.dump_tensor_to_file(
                node.name + NNDCT_KEYS.FIX_OP_SUFFIX,
                tensor.data,
                bit_width,
                fix_point,
                round_method=round_method)

    def _dump_graph_info()-> NoReturn:
      # dump tensor shape information
      file_name = os.path.join(self._full_folder, "shape.txt")
      with open(file_name, "w") as file_obj:
        for node in nndct_graph.nodes:
          if node.name in quant_configs['blobs']:
            for tensor in node.out_tensors:
              try:
                file_obj.write("{}: {}\n".format(tensor.data.shape, node.name))
              except AttributeError as e:
                NndctScreenLogger().warning(f"{tensor.name} is not tensor.It's shape info is ignored.")
    
    nndct_utils.create_work_dir(self._full_folder)
    if self._quant_off:
      _dump_floating_model()
    else:
      _dump_floating_model()
      _dump_fixed_model()
      _dump_graph_info()

  def dump_tensor_to_file(self,
                          file_name: str,
                          data: np.ndarray,
                          bit_width: Optional[int] = None,
                          fix_point: Optional[int] = None,
                          round_method: int = 2) -> NoReturn:
    pattern = re.compile(r'[^0-9A-Za-z]')
    formal_file_name = re.sub(pattern, "_", file_name)
    formal_file_name = "".join([self.dump_file_prefix, formal_file_name, self.dump_file_suffix])
    dump_int_file = os.path.join(self._full_folder, ".".join([formal_file_name, self._data_format]))
    dump_float_file = os.path.join(self._full_folder, ".".join([formal_file_name, self._data_format]))
    if data is not None and isinstance(data, np.ndarray):
      if bit_width is None or fix_point is None:
        if self._data_format == "bin":
          data = np.copy(data).astype("float32")
          data.flatten().tofile(dump_float_file)
        else:
          data.flatten().tofile(dump_float_file, sep="\n", format="%12g")
      else:
        if self._data_format == "bin":
          quantize_data2int(data.flatten(), bit_width, fix_point,
                            round_method).tofile(dump_int_file)
        else:
          quantize_data2int(data.flatten(), bit_width, fix_point,
                            round_method).tofile(dump_int_file, sep="\n", format="%12g")

  @property
  def dump_folder(self):
    return self._full_folder
  
