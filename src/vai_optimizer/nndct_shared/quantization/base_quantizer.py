
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
import json
import pathlib

import numpy as np
from typing import Dict, Union

from nndct_shared import utils as nndct_utils
from nndct_shared.base import NNDCT_KEYS, NNDCT_OP
from nndct_shared.utils import NndctScreenLogger, NndctOption, QError, QWarning
from .quant_info import QuantInfoMgr
import pprint
pp = pprint.PrettyPrinter(indent=4)


class BaseQuantizer():

  #def __init__(self, quant_mode: int, output_dir: str, bitwidth_w: int,
  #        bitwidth_a: int, mix_bit: bool = False):
  def __init__(self, quant_mode: int, output_dir: str, 
          quant_strategy_info: Dict[str, Union[str, int, bool]], is_lstm=False):

    # initialization
    self.quant_mode = quant_mode
    self.output_dir = output_dir
    self._quant_strategy_info = quant_strategy_info
    self._quant_model = None
    self.bitwidth_w = quant_strategy_info['weights']['bit_width']
    self.bitwidth_a = quant_strategy_info['activation']['bit_width']
    self.mix_bit = quant_strategy_info['mix_bit']
    self.export_file = '/'.join([output_dir, 'quant_info.json'])
    self.quant_file = '/'.join([output_dir, 'quant_info.json'])
    self.dtype_file = '/'.join([output_dir, 'quant_dtype.json'])
    self.quant_table_file = None
    self.platform_type = 'nndct'
    # add a rnn_front_end flag while lstm flag represents rnn_back_end.
    # maybe we have better name to give the two flags, but because lstm
    # exists already, I just obey the origin naming without changing.
    self.rnn_front_end = False
    self.lstm = False
    self.bias_corr = None
    self._bias_corr_loaded = False
    self._finetuned_para_loaded = False
    if NndctOption.nndct_param_corr.value > 0:
      self.bias_corr = {}
    self.bias_corr_file = '/'.join([output_dir, 'bias_corr.pth'])
    self.param_file = '/'.join([output_dir, 'param'])
    self.float_param_path = '/'.join([output_dir, '.float_params'])
    self.keep_fp = False
    self.quant_strategy = None
    self.quant_config_imp = None
    self.Nndctgraph = None
    self._configer = None

  @classmethod
  def create_from_strategy(cls, quant_mode, output_dir, quant_strategy_info, is_lstm=False):
    return cls(quant_mode, 
               output_dir, 
               quant_strategy_info,
               is_lstm = is_lstm)
  
  def setup(self, nndct_graph, rnn_front_end=False, lstm=False, custom_quant_ops=None, target=None, dynamo=False):
    self.Nndctgraph = nndct_graph
    self.rnn_front_end = rnn_front_end
    self.lstm = lstm
    self.calibration_method = 'DiffS'

    # further setup
    if self.quant_mode > 0:
      model_type = self.get_model_type()
      if NndctOption.nndct_stat.value > 1:
        print('nndct graph:')
        print(self.Nndctgraph)
      if target is None:
        self._configer = QuantInfoMgr(self.Nndctgraph, 
                                      model_type,
                                      self.lstm,
                                      self.quant_strategy_info,
                                      self.quant_strategy,
                                      custom_quant_ops=custom_quant_ops)
      else:
        from nndct_shared.quantization.target_quant_info import TargetQuantInfoMgr
        self._configer = TargetQuantInfoMgr(target,
                                      self.Nndctgraph, 
                                      model_type,
                                      self.lstm,
                                      self.quant_strategy_info,
                                      self.quant_strategy,
                                      custom_quant_ops=custom_quant_ops)
    
      self.quant_opt = {
          'range': 2,
          'round_method': 2,
      }

      # calibration and quantization awared training mode
      if self.quant_mode in [1, 3]:
        self.init_quant_config()
      if self.quant_mode > 1:
        # param/output/input names 
        
        self.load_quant_config(dynamo=dynamo)

        if NndctOption.nndct_stat.value > 0:
          print('Loaded quantization infos:')
          pp.pprint(self.quant_config)
          
      # initialize param correction
      self.init_param_correction()

  # @nndct_utils.not_implement
  # def calibrate(self, res, max, min, name, node, tensor_type='input', idx=0, method=None):
  #   pass

  # @nndct_utils.not_implement
  # def quantize(self, blob, name, node, tensor_type='input', idx=0):
  #   pass

  def init_quant_config(self):
    self.quant_config_imp.init_quant_config()

  def get_quant_len(self, name, tensor_type='output'):
    return self.quant_config_imp.get_quant_len(name, self.configer, tensor_type)
  
  def get_quant_config(self, name, real_value=True, tensor_type='output', idx=0):
    return self.quant_config_imp.get_quant_config(name, self.configer, real_value, tensor_type, idx)

  def set_quant_config(self, name, config, tensor_type='output', idx=0):
    self.quant_config_imp.set_quant_config(name, config, self.configer, tensor_type, idx)
    
  def get_in_node_config(self, node, in_node_index, real_value=True, tensor_type='output'):
    return self.quant_config_imp.get_in_node_config(node, in_node_index, self.configer, real_value, tensor_type)
  
  def get_fix_position(self, name, tensor_type='output', idx=0):
    return self.quant_config_imp.get_fix_position(name, self.configer, tensor_type, idx)
   
  def set_fix_position(self, name, fp, tensor_type='output', idx=0):
    self.quant_config_imp.set_fix_position(name, fp, self.configer, tensor_type, idx)
  
  def insert_local_quant_config(self, name, config, tensor_type='output'):
    self.quant_config_imp.insert_local_quant_config(name, config, tensor_type)
  
  def have_quant_or_not(self, name, tensor_type='output'):
    return self.quant_config_imp.have_quant_or_not(name, tensor_type)

  def get_quant_algo(self, name, tensor_type='output', idx=0):
    return self.quant_config_imp.get_quant_algo(name, self.configer, tensor_type, idx)
  
  def get_quant_dtype(self, name, tensor_type='output', idx=0):
    return self.quant_config_imp.get_quant_dtype(name, self.configer, tensor_type, idx)

  def get_tensor_des(self, tensor):
    return str(tensor)

  def load_quant_config(self, config=None, dynamo=False):
    path = pathlib.Path(self.quant_file)
    if not (path.exists() and path.is_file()):
      NndctScreenLogger().error2user(QError.NO_CALIB_RESULT, f"Quantization \
calibration result file does not exist. \
Please check calibration is done or not.")
      exit(2)
    config = config or self.quant_file
    return self.quant_config_imp.load_quant_config(config, self.Nndctgraph, dynamo)

  def init_param_correction(self):
    if NndctOption.nndct_param_corr.value > 0:
      if self.quant_mode == 1: 
        for node in self.Nndctgraph.all_nodes():
          if node.op.type in [NNDCT_OP.CONV2D,
                              NNDCT_OP.CONVTRANSPOSE2D,
                              NNDCT_OP.DEPTHWISE_CONV2D,
                              NNDCT_OP.DENSE,
                              NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D]:
            self.set_bias_corr(node.normalized_name, None)
      elif self.quant_mode == 2:
        self.recover_bias_corr()

  def need_quantize_tensor(self, node_name, tensor_type='output'):
    # use quant info to control a tensor of a node need quantization or not
    ret = False
    quantizable_info = self.quant_config[tensor_type] if NndctOption.nndct_only_int_quant.value is True \
      else self.quant_dtype[tensor_type]
    if node_name in quantizable_info.keys():
      ret = True
    return ret

  def normalized_quant_config(self):
    return self.quant_config_imp.normalized_quant_config(self.Nndctgraph)
  
  def normalized_quant_dtype(self):
    return self.quant_config_imp.normalized_quant_dtype(self.Nndctgraph)
  
  def add_quant_info_for_export(self, key, item, quant_info):
    return self.quant_config_imp.add_quant_info_for_export(key, item, quant_info)

  def get_bias_corr(self, node):
    return self.bias_corr[node.name] if node.name in self.bias_corr else self.bias_corr[node.normalized_name]
  
  def set_bias_corr(self, bias_k, bias_corr):
    self.bias_corr[bias_k] = bias_corr

  def recover_bias_corr(self):
    norm2debug = {node.normalized_name: node.name for node in self.Nndctgraph.all_nodes()}
    bias_norm_keys = list(self.bias_corr.keys())
    for k in bias_norm_keys:
      debug_k = norm2debug.get(k, k)
      v = self.bias_corr[k]
      del self.bias_corr[k]
      self.bias_corr[debug_k] = v
    return self.bias_corr

  def has_bias_corr(self, node):
    return any([key in self.bias_corr for key in [node.name, node.normalized_name]])

  @property
  def configer(self):
    return self._configer

  @property 
  def config_history(self):
    return self.quant_config_imp.config_history

  @property
  def quant_config(self):
    return self.quant_config_imp.quant_config

  @property
  def quant_algo(self):
    return self.quant_config_imp.quant_algo
  
  @property
  def quant_dtype(self):
    return self.quant_config_imp.quant_dtype

  @property
  def quant_strategy_info(self):
    return self._quant_strategy_info

  @property
  def graph(self):
    return self.Nndctgraph

  @property
  def bitw(self):
    return self.bitwidth_w

  @property
  def bita(self):
    return self.bitwidth_a

  @property
  def is_lstm(self):
    return self.lstm

  @property
  def bias_correction(self):
    return self.bias_corr

  @property
  def weight_correction(self):
    return self.weight_corr

  @property
  def quant_model(self):
    return self._quant_model

  @quant_model.setter
  def quant_model(self, quant_model):
    self._quant_model = quant_model

  @property
  def fast_finetuned(self):
    return self.quant_config_imp.fast_finetuned
 
  @fast_finetuned.setter
  def fast_finetuned(self, val):
    self.quant_config_imp.fast_finetuned = val

  @property
  def bias_corrected(self):
    return self.quant_config_imp.bias_corrected
 
  @bias_corrected.setter
  def bias_corrected(self, val):
    self.quant_config_imp.bias_corrected = val
  
  @property
  def version(self):
    return self.quant_config_imp.version
 
  @version.setter
  def version(self, val):
    self.quant_config_imp.version = val
  
  @property
  def graph_md5(self):
    return self.quant_config_imp.graph_md5
  
  @graph_md5.setter
  def graph_md5(self, val):
    self.quant_config_imp.graph_md5 = val
