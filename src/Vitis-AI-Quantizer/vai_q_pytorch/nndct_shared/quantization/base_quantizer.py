

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
from nndct_shared.utils import NndctScreenLogger, NndctOption
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
    self._quant_strategy_info = quant_strategy_info
    self._quant_model = None
    self.bitwidth_w = quant_strategy_info['weights']['bit_width']
    self.bitwidth_a = quant_strategy_info['activation']['bit_width']
    self.mix_bit = quant_strategy_info['mix_bit']
    self.export_file = '/'.join([output_dir, 'quant_info.json'])
    self.quant_file = '/'.join([output_dir, 'quant_info.json'])
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
    self.param_file = '/'.join([output_dir, 'param.pth'])
    self.keep_fp = False
    self.quant_strategy = None

  @classmethod
  def create_from_strategy(cls, quant_mode, output_dir, quant_strategy_info, is_lstm=False):
    return cls(quant_mode, 
               output_dir, 
               quant_strategy_info,
               is_lstm = is_lstm)
  
  def setup(self, nndct_graph, rnn_front_end=False, lstm=False, custom_quant_ops=None):

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
      self._configer = QuantInfoMgr(self.Nndctgraph, 
                                    model_type,
                                    self.lstm,
                                    self.quant_strategy_info,
                                    self.quant_strategy,
                                    custom_quant_ops=custom_quant_ops)
      self._QuantInfo, self._QuantAlgo = self._configer.quant_info, self._configer.quant_algo
      self.quant_opt = {
          'range': 2,
          'round_method': 2,
      }

      # calibration and quantization awared training mode
      if self.quant_mode in [1, 3]:
        self.init_quant_config()
      if self.quant_mode > 1:
        # param/output/input names 
        paramBak = self._QuantInfo['param'].keys()
        outputBak = self._QuantInfo['output'].keys()
        inputBak = self._QuantInfo['input'].keys()
        self.load_quant_config()
        # check node names in loaded quant_info.json are all the same as those in test mode
        if (paramBak != self._QuantInfo['param'].keys() or
            outputBak != self._QuantInfo['output'].keys() or
            inputBak != self._QuantInfo['input'].keys()):
          NndctScreenLogger().error(f"Node name mismatch is found when \
loading quantization steps of tensors. \
Please make sure Vai_q_pytorch version and pytorch version for test mode \
are the same as those in calibration (or QAT training) mode.") 
          #exit(2)
        if NndctOption.nndct_stat.value > 0:
          print('Loaded quantization infos:')
          pp.pprint(self._QuantInfo)
          
      
      # initialize param correction
      self.init_param_correction()

  @nndct_utils.not_implement
  def do_scan(self, res, max, min, name, node, tensor_type='input'):
    pass

  @nndct_utils.not_implement
  def do_quantize(self, blob, name, node, tensor_type='input'):
    pass

  def init_quant_config(self):
    self.__config_history = {'output':{}, 'input':{}}
    for item in self._QuantInfo['output']:
      self.__config_history['output'][item] = []
    for item in self._QuantInfo['input']:
      self.__config_history['input'][item] = []
    self._QuantInfo['fast_finetuned'] = False
    self._QuantInfo['bias_corrected'] = False

  #@abstractmethod
  @nndct_utils.not_implement
  def get_quant_config(self, name, real_value=True, tensor_type='output'):
    pass

  #@abstractmethod
  def set_quant_config(self, name, config, tensor_type='output'):
    pass

  # def get_quant_algo(self, name, tensor_type='output'):
  #   if (tensor_type == 'output' and 
  #       name not in self._QuantAlgo[tensor_type].keys()):
  #     name = self.configer.quant_output(name).name
  #   quant_algo = copy.deepcopy(self._QuantAlgo[tensor_type][name])
  #   return quant_algo

  def get_tensor_des(self, tensor):
    return str(tensor)

  def load_quant_config(self, config=None):
    path = pathlib.Path(self.quant_file)
    if not (path.exists() and path.is_file()):
      NndctScreenLogger().error(f"quantization result file does not exist. \
Please check calibration is done or not.")
      exit(2)
    config = config or self.quant_file
    self._QuantInfo = nndct_utils.load_json_obj(config)

  def init_param_correction(self):
    if NndctOption.nndct_param_corr.value > 0:
      if self.quant_mode == 1: 
        for node in self.Nndctgraph.nodes:
          if node.op.type in [NNDCT_OP.CONV2D,
                              NNDCT_OP.CONVTRANSPOSE2D,
                              NNDCT_OP.DEPTHWISE_CONV2D,
                              NNDCT_OP.DENSE,
                              NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D]:
            self.bias_corr[node.name] = None

  def need_quantize_tensor(self, node_name, tensor_type='output'):
    # use quant info to control a tensor of a node need quantization or not
    ret = False
    if node_name in self._QuantInfo[tensor_type].keys():
      ret = True
    return ret
  
  @property
  def configer(self):
    return self._configer

  @property
  def config_history(self):
    return self.__config_history

  @property
  def quant_config(self):
    return self._QuantInfo

  @property
  def quant_algo(self):
    return self._QuantAlgo

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
    return self.quant_config['fast_finetuned']
 
  @fast_finetuned.setter
  def fast_finetuned(self, val):
    self.quant_config['fast_finetuned'] = val

  @property
  def bias_corrected(self):
    return self.quant_config['bias_corrected']
 
  @bias_corrected.setter
  def bias_corrected(self, val):
    self.quant_config['bias_corrected'] = val

class OriginBaseQuantizer(BaseQuantizer):

  def get_quant_config(self, name, real_value=True, tensor_type='output'):
    if (tensor_type == 'output' and 
        name not in self._QuantInfo[tensor_type].keys()):
      name = self.configer.quant_output(name).name
    bnfp = copy.deepcopy(self._QuantInfo[tensor_type][name])
    # get max quantized integer and 2^fix_pos
    if real_value:
      # BN layers are not quantized
      if self.quant_mode == 2 and bnfp[1] is None:
        print('Warning!!! The parameter/activation is not quantized: %s' % name)
        bnfp[0] = 65536 * 1024
        bnfp[1] = 4096
        return bnfp
      try:
        bnfp = self._get_amp_bnfps(bnfp)
      except OverflowError as e:
        print("fragpos of {} : {}".format(name, repr(e)))
    return bnfp

  def _get_amp_bnfps(self, bnfp):
    bn, fp = bnfp
    bn = 2**(bn - 1)
    if fp is not None:
      fp = 2**fp if fp > 0 else 1.0 / 2**(-fp)
    return [bn, fp]

  def set_quant_config(self, name, config, tensor_type='output'):
    assert len(config)==2, "expect 2 parameters, got " + str(config)
    if (tensor_type == 'output' and 
        name not in self._QuantInfo[tensor_type].keys()):
      name = self.configer.quant_output(name).name
    self._QuantInfo[tensor_type][name][0] = config[0]
    self._QuantInfo[tensor_type][name][1] = config[1]

class NewBaseQuantizer(BaseQuantizer):

  def get_quant_config(self, name, real_value=True, tensor_type='output'):
    if (tensor_type == 'output' and 
        name not in self._QuantInfo[tensor_type].keys()):
      name = self.configer.quant_output(name).name
    quant_config = copy.deepcopy(self._QuantInfo[tensor_type][name])
    # get max quantized integer and 2^fix_pos
    if real_value:
      # BN layers are not quantized
      if self.quant_mode == 2 and quant_config[1] is None:
        print('Warning!!! The parameter/activation is not quantized: %s' % name)
        quant_config[0] = 65536 * 1024
        quant_config[1] = 1.0
        quant_config[2] = 0
        quant_config[3] = 0.0
        return quant_config
      try:
        quant_config = self._get_amp_configs(quant_config)
      except OverflowError as e:
        print("fragpos of {} : {}".format(name, repr(e)))
    return quant_config

  def _get_amp_configs(self, config):
    bn, scale, zero_point, float_max = config
    #bn = 2**(bn - 1)
    return [bn, scale, zero_point, float_max]

  def set_quant_config(self, name, config, tensor_type='output'):
    assert len(config)==4, "expect 4 parameters, got " + str(config)
    if (tensor_type == 'output' and 
        name not in self._QuantInfo[tensor_type].keys()):
      name = self.configer.quant_output(name).name
    self._QuantInfo[tensor_type][name][0] = config[0]
    self._QuantInfo[tensor_type][name][1] = config[1]
    self._QuantInfo[tensor_type][name][2] = config[2]
    self._QuantInfo[tensor_type][name][3] = config[3]

  def get_quant_algo(self, name, tensor_type='output'):
    if (tensor_type == 'output' and 
        name not in self._QuantAlgo[tensor_type].keys()):
      name = self.configer.quant_output(name).name
    #quant_algo = self._QuantAlgo[tensor_type][name]
    #quant_algo = copy.deepcopy(self._QuantAlgo[tensor_type][name])
    quant_algo = self._QuantAlgo[tensor_type].get(name, None)
    return quant_algo


