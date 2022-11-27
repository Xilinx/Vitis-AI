

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

import numpy as np

from nndct_shared import utils as nndct_utils
from nndct_shared.base import NNDCT_KEYS, NNDCT_OP
from nndct_shared.utils import NndctScreenLogger, NndctOption
from .quant_info import QuantInfoMgr


class BaseQuantizer():

  def __init__(self, quant_mode: int, output_dir: str, bitwidth_w: int,
          bitwidth_a: int, mix_bit: bool = False):

    # initialization
    self.quant_mode = quant_mode
    self.bitwidth_w = bitwidth_w
    self.bitwidth_a = bitwidth_a
    self.mix_bit = mix_bit
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
    if NndctOption.nndct_param_corr.value > 0:
      self.bias_corr = {}
    self.bias_corr_file = '/'.join([output_dir, 'bias_corr.pth'])
    self.param_file = '/'.join([output_dir, 'param.pth'])
    self.keep_fp = False

  @classmethod
  def create_from_strategy(cls, quant_mode, output_dir, quant_strategy):
    return cls(quant_mode, 
               output_dir, 
               quant_strategy.num_bits_w, 
               quant_strategy.num_bits_a,
               quant_strategy.mix_bit)
  
  def setup(self, nndct_graph, rnn_front_end=False, lstm=False, mix_bit=False, custom_quant_ops=None):

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
                                    self.bitwidth_w, 
                                    self.bitwidth_a, 
                                    self.lstm,
                                    self.mix_bit,
                                    custom_quant_ops=custom_quant_ops)
      self._QuantInfo = self._configer.quant_info
      self.quant_opt = {
          'range': 2,
          'round_method': 2,
      }

      # calibration and quantization awared training mode
      if self.quant_mode in [1, 3]:
        self.init_scan_config()
      if self.quant_mode > 1:
        self.init_quant_config()
      
      # initialize param correction
      self.init_param_correction()

  @nndct_utils.not_implement
  def do_scan(self, res, max, min, name, node, tensor_type='input'):
    pass

  @nndct_utils.not_implement
  def do_quantize(self, blob, name, node, tensor_type='input'):
    pass

  def init_scan_config(self):
    self.__fp_history= {'output':{}, 'input':{}}
    for item in self._QuantInfo['output']:
      self.__fp_history['output'][item] = []
    for item in self._QuantInfo['input']:
      self.__fp_history['input'][item] = []

  def get_bnfp(self, name, real_value=True, tensor_type='output'):
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

  def set_bnfp(self, name, bnfp, tensor_type='output'):
    if (tensor_type == 'output' and 
        name not in self._QuantInfo[tensor_type].keys()):
      name = self.configer.quant_output(name).name
    self._QuantInfo[tensor_type][name][0] = bnfp[0]
    self._QuantInfo[tensor_type][name][1] = bnfp[1]

  def get_tensor_des(self, tensor):
    return str(tensor)

  def init_quant_config(self, config=None):
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
  def fp_history(self):
    return self.__fp_history

  @property
  def quant_config(self):
    return self._QuantInfo

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

