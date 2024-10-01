
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
import numpy as np
from abc import ABC, abstractmethod
from nndct_shared.base import NNDCT_KEYS, GLOBAL_MAP, NNDCT_OP
from nndct_shared import utils as nndct_utils
from nndct_shared.utils import NndctScreenLogger, QWarning, QError

class QuantConfigImpBase(ABC):
  def __init__(self):
    self._QuantInfo = {"param": {}, "output": {}, "input": {}}
    self._QuantAlgo = {'param': {}, 'output': {}, 'input': {}}
    self._QuantDtype = {"param": {}, "output": {}, "input": {}}
    self._config_history = {'output':{}, 'input':{}}
  
  @abstractmethod
  def make_quant_config(self, bitwidth):
    pass
  
  def add_quant_config(self, name, bitwidth, tensor_type='output'):
    quant_config = self.make_quant_config(bitwidth)
    if name not in self._QuantInfo[tensor_type].keys():
      self._QuantInfo[tensor_type][name] = [quant_config]
    else:
      self._QuantInfo[tensor_type][name].append(quant_config)
    
  @abstractmethod
  def get_quant_config(self, name, configer, real_value=True, tensor_type='output', idx=0):
    pass

  @abstractmethod
  def set_quant_config(self, name, config, tensor_type='output', idx=0):
    pass
  
  @abstractmethod
  def insert_local_quant_config(self, name, config, tensor_type='output'):
    pass
  
  def get_fix_position(self, name, tensor_type='output', idx=0):
    pass
   
  def set_fix_position(self, name, fp, tensor_type='output', idx=0):
    pass
  
  @property
  def quant_config(self):
    return self._QuantInfo

  @property
  def quant_algo(self):
    return self._QuantAlgo
  
  @property
  def quant_dtype(self):
    return self._QuantDtype
  
  @property 
  def config_history(self):
    return self._config_history
  
  @property
  def fast_finetuned(self):
    return self._QuantInfo['fast_finetuned']
 
  @fast_finetuned.setter
  def fast_finetuned(self, val):
    self._QuantInfo['fast_finetuned'] = val

  @property
  def bias_corrected(self):
    return self._QuantInfo['bias_corrected']
 
  @bias_corrected.setter
  def bias_corrected(self, val):
    self._QuantInfo['bias_corrected'] = val
  
  @property
  def version(self):
    return self._QuantInfo['version']
 
  @version.setter
  def version(self, val):
    self._QuantInfo['version'] = val
  
  @property
  def graph_md5(self):
    return self._QuantInfo['graph_md5']
  
  @graph_md5.setter
  def graph_md5(self, val):
    self._QuantInfo['graph_md5'] = val
  
  def get_quant_len(self, name, configer, tensor_type='output'):
    if (tensor_type == 'output' and 
      name not in self._QuantInfo[tensor_type].keys()):
      name = configer.quant_output(name).name
    return len(self._QuantInfo[tensor_type][name])
  
  def have_quant_or_not(self, name, tensor_type='output'):
    if name not in self._QuantInfo[tensor_type].keys() or \
        self._QuantInfo[tensor_type][name] is None or \
        len(self._QuantInfo[tensor_type][name])==0:
      return False
    else:
      for config in self._QuantInfo[tensor_type][name]:
        if len(config)>1 and config[1] is not None and isinstance(config[1], int):
          return True
      return False
  
  def init_quant_config(self):
    for item in self._QuantInfo['output']:
      self._config_history['output'][item] = []
      for i in range(len(self._QuantInfo['output'][item])):
        self._config_history['output'][item].append([])
    
    for item in self._QuantInfo['input']:
      self._config_history['input'][item] = []
      for i in range(len(self._QuantInfo['input'][item])):
        self._config_history['input'][item].append([])
    
    self._QuantInfo['fast_finetuned'] = False
    self._QuantInfo['bias_corrected'] = False
    self._QuantInfo['version'] = "Uknown"
  
  def clear_quant_config(self):
    self._QuantInfo = {"param": {}, "output": {}, "input": {}}
    self._QuantAlgo = {'param': {}, 'output': {}, 'input': {}}
    self._QuantDtype = {"param": {}, "output": {}, "input": {}}
  
  def normalized_quant_config(self, graph):
    norm_quant_config = copy.deepcopy(self._QuantInfo)
    for k, v in self._QuantInfo["output"].items():
      norm_quant_config["output"][graph.node(k).normalized_name] = v
      del norm_quant_config["output"][k]

    for k, v in self._QuantInfo["input"].items():
      norm_quant_config["input"][graph.node(k).normalized_name] = v
      del norm_quant_config["input"][k]

    return norm_quant_config
  
  def normalized_quant_dtype(self, graph):
    norm_quant_dtype = copy.deepcopy(self._QuantDtype)
    for k, v in self._QuantDtype["output"].items():
      norm_quant_dtype["output"][graph.node(k).normalized_name] = v
      del norm_quant_dtype["output"][k]

    for k, v in self._QuantDtype["input"].items():
      norm_quant_dtype["input"][graph.node(k).normalized_name] = v
      del norm_quant_dtype["input"][k]

    return norm_quant_dtype
  
  def add_quant_info_for_export(self, key, item, quant_info):
    quant_info[key] = item
    return quant_info

  def load_quant_config(self, config, graph, dynamo=False):
    normalized_quant_config = nndct_utils.load_json_obj(config)
    
    norm2debug = {node.normalized_name: node.name for node in graph.all_nodes()}
    quant_config = copy.deepcopy(normalized_quant_config)
    for k, v in normalized_quant_config["output"].items():
      del quant_config["output"][k]
      debug_k = norm2debug.get(k, k)
      quant_config["output"][debug_k] = v
    
    for k, v in normalized_quant_config["input"].items():
      del quant_config["input"][k]
      debug_k = norm2debug.get(k, k)
      quant_config["input"][debug_k] = v
    
    if dynamo is True and self.illegal_quant_info(quant_config, graph):
      raise RuntimeError()
    else:
      NndctScreenLogger().check2user(QError.CALIB_RESULT_MISMATCH,
          f"Node name mismatch is found when \
  loading quantization steps of tensors. \
  Please make sure Vai_q_pytorch version and pytorch version for test mode \
  are the same as those in calibration (or QAT training) mode.", 
          not self.illegal_quant_info(quant_config, graph), exit_status=0)
    self._QuantInfo = quant_config
    
  def _check_graph_md5(self, quant_info, graph):
    if "graph_md5" not in quant_info:
      return True
    if any([node.op.type == NNDCT_OP.BLOCK for node in graph.all_nodes()]):
      return True
    return quant_info["graph_md5"] == graph.get_md5()
  
  def illegal_quant_info(self, quant_info, graph):
    if (any([p not in quant_info['param'].keys() for p in self._QuantInfo['param'].keys()]) or
      any([o not in quant_info['output'].keys() for o in self._QuantInfo['output'].keys()]) or
      any([i not in quant_info['input'].keys() for i in self._QuantInfo['input'].keys()]) or 
      (not self._check_graph_md5(quant_info, graph))):
      return True
    return False
  
  @staticmethod
  def _get_in_node_quant_config_index(node, in_node_index):
    """
    Maybe the node input tensor comes from a mult-outputs op outputs, 
    so we need to figure out which output is the node input tensor.
    """
    pn = node.owning_graph.parents(node)[in_node_index]
    for idx, o_tensor in enumerate(pn.out_tensors):
      if any([use.user is node for use in o_tensor.uses]):
        return idx

  def get_in_node_config(self, node, in_node_index, configer, real_value=True, tensor_type='output'):
    if node.name in self._QuantInfo['input']:
      return self.get_quant_config(node.name, configer, real_value, 'input', idx=in_node_index)
    else:
      idx = self._get_in_node_quant_config_index(node, in_node_index)
      return self.get_quant_config(node.in_nodes[in_node_index], configer, real_value, tensor_type, idx=idx)
  
  def add_quant_algo(self, name, algo, tensor_type='output'):
    if name not in self._QuantAlgo[tensor_type].keys():
      self._QuantAlgo[tensor_type][name] = [algo]
    else:
      self._QuantAlgo[tensor_type][name].append(algo)
    
  def get_quant_algo(self, name, configer, tensor_type='output', idx=0):
    if (tensor_type == 'output' and 
        name not in self._QuantAlgo[tensor_type].keys()):
      name = configer.quant_output(name).name
    #quant_algo = self._QuantAlgo[tensor_type][name]
    #quant_algo = copy.deepcopy(self._QuantAlgo[tensor_type][name])
    quant_algo = (self._QuantAlgo[tensor_type].get(name, None))[idx]
    return quant_algo
  
  def add_quant_dtype(self, name, dtype, tensor_type='output'):
    if name not in self._QuantDtype[tensor_type].keys():
      self._QuantDtype[tensor_type][name] = [dtype]
    else:
      self._QuantDtype[tensor_type][name].append(dtype)
      
  def get_quant_dtype(self, name, configer, tensor_type='output', idx=0):
    if (tensor_type == 'output' and 
        name not in self._QuantDtype[tensor_type].keys()):
      name = configer.quant_output(name).name
    quant_dtype = (self._QuantDtype[tensor_type].get(name, None))[idx]
    return quant_dtype
  
class DPUQConfigImp(QuantConfigImpBase):
  def make_quant_config(self, bitwidth):
    return [bitwidth, None] if bitwidth is not None else None
  
  def get_quant_config(self, name, configer, real_value=True, tensor_type='output', idx=0):
    if (len(self._QuantInfo) == 0) or (tensor_type not in self._QuantInfo.keys()) or (len(self._QuantInfo[tensor_type]) == 0):
      return None
    
    if (tensor_type == 'output' and 
        name not in self._QuantInfo[tensor_type].keys()):
      name = configer.quant_output(name).name
      
    if name not in self._QuantInfo[tensor_type].keys():
      return None

    if idx >= len(self._QuantInfo[tensor_type][name]):
      idx = len(self._QuantInfo[tensor_type][name]) - 1
    bnfp = copy.deepcopy(self._QuantInfo[tensor_type][name][idx]) 
    quant_mode = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_MODE)
    # get max quantized integer and 2^fix_pos
    if real_value and bnfp is not None:
      # BN layers are not quantized
      if quant_mode == 2 and bnfp[1] is None:
        NndctScreenLogger().warning2user(QWarning.TENSOR_NOT_QUANTIZED, 'The tensor is not quantized: %s' % name)
        # bnfp[0] = 65536 * 1024
        # bnfp[1] = 4096
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

  def set_quant_config(self, name, config, configer, tensor_type='output', idx=0):
    assert len(config)==2, "expect 2 parameters, got " + str(config)
    if (tensor_type == 'output' and 
        name not in self._QuantInfo[tensor_type].keys()):
      name = configer.quant_output(name).name
    self._QuantInfo[tensor_type][name][idx][0] = config[0]
    self._QuantInfo[tensor_type][name][idx][1] = config[1]
    
  def insert_local_quant_config(self, name, config, tensor_type='output'):
    assert len(config)==2, "expect 2 parameters, got " + str(config)
    if name in self._QuantInfo[tensor_type].keys() and self._QuantInfo[tensor_type][name][0][1] is not None:
      return
    self._QuantInfo[tensor_type][name]=[config]
    
  def get_fix_position(self, name, configer, tensor_type='output', idx=0):
    if (tensor_type == 'output' and 
        name not in self._QuantInfo[tensor_type].keys()):
      name = configer.quant_output(name).name
    bnfp = copy.deepcopy(self._QuantInfo[tensor_type][name][idx])
    return bnfp[1] if bnfp is not None else None
  
  def set_fix_position(self, name, fp, configer, tensor_type='output', idx=0):
    if (tensor_type == 'output' and 
        name not in self._QuantInfo[tensor_type].keys()):
      name = configer.quant_output(name).name
    if self._QuantInfo[tensor_type][name][idx] is not None:
      self._QuantInfo[tensor_type][name][idx][1] = fp
  
class CPUGPUQConfigImp(QuantConfigImpBase):
  
  def make_quant_config(self, bitwidth):
    return [bitwidth, None, None, None] if bitwidth is not None else None 
      
  def get_quant_config(self, name, configer, real_value=True, tensor_type='output', idx=0):
    if (len(self._QuantInfo) == 0) or (tensor_type not in self._QuantInfo.keys()) or (len(self._QuantInfo[tensor_type]) == 0):
      return None
    
    if (tensor_type == 'output' and 
        name not in self._QuantInfo[tensor_type].keys()):
      name = configer.quant_output(name).name
    
    if name not in self._QuantInfo[tensor_type].keys():
      return None
    
    quant_config = copy.deepcopy(self._QuantInfo[tensor_type][name][idx])
    quant_mode = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_MODE)
    # get max quantized integer and 2^fix_pos
    if real_value:
      # BN layers are not quantized
      if quant_mode == 2 and quant_config[1] is None:
        print('Warning!!! The parameter/activation is not quantized: %s' % name)
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

  def set_quant_config(self, name, config, configer, tensor_type='output', idx=0):
    assert len(config)==4, "expect 4 parameters, got " + str(config)
    if (tensor_type == 'output' and 
        name not in self._QuantInfo[tensor_type].keys()):
      name = configer.quant_output(name).name
    self._QuantInfo[tensor_type][name][idx][0] = config[0]
    self._QuantInfo[tensor_type][name][idx][1] = config[1]
    self._QuantInfo[tensor_type][name][idx][2] = config[2]
    self._QuantInfo[tensor_type][name][idx][3] = config[3]
    
  def insert_local_quant_config(self, name, config, tensor_type='output'):
    assert len(config)==4, "expect 4 parameters, got " + str(config)
    if name in self._QuantInfo[tensor_type].keys() and self._QuantInfo[tensor_type][name][0][1] is not None:
      return
    self._QuantInfo[tensor_type][name] = [config]
