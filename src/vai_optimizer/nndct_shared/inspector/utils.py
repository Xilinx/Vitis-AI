
from collections import defaultdict
from nndct_shared.utils import NndctDebugLogger, NndctOption

class TensorwiseQuantInfo(object):
  def __init__(self, bw, fp=None) -> None:
    self._bw = bw
    self._fp = fp

  def get_bw_fp(self):
    return self._bw, self._fp
  
class QuantConfig(object):
  def __init__(self, bw, fake_fp=None) -> None:
    self._quant_config = {"param": {}, "output" : defaultdict(list), "input": defaultdict(list),}
    self._output_quant_info_cls = TensorwiseQuantInfo
    self._param_quant_info_cls = TensorwiseQuantInfo
    self._input_quant_info_cls = TensorwiseQuantInfo
    self._bw = bw
    self._fp = fake_fp

  def insert_output_quant_fp(self, node_name, fp=None):
    # bw, fp = quant_info.get_bw_fp()
    fp = fp if fp is not None else self._fp
    self._quant_config["output"][node_name].append(self._output_quant_info_cls(self._bw, fp))
  
  def set_output_quant_fp_at(self, node_name, offset, fp=None):
    # bw, fp = quant_info.get_bw_fp()
    fp = fp if fp is not None else self._fp
    self._quant_config["output"][node_name][offset] = self._output_quant_info_cls(self._bw, fp)
  

  def get_output_quant_info(self, node_name):
    return self._quant_config["output"][node_name]
  
  def get_input_quant_info(self, node_name):
    return self._quant_config["input"][node_name]

  def get_param_quant_info(self, param_name):
    return self._quant_config["param"][param_name]

  def get_output_bw_fp_at(self, node_name, offset):
    if node_name not in self._quant_config["output"]:
      return None
    return  self._quant_config["output"][node_name][offset].get_bw_fp()

  def get_output_bw_fp(self, node_name):
    if node_name not in self._quant_config["output"]:
      return None
    return  self._quant_config["output"][node_name][0].get_bw_fp()

  def get_input_bw_fp_at(self, node_name, offset):
    if node_name not in self._quant_config["input"]:
      return None
    return  self._quant_config["input"][node_name][offset].get_bw_fp()

  def get_param_bw_fp(self, node_name):
    if node_name not in self._quant_config["param"]:
      return None
    return  self._quant_config["param"][node_name].get_bw_fp()

  
  def insert_param_quant_fp(self, param_name, fp=None):
    # bw, fp = quant_info.get_bw_fp()
    fp = fp if fp is not None else self._fp
    if param_name not in self._quant_config["param"]:
      self._quant_config["param"][param_name] = self._param_quant_info_cls(self._bw, fp)

  def insert_input_quant_fp(self, node_name, fp=None):
    # bw, fp = quant_info.get_bw_fp()
    fp = fp if fp is not None else self._fp
    self._quant_config["input"][node_name].append(self._input_quant_info_cls(self._bw, fp))
  
  def set_input_quant_fp_at(self, node_name, offset, fp=None):
    # bw, fp = quant_info.get_bw_fp()
    fp = fp if fp is not None else self._fp
    self._quant_config["input"][node_name][offset] = self._input_quant_info_cls(self._bw, fp)

  def get_fake_fp(self):
    return self._fp

  def set_fake_fp(self, fp):
    self._fp = fp
  
  def get_bw(self):
    return self._bw

  def get_output_keys(self):
    return self._quant_config["output"].keys()
  
  def get_input_keys(self):
    return self._quant_config["input"].keys()

  
  def get_param_keys(self):
    return self._quant_config["param"].keys()

  def remove_output_fp(self, node_name):
    self._quant_config["output"].pop(node_name, None)
      



def convert_quant_config_to_dict(quant_config, init=False):
  config = {'param': defaultdict(list), 'output': defaultdict(list), 'input': defaultdict(list)}
  for key in quant_config.get_output_keys():
    for quant_info in quant_config.get_output_quant_info(key):
      bw, fp = quant_info.get_bw_fp()
      if init is True:
        fp = None
      config["output"][key].append([bw, fp])
  
  for key in quant_config.get_input_keys():
    for quant_info in quant_config.get_input_quant_info(key):
      bw, fp = quant_info.get_bw_fp()
      if init is True:
        fp = None
      config["input"][key].append([bw, fp])

  for key in quant_config.get_param_keys():
    quant_info = quant_config.get_param_quant_info(key)
    bw, fp = quant_info.get_bw_fp()
    if init is True:
      fp = None
    config["param"][key].append([bw, fp])
  return config


def build_xir_nndct_op_map():
  from nndct_shared.compile.xop_creator import NNDCTIR2XIR_CONVERTOR
  supported_nndct = []
  xir2nndct = defaultdict(set)
  for nndct_op_type, (xir_op_type, _) in NNDCTIR2XIR_CONVERTOR.items():
    xir2nndct[xir_op_type].add(nndct_op_type)
    supported_nndct.append(nndct_op_type)
  return supported_nndct, xir2nndct


def log_debug_info(msg):
  if NndctOption.nndct_inspect_debug.value:
    NndctDebugLogger.write(f"{msg}\n")
