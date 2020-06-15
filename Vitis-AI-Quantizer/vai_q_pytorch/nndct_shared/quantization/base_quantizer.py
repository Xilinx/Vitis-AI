import copy
import json

import numpy as np

from nndct_shared import utils as nndct_utils
from nndct_shared.base import NNDCT_KEYS, NNDCT_OP
from nndct_shared.utils import NndctScreenLogger

from .utils import QuantInfoConfiger, check_quant_config, get_amp_bnfps


class BaseQuantizer():

  def __init__(self, quant_mode: int, output_dir: str, bitwidth_w: int,
               bitwidth_a: int):

    # initialization
    self.quant_mode = quant_mode
    self.bitwidth_w = bitwidth_w
    self.bitwidth_a = bitwidth_a
    self.export_file = '/'.join([output_dir, 'quant_info.json'])
    self.quant_file = '/'.join([output_dir, 'quant_info.json'])
    self.quant_commands = None
    self.quant_table_file = None
    self.keep_all_graph_info = True
    self.platform_type = 'nndct'

  def setup(self, nndct_graph, lstm=False):

    self.Nndctgraph = nndct_graph
    self.lstm = lstm
    self.calibration_method = 'DiffS'

    # further setup
    if self.quant_mode > 0:
      model_type = self.get_model_type()
      self._configer = QuantInfoConfiger(
          graph_or_file=self.Nndctgraph, model_type=model_type)
      quant_groups = self._configer.get_info(self.quant_commands)
      self._configer.load_quant_info(quant_groups)
      self.quant_opt = {
          'range': 5,
          'round_method': 2,
      }
      self.__QuantInfo = self._configer.fill_value(
          self.bitwidth_w, self.bitwidth_a, lstm=lstm)
      self.__QuantInfo['input_blobs'] = {}

      # calibration and quantization awared training mode
      if self.quant_mode in [1, 3]:
        self.init_scan_config()
      if self.quant_mode > 1:
        self.init_quant_config()

  @property
  def configer(self):
    return self._configer

  @nndct_utils.not_implement
  def do_scan(self, res, max, min, name, node, tensor_type='input'):
    pass

  @nndct_utils.not_implement
  def do_quantize(self, blob, name, node, tensor_type='input'):
    pass

  def init_scan_config(self):
    self.__MaxMins = {}
    for item in self.__QuantInfo['params']:
      self.__MaxMins[item] = [None, None]
    for item in self.__QuantInfo['blobs']:
      if self.calibration_method == 'DiffS':
        # record the fragpos
        self.__MaxMins[item] = []

  def get_bnfp(self, name, real_value=True):
    collection, name = self.__get_config_name(name)
    bnfp = copy.deepcopy(self.__QuantInfo[collection][name])
    # get max quantized integer and 2^fix_pos
    if real_value:
      # BN layers are not quantized
      if self.quant_mode == 2 and bnfp[1] is None:
        print('Warning!!! The parameter/activation is not quantized: %s' % name)
        bnfp[0] = 65536 * 1024
        bnfp[1] = 4096
        return bnfp
      try:
        bnfp = get_amp_bnfps(bnfp)
      except OverflowError as e:
        print("fragpos of {} : {}".format(name, repr(e)))
    return bnfp

  def set_bnfp(self, name, bnfp):
    collection, name = self.__get_config_name(name)
    self.__QuantInfo[collection][name][0] = bnfp[0]
    self.__QuantInfo[collection][name][1] = bnfp[1]

  def get_tensor_des(self, tensor):
    return str(tensor)

  def init_quant_config(self, config=None):
    config = config or self.quant_file
    self.__QuantInfo = nndct_utils.load_json_obj(config)
    check_quant_config(self.__QuantInfo)
    self.__Params = self.__QuantInfo['params']
    self.__QuantTable = {}
    if self.quant_table_file:
      with open(self.quant_table_file, 'r') as f:
        table_conf = json.load(f)
      for item in table_conf:
        self.__QuantTable[item] = {}
        self.__QuantTable[item]['data'] = np.array(
            table_conf[item]['data']).astype(np.int)
        self.__QuantTable[item]['bitwidth'] = table_conf[item]['Expo'] + 1
        self.__QuantTable[item]['fragpos'] = table_conf[item]['frag_pos']

  def __get_config_name(self, name):

    def __find_name(coll, name):
      if name in self.__QuantInfo[coll]:
        return coll, name
      for n in self.__QuantInfo[coll]:
        if name == n or name.endswith('/' + n):
          return coll, n
      return None

    res = None
    for k in ['params', 'blobs']:
      res = __find_name(k, name)
      if res:
        break
    if not res:
      for k, v in self.__QuantInfo['input_blobs'].items():
        if all(c.name in v for c in self.Nndctgraph.children(name)):
          res = 'blobs', k
    if not res:
      # for debug use only
      name = nndct_utils.remove_trans_scp_prefix(name, self.configer.name_scp)
      res = __find_name('blobs', self.configer.quant_end_node(name).name)
    if not res:
      raise KeyError("{} do not exist in Config, please check!".format(name))
    return res

  def __update_item_maxmin(self, maxmin, mode, val, force_reset=False):
    if mode == NNDCT_KEYS.DIFFS_SCAN_SUFFIX.replace(NNDCT_KEYS.SUFFIX_CONNECT,
                                                    ''):
      maxmin.append(val)
    else:
      idx = {
          NNDCT_KEYS.MAX_SCAN_SUFFIX.replace(NNDCT_KEYS.SUFFIX_CONNECT, ''): 0,
          NNDCT_KEYS.MIN_SCAN_SUFFIX.replace(NNDCT_KEYS.SUFFIX_CONNECT, ''): 1,
      }[mode]
      if maxmin[idx] is None or force_reset:
        maxmin[idx] = [val]
      else:
        maxmin[idx].append(val)

  @property
  def maxmins(self):
    return self.__MaxMins

  @property
  def quant_config(self):
    return self.__QuantInfo

  @property
  def quant_table(self):
    return self.__QuantTable

  @property
  def bitw(self):
    return self.bitwidth_w

  @property
  def bita(self):
    return self.bitwidth_a

