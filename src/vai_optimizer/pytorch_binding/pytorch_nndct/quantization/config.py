# Copyright 2022 Xilinx Inc.
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

import copy
import json
import yaml

from typing import Union

class ConfigDict(object):

  def __init__(self, params=None):
    if params:
      self.load(params)

  def load(self, params):
    for k, v in params.items():
      if k not in self.__dict__.keys():
        continue
      if isinstance(v, dict):
        self.__dict__[k].load(v)
      else:
        self.__dict__[k] = copy.deepcopy(v)

  def __setattr__(self, k, v):
    """Sets the value of the existing key.

    Args:
      k: the key string.
      v: the value to be used to set the key `k`.

    Raises:
      KeyError: if k is not defined in the ConfigDict.
    """
    if isinstance(v, dict):
      self.__dict__[k] = ConfigDict(v)
    else:
      self.__dict__[k] = copy.deepcopy(v)

  def __getattr__(self, k):
    """Gets the value of the existing key.

    Args:
      k: the key string.

    Returns:
      the value of the key.

    Raises:
      AttributeError: if k is not defined in the ConfigDict.
    """

    if k not in self.__dict__.keys():
      raise AttributeError('The key `{}` does not exist. '.format(k))
    return self.__dict__[k]

  def __contains__(self, key):
    """Implements the membership test operator."""
    return key in self.__dict__

  def get(self, key, value=None):
    """Accesses through built-in dictionary get method."""
    return self.__dict__.get(key, value)

  def as_dict(self):
    key_values = {}
    for k, v in self.__dict__.items():
      if isinstance(v, ConfigDict):
        key_values[k] = v.as_dict()
      else:
        key_values[k] = copy.deepcopy(v)
    return key_values

  def __str__(self):
    return '{}'.format(self.as_dict())

class Config(ConfigDict):
  @classmethod
  def from_json(cls, path):
    with open(path, 'r') as f:
      return cls(json.load(f))

  @classmethod
  def from_yaml(cls, path):
    with open(path, 'r') as f:
      return cls(yaml.load(f, Loader=yaml.FullLoader))

class RuntimeConfig(Config):

  def __init__(self, params=None):
    self.bfp: BFPConfig = BFPConfig()
    self.non_linear_approx: NonLinearApproxConfig = NonLinearApproxConfig()

    super(RuntimeConfig, self).__init__(params)

  @classmethod
  def mx6(cls):
    cfg = cls()
    cfg.bfp.bitwidth = 13
    cfg.bfp.block_size = 16
    cfg.bfp.prime.mode = PrimeModes.SHARED
    cfg.bfp.prime.sub_block_size = 2
    cfg.bfp.prime.sub_block_shift_bits = 1
    cfg.bfp.rounding_mode = 'round_to_nearest'
    return cfg

  @classmethod
  def mx9(cls):
    cfg = cls.mx6()
    cfg.bfp.bitwidth = 16
    return cfg

class BFPConfig(Config):
  def __init__(self, params=None):
    self.bitwidth: int = 16
    self.block_size: int = 8
    self.rounding_mode: str = 'round_to_nearest'
    self.prime: BFPPrimeConfig = BFPPrimeConfig()

    super(BFPConfig, self).__init__(params)

class BFPPrimeConfig(Config):
  def __init__(self, params=None):
    self.mode: PrimeModes = None
    self.sub_block_size: int = 2
    self.sub_block_shift_bits: int = 1

    super(BFPPrimeConfig, self).__init__(params)

class PrimeModes(object):
  NORMAL = 'normal'
  SHARED = 'shared'

class NonLinearApproxConfig(Config):
  def __init__(self, params=None):
    self.mode: str = 'no_approx'
    self.degree: int = 3
    self.exp_table_size: int = 1

    super(NonLinearApproxConfig, self).__init__(params)

class FPConfig(Config):
  def __init__(self, params=None):
    self.exp_bits: int = 5
    self.exp_bias_mode: str = 'ieee'
    self.exp_bias: Union[None, int] = None

    super(FPConfig, self).__init__(params)

mx6 = RuntimeConfig.mx6
mx9 = RuntimeConfig.mx9

def get(identifier):
  globs = globals()
  if identifier not in globs:
    raise ValueError(f'Unknown dtype: {identifier}')
  return globs[identifier]()

class LayerRuntimeSpec(object):

  def __init__(self, config=None):
    self._input_quantizers = []
    self._output_quantizers = []
    self._weight_quantizers = {}

    self._config: RuntimeConfig = config

  def add_weight_quantizer(self, name, quantizer):
    self._weight_quantizers[name] = quantizer

  def add_input_quantizer(self, quantizer):
    self._input_quantizers.append(quantizer)

  def add_output_quantizer(self, quantizer):
    self._output_quantizers.append(quantizer)

  def maybe_get_weight_quantizer(self, name):
    if name not in self._weight_quantizers:
      return None
    return self._weight_quantizers[name]

  def get_weight_quantizer(self, name):
    if name not in self._weight_quantizers:
      raise ValueError('No quantizer for given weight name "{}"'.format(name))
    return self._weight_quantizers[name]

  @property
  def weight_quantizers(self):
    return [(name, quantizer)
            for name, quantizer in self._weight_quantizers.items()]

  @property
  def input_quantizers(self):
    return self._input_quantizers

  @property
  def output_quantizers(self):
    return self._output_quantizers

  @property
  def config(self):
    return self._config

  @config.setter
  def config(self, value):
    self._config = value

  def __repr__(self):
    return ('LayerRuntimeSpec(input_quantizers={}, output_quantizers={}, '
        'weight_quantiers={}, config={})').format(
        self._input_quantizers, self._output_quantizers,
        self._weight_quantizers, self._config)
