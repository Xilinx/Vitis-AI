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
import collections
import json

class ConfigDict(object):

  def __init__(self, params=None):
    if params:
      self.load(params, is_strict=False)

  def load(self, params):
    for k, v in params.items():
      if k not in self.__dict__.keys():
        continue
      if isinstance(v, dict):
        self.__dict__[k].load(v)
      else:
        self.__dict__[k] = copy.deepcopy(v)

  def _set(self, k, v):
    if isinstance(v, dict):
      self.__dict__[k] = ConfigDict(v)
    else:
      self.__dict__[k] = copy.deepcopy(v)

  def __setattr__(self, k, v):
    """Sets the value of the existing key.

    Note that this does not allow directly defining a new key. Use the
    `override` method with `is_strict=False` instead.

    Args:
      k: the key string.
      v: the value to be used to set the key `k`.

    Raises:
      KeyError: if k is not defined in the ConfigDict.
    """
    self._set(k, v)

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

  def values(self):
    key_values = {}
    for k, v in self.__dict__.items():
      if isinstance(v, ConfigDict):
        key_values[k] = v.as_dict()
      else:
        key_values[k] = copy.deepcopy(v)
    return key_values

  def __str__(self):
    return '{}'.format(self.values())

class RuntimeConfig(ConfigDict):

  def __init__(self, params=None):
    self.data_format = 'fp32'
    self.bfp_bitwidth = 16
    self.bfp_tile_size = 8
    self.round_mode = 'round_even'
    self.approx_mode = 'exp_poly'
    self.approx_degree = 3
    self.exp_table_size = 1
    self.training = False
    self.load(params)

  @classmethod
  def from_json(cls, path):
    with open(path, 'r') as f:
      return cls(json.load(f))

class LayerRuntimeSpec(object):

  def __init__(self, config=None):
    self._input_quantizers = []
    self._output_quantizers = []
    self._weight_quantizers = {}

    self._config = config

  def add_weight_quantizer(self, name, quantizer):
    self._weight_quantizers[name] = quantizer

  def add_input_quantizer(self, quantizer):
    self._input_quantizers.append(quantizer)

  def add_output_quantizer(self, quantizer):
    self._output_quantizers.append(quantizer)

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
