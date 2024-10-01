
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

from nndct_shared.quantization import QConfigBase

class RNNTFQConfig(QConfigBase):

  def __init__(self):
    super().__init__()
    self._default_qconfig['weights']['bit_width'] = 16
    self._default_qconfig['bias']['bit_width'] = 16
    self._default_qconfig['activation']['bit_width'] = 16

    self._qconfig['weights']['bit_width'] = 16
    self._qconfig['bias']['bit_width'] = 16
    self._qconfig['activation']['bit_width'] = 16

    self._legal_qconfigs['activation']['bit_width'] = [16]

  def parse_bit_width(self, name, key, config_value):
    if name == 'activation':
      if config_value in self._legal_qconfigs[name][key]:
        self._qconfig[name][key] = config_value
      else:
        raise TypeError(
            "The {key} configuration of {name} should be in the list {self._legal_qconfigs[name][key]}"
        )
    else:
      if isinstance(config_value, int) and (config_value >= 0 and
                                            config_value <= 32):
        self._qconfig[name][key] = config_value
      else:
        raise TypeError(
            "The {key} configuration of {name} type should be int, and in range of [0, 32]"
        )
