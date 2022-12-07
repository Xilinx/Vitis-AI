

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
from nndct_shared.quantization import QConfigBase
from nndct_shared.utils import NndctScreenLogger, QError, QWarning, QNote
#from pytorch_nndct.utils.nndct2torch_op_map import get_nndct_op_type

class TorchQConfig(QConfigBase):
    def __init__(self):
        super().__init__()
    
    def parse_bit_width(self, name, key, config_value, config_use):
        if isinstance(config_value, int) and (config_value >= 0  and config_value <= 32):
            config_use[key] = config_value
        else:
            NndctScreenLogger().error2user(QError.ILLEGAL_BITWIDTH, f"The {key} type of {name} should be int, and in range of [0,32].")
            exit(2)

class RNNTorchQConfig(QConfigBase):
    def __init__(self):
        super().__init__()
        self._default_qconfig['weights']['bit_width'] = 16
        self._default_qconfig['bias']['bit_width'] = 16
        self._default_qconfig['activation']['bit_width'] = 16
        
        self._qconfig['weights']['bit_width'] = 16
        self._qconfig['bias']['bit_width'] = 16
        self._qconfig['activation']['bit_width'] = 16
        
        self._legal_qconfigs['activation']['bit_width'] = [16]

    def parse_bit_width(self, name, key, config_value, config_use):
        if name == 'activation':
            if config_value in self._legal_qconfigs[name][key]:
                config_use[key] = config_value
            else:
                bitwidth_legels = self._legal_qconfigs[name][key]
                NndctScreenLogger().error2user(QError.ILLEGAL_BITWIDTH, f"The {key} configuration of {name} should be in the list {bitwidth_legels}.")
                exit(2)
        else:
            if isinstance(config_value, int) and (config_value >= 0  and config_value <= 32):
                config_use[key] = config_value
            else:
                NndctScreenLogger().error2user(QError.ILLEGAL_BITWIDTH, f"The {key} configuration of {name} type should be int, and in range of [0, 32].")
                exit(2)

