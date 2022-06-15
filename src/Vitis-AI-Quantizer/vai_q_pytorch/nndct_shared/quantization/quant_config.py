

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

from inspect import stack
from os import stat
import sys
import copy
import numpy as np
import json
from typing import Dict, List, Optional
from abc import ABCMeta, abstractmethod

from nndct_shared.utils import option_util, NndctScreenLogger

class QConfigBase(metaclass=ABCMeta):
    #_quant_tensor_types = ["weights", "bias", "activation", "input"]
    _default_qconfig = {
        "target_device": "DPU",
        "mix_bit": False,
        "quantizable_data_type": ["input", "weights", "bias", "activation"],
        "weights": {
            "bit_width": 8,
            "symmetric_mode": "symmetric",
            "signed": True,
            "method": "diffs",
            "round_method": "std_round",
            "quant_max": sys.maxsize,
            "quant_min": -sys.maxsize,
            "granularity": "per_tensor",
            "scale_type": "poweroftwo",
            "narrow_range": False
        },
        "bias": {
            "bit_width": 8,
            "symmetric_mode": "symmetric",
            "signed": True,
            "method": "diffs",
            "round_method": "std_round",
            "quant_max": sys.maxsize,
            "quant_min": -sys.maxsize,
            "granularity": "per_tensor",
            "scale_type": "poweroftwo",
            "narrow_range": False
        },
        "activation": {
            "bit_width": 8,
            "symmetric_mode": "symmetric",
            "signed": True,
            "method": "diffs",
            "round_method": "half_up",
            "quant_max": sys.maxsize,
            "quant_min": -sys.maxsize,
            "granularity": "per_tensor",
            "scale_type": "poweroftwo",
            "narrow_range": False,
            "calib_statistic_method": "modal"
        },
        "input": {
            "bit_width": 8,
            "symmetric_mode": "symmetric",
            "signed": True,
            "method": "diffs",
            "round_method": "std_round",
            "quant_max": sys.maxsize,
            "quant_min": -sys.maxsize,
            "granularity": "per_tensor",
            "scale_type": "poweroftwo",
            "narrow_range": False,
            "calib_statistic_method": "modal"
        },
        "layer_type_config": {},
        "layer_name_config": {}
    }
    _legal_qconfigs = {
        "target_device": ["DPU", "CPU", "GPU"],
        "mix_bit": [True, False],
        "weights": {
            "symmetric_mode": ["symmetric", "asymmetric"],
            "signed": [True, False],
            "method": ["maxmin", "entropy", "mse", "percentile", "diffs"],
            "round_method": ["half_even", "half_up", "half_down", "std_round"],
            "granularity": ["per_tensor", "per_channel"],
            "scale_type": ["poweroftwo", "float"],
            "narrow_range": [True, False]
        },
        "bias": {
            "symmetric_mode": ["symmetric", "asymmetric"],
            "signed": [True, False],
            "method": ["maxmin", "entropy", "mse", "percentile", "diffs"],
            "round_method": ["half_even", "half_up", "half_down", "std_round"],
            "granularity": ["per_tensor", "per_channel"],
            "scale_type": ["poweroftwo", "float"],
            "narrow_range": [True, False]
        },
        "activation": {
            "symmetric_mode": ["symmetric", "asymmetric"],
            "signed": [True, False],
            "method": ["maxmin", "entropy", "mse", "percentile", "diffs"],
            "round_method": ["half_even", "half_up", "half_down", "std_round"],
            "granularity": ["per_tensor", "per_channel"],
            "scale_type": ["poweroftwo", "float"],
            "narrow_range": [True, False],
            "calib_statistic_method": ["modal", "mean", "median", "max"],
        },
        "input": {
            "symmetric_mode": ["symmetric", "asymmetric"],
            "signed": [True, False],
            "method": ["maxmin", "entropy", "mse", "percentile", "diffs"],
            "round_method": ["half_even", "half_up", "half_down", "std_round"],
            "granularity": ["per_tensor", "per_channel"],
            "scale_type": ["poweroftwo", "float"],
            "narrow_range": [True, False],
            "calib_statistic_method": ["modal", "mean", "median", "max"]
        }
    }
    def __init__(self):
        self._qconfig = copy.deepcopy(self._default_qconfig)

    @abstractmethod
    def parse_bit_width(self, name, key, config_value):
        pass

    def parse_config_file(self, config_file: Optional[str], bit_width_w = None, bit_width_a = None, mix_bit = None):
        if config_file is None:
            NndctScreenLogger().info(f"Quant config file is empty, use default quant configuration")
            if bit_width_w:
                self._qconfig['weights']['bit_width'] = bit_width_w
                self._qconfig['bias']['bit_width'] = bit_width_w
            if bit_width_a:
                self._qconfig['activation']['bit_width'] = bit_width_a
                self._qconfig['input']['bit_width'] = bit_width_a
            if mix_bit:
                self._qconfig['mix_bit'] = mix_bit
            return
        
        with open(config_file, "r") as config_f:
            json_config = json.load(config_f)
        self._keywords_legel(json_config)
        self._nndct_switch_option(json_config)
        self._qconfig['target_device'] = json_config['target_device']
        self._qconfig['quantizable_data_type'] = json_config['quantizable_data_type']
        self.set_tensor_quant_config(json_config, self._qconfig)
        self.set_layer_quant_config(json_config, self._qconfig)
        
        if bit_width_w:
            if self._qconfig['weights']['bit_width'] != bit_width_w:
                NndctScreenLogger().warning(f"Bitwidth of weights in configuration file is different from that passed from torch_quantizer api, use the bitwidth in configuration file")
            if self._qconfig['bias']['bit_width'] != bit_width_w:
                NndctScreenLogger().warning(f"Bitwidth of bias in configuration file is different from that passed from torch_quantizer api, use the bitwidth in configuration file")
        if bit_width_a:
            if self._qconfig['activation']['bit_width'] != bit_width_a:
                NndctScreenLogger().warning(f"Bitwidth of activation in configuration file is different from that passed from torch_quantizer api, use the bitwidth in configuration file")
            if self._qconfig['input']['bit_width'] != bit_width_a:
                NndctScreenLogger().warning(f"Bitwidth of input in configuration file is different from that passed from torch_quantizer api, use the bitwidth in configuration file")
        if mix_bit:
            if self._qconfig['mix_bit'] != mix_bit:
                NndctScreenLogger().warning(f"Mix_bit parameter in configuration file is different from that passed from torch_quantizer api, use mix_bit parameter in configuration file")
            
        # if bit_width_w:
        #     self._qconfig['weights']['bit_width'] = bit_width_w
        #     self._qconfig['bias']['bit_width'] = bit_width_w
        # if bit_width_a:
        #     self._qconfig['activation']['bit_width'] = bit_width_a
        #     self._qconfig['input']['bit_width'] = bit_width_a
        # if mix_bit:
        #     self._qconfig['mix_bit'] = mix_bit
        
        # for name, config in json_config.items():
        #     if name == 'weights' or name == 'bias' or name == 'activation' or name == 'input':
        #         for key, key_config in config.items():
        #             if key == 'bit_width':
        #                 self.parse_bit_width(name, key, key_config)
        #             elif key in self._legal_qconfigs[name].keys():
        #                 if key_config in self._legal_qconfigs[name][key]:
        #                     self._qconfig[name][key] = key_config
        #                 else:
        #                     raise TypeError("The {key} configuration of {name} should be in the list {self._legal_qconfigs[name][key]}")
        #             else:
        #                 self._qconfig[name][key] = key_config
        #     else:
        #         if name in self._legal_qconfigs.keys():
        #             if config in self._legal_qconfigs[name]:
        #                 self._qconfig[name] = config
        #             else:
        #                 raise TypeError("The {name} configuration should be in the list {self._legal_qconfigs[name]}")
        #         else:
        #             self._qconfig[name] = config
        
        self._compute_q_maxmin()
        #self._qconfig_handle_conflict()
    
    @staticmethod
    def _keywords_legel(json_configs):
        model_config_keys = ["convert_relu6_to_relu", "include_cle", "keep_first_last_layer_accuracy", 
                            "keep_add_layer_accuracy", "include_bias_corr", "target_device",
                            "quantizable_data_type", "overall_quantize_config", "tensor_quantize_config",
                            "layer_quantize_config"]
        quant_param_keys = ["bit_width", "method", "round_mode", 
                            "symmetry", "per_channel", "signed", 
                            "narrow_range", "scale_type", "calib_statistic_method",
                            "percentage"]
        tensor_keys = ["input", "weights", "bias", "activation"]
        layer_quant_keys = ["layer_type", "layer_name", "quantizable_data_type",
                            "overall_quantize_config", "tensor_quantize_config"]
        for key_word, _ in json_configs.items():
            if key_word not in model_config_keys:
                NndctScreenLogger().error(f"Unsupported keyword in quantization config: '{key_word}'. ")
                #raise ValueError("Unsupported keyword {key_word}")
                exit(2)
                
            if key_word == "overall_quantize_config":
                overall_config = json_configs["overall_quantize_config"]
                for key1_word, _ in overall_config.items():
                    if key1_word not in quant_param_keys:
                        NndctScreenLogger().error(f"Unsupported keyword in quantization config: '{key1_word}'.")
                        #raise ValueError("Unsupported keyword {key1_word}")
                        exit(2)
            
            if key_word == "tensor_quantize_config":
                tensor_config = json_configs["tensor_quantize_config"]
                for key1_word, _ in tensor_config.items():
                    if key1_word not in tensor_keys:
                        NndctScreenLogger().error(f"Unsupported keyword in quantization config: '{key1_word}'.")
                        #raise ValueError("Unsupported keyword {key1_word}")
                        exit(2)
                        
                    tensor_overall_config = tensor_config[key1_word]
                    for key2_word, _ in tensor_overall_config.items():
                        if key2_word not in quant_param_keys:
                            NndctScreenLogger().error(f"Unsupported keyword in quantization config: '{key2_word}'.")
                            #raise ValueError("Unsupported keyword {key2_word}")
                            exit(2)
            
            if key_word == "layer_quantize_config":
                layer_configs = json_configs["layer_quantize_config"]
                for layer_config in layer_configs:
                    for key_word1, _ in layer_config.items():
                        if key_word1 not in layer_quant_keys:
                            NndctScreenLogger().error(f"Unsupported keyword in quantization config: '{key_word1}'.")
                            #raise ValueError("Unsupported keyword {key_word1}")
                            exit(2)
                        
                        if key_word1 == "overall_quantize_config":
                            overall_config = layer_config["overall_quantize_config"]
                            for key1_word, _ in overall_config.items():
                                if key1_word not in quant_param_keys:
                                    NndctScreenLogger().error(f"Unsupported keyword in quantization config: '{key1_word}'.")
                                    #raise ValueError("Unsupported keyword {key1_word}")
                                    exit(2)
                                    
                        if key_word1 == "tensor_quantize_config":
                            tensor_config = layer_config["tensor_quantize_config"]
                            for key1_word, _ in tensor_config.items():
                                if key1_word not in tensor_keys:
                                    NndctScreenLogger().error(f"Unsupported keyword in quantization config: '{key1_word}'.")
                                    #raise ValueError("Unsupported keyword {key1_word}")
                                    exit(2)
                                    
                                tensor_overall_config = tensor_config[key1_word]
                                for key2_word, _ in tensor_overall_config.items():
                                    if key2_word not in quant_param_keys:
                                        NndctScreenLogger().error(f"Unsupported keyword in quantization config: '{key2_word}'.")
                                        #raise ValueError("Unsupported keyword {key2_word}")
                                        exit(2)
            
                
    @staticmethod
    def _nndct_switch_option(json_configs):
        if (json_configs.get('convert_relu6_to_relu', None) is not None) and isinstance(json_configs['convert_relu6_to_relu'], bool):
            option_util.set_option_value("nndct_convert_relu6_to_relu", json_configs['convert_relu6_to_relu'])
        if (json_configs.get('include_cle', None) is not None) and isinstance(json_configs['include_cle'], bool):
            option_util.set_option_value("nndct_equalization", json_configs['include_cle'])
        if (json_configs.get('include_bias_corr', None) is not None) and isinstance(json_configs['include_bias_corr'], bool):
            option_util.set_option_value("nndct_param_corr", json_configs['include_bias_corr'])
        if (json_configs.get('keep_first_last_layer_accuracy', None) is not None) and isinstance(json_configs['keep_first_last_layer_accuracy'], bool):
            if json_configs['keep_first_last_layer_accuracy']:
                NndctScreenLogger().info(f"keep_first_last_layer_accuracy feature will be supported in next version")
            else:
                option_util.set_option_value("nndct_keep_first_last_layer_accuracy", json_configs['keep_first_last_layer_accuracy'])

        if (json_configs.get('keep_add_layer_accuracy', None) is not None) and isinstance(json_configs['keep_add_layer_accuracy'], bool):
            if json_configs['keep_add_layer_accuracy']:
                NndctScreenLogger().info(f"keep_add_layer_accuracy feature will be supported in next version")
            else:
                option_util.set_option_value("nndct_keep_add_layer_accuracy", json_configs['keep_add_layer_accuracy'])
    
    def set_tensor_quant_config(self, json_configs, q_config):
        if q_config is None:
            q_config = {}
        overall_quant_config = json_configs.get("overall_quantize_config", None)
        if overall_quant_config:
            for tensor_type in json_configs["quantizable_data_type"]:
                self._map_tensor_type_config(tensor_type, overall_quant_config, q_config)
        tensor_quant_configs = json_configs.get("tensor_quantize_config", None)
        if tensor_quant_configs:
            for tensor_type, tensor_config in tensor_quant_configs.items():
                self._map_tensor_type_config(tensor_type, tensor_config, q_config)
    
    def set_layer_quant_config(self, json_configs, q_config):
        layer_quant_configs = json_configs.get("layer_quantize_config", None)
        if layer_quant_configs:
            for layer_config in layer_quant_configs:
                if layer_config.get('layer_type', None):
                    torch_layer_type = layer_config.get('layer_type').split('.')[-1]
                    #nndct_layer_type = get_nndct_op_type(torch_layer_type)
                    layer_tensor_types = layer_config.get('quantizable_data_type')
                    nndct_layer_config = {'quantizable_data_type':layer_tensor_types}
                    for tensor_type in nndct_layer_config['quantizable_data_type']:
                        nndct_layer_config[tensor_type] = copy.deepcopy(self._default_qconfig[tensor_type])
                    
                    self.set_tensor_quant_config(layer_config, nndct_layer_config)
                    #q_config['layer_type_config'].append(nndct_layer_config)
                    q_config['layer_type_config'][torch_layer_type] = nndct_layer_config
                elif layer_config.get('layer_name', None):
                    nndct_layer_name = layer_config.get('layer_name')
                    layer_tensor_types = layer_config.get('quantizable_data_type')
                    nndct_layer_config = {'quantizable_data_type':layer_tensor_types}
                    for tensor_type in nndct_layer_config['quantizable_data_type']:
                        nndct_layer_config[tensor_type] = copy.deepcopy(self._default_qconfig[tensor_type])
                    self.set_tensor_quant_config(layer_config, nndct_layer_config)
                    #q_config['layer_type_config'].append(nndct_layer_config)
                    q_config['layer_name_config'][nndct_layer_name] = nndct_layer_config
    
    @staticmethod
    def _map_tensor_type_config(tensor_type, export_config, q_config):
        if q_config.get(tensor_type, None) is None:
            q_config[tensor_type] = {}
        config_to_use = q_config[tensor_type]
        q_config[tensor_type] = QConfigBase._generate_config_from_export(tensor_type, export_config, config_to_use)
    
    @staticmethod
    def _generate_config_from_export(tensor_type, export_config, config_to_use):
        if export_config.get('bit_width', None) and isinstance(export_config.get('bit_width', None), int):
            config_to_use['bit_width'] = export_config.get('bit_width')
        if (export_config.get('method', None)) and \
            (export_config['method'] in QConfigBase._legal_qconfigs[tensor_type]['method']):
            config_to_use['method'] = export_config.get('method')        
        if (export_config.get('round_mode', None)) and \
            (export_config['round_mode'] in QConfigBase._legal_qconfigs[tensor_type]['round_method']):
            config_to_use['round_method'] = export_config.get('round_mode')
        if (export_config.get('scale_type', None)) and \
            (export_config['scale_type'] in QConfigBase._legal_qconfigs[tensor_type]['scale_type']):
            config_to_use['scale_type'] = export_config.get('scale_type')
        if (not export_config.get('symmetry', None) is None):
            config_to_use['symmetric_mode'] = 'symmetric' if export_config.get('symmetry') else 'asymmetric'
        if (not export_config.get('per_channel', None) is None):
            config_to_use['granularity'] = 'per_channel' if export_config.get('per_channel') else 'per_tensor'
        if (not export_config.get('signed', None) is None):
            config_to_use['signed'] = export_config.get('signed')
        if (not export_config.get('narrow_range', None) is None):
            config_to_use['narrow_range'] = export_config.get('narrow_range')

        if tensor_type in ['activation', 'input']:
            if (export_config.get('calib_statistic_method', None)) and \
                (export_config['calib_statistic_method'] in QConfigBase._legal_qconfigs[tensor_type]['calib_statistic_method']):
                config_to_use['calib_statistic_method'] = export_config.get('calib_statistic_method')
        
        if config_to_use.get('method') == 'percentile':
            if export_config.get('percentage', None) is None:
                config_to_use['percentage'] = 99.99
            else:
                if ((not isinstance(export_config.get('percentage'), float)) or \
                    (export_config['percentage'] <= 0.0) or \
                    (export_config['percentage'] > 100.0)):
                    raise TypeError("Percentage should be larger than 0.0 and smaller than 100.0")
                else:
                    config_to_use['percentage'] = export_config['percentage']
                
        return config_to_use

    def _compute_q_maxmin(self):
        for tensor_type in self._qconfig['quantizable_data_type']:
            self._qconfig[tensor_type]['quant_max'] = self._reset_range(self._qconfig[tensor_type]['quant_max'],
                                                                        self._qconfig[tensor_type]['bit_width'],
                                                                        self._qconfig[tensor_type]['symmetric_mode'],
                                                                        self._qconfig[tensor_type]['signed'],
                                                                        self._qconfig[tensor_type]['narrow_range'])
            self._qconfig[tensor_type]['quant_min'] = self._reset_range(self._qconfig[tensor_type]['quant_min'],
                                                                        self._qconfig[tensor_type]['bit_width'],
                                                                        self._qconfig[tensor_type]['symmetric_mode'],
                                                                        self._qconfig[tensor_type]['signed'],
                                                                        self._qconfig[tensor_type]['narrow_range'])
        #for i in range(len(self._qconfig['layer_type_config'])):
        for layer_type, layer_quant_config in self._qconfig['layer_type_config'].items():
            #layer_quant_config = self._qconfig['layer_type_config'][i]
            for tensor_type in layer_quant_config['quantizable_data_type']:
                if layer_quant_config[tensor_type].get('quant_max', None) is None:
                    layer_quant_config[tensor_type]['quant_max'] = sys.maxsize
                self._qconfig['layer_type_config'][layer_type][tensor_type]['quant_max'] = self._reset_range(layer_quant_config[tensor_type]['quant_max'],
                                                                                                             layer_quant_config[tensor_type]['bit_width'],
                                                                                                             layer_quant_config[tensor_type]['symmetric_mode'],
                                                                                                             layer_quant_config[tensor_type]['signed'],
                                                                                                             layer_quant_config[tensor_type]['narrow_range'])
                if layer_quant_config[tensor_type].get('quant_min', None) is None:
                    layer_quant_config[tensor_type]['quant_min'] = -sys.maxsize
                self._qconfig['layer_type_config'][layer_type][tensor_type]['quant_min'] = self._reset_range(layer_quant_config[tensor_type]['quant_min'],
                                                                                                             layer_quant_config[tensor_type]['bit_width'],
                                                                                                             layer_quant_config[tensor_type]['symmetric_mode'],
                                                                                                             layer_quant_config[tensor_type]['signed'],
                                                                                                             layer_quant_config[tensor_type]['narrow_range'])
        for layer_name, layer_quant_config in self._qconfig['layer_name_config'].items():
            #layer_quant_config = self._qconfig['layer_type_config'][i]
            for tensor_type in layer_quant_config['quantizable_data_type']:
                if layer_quant_config[tensor_type].get('quant_max', None) is None:
                    layer_quant_config[tensor_type]['quant_max'] = sys.maxsize
                self._qconfig['layer_name_config'][layer_name][tensor_type]['quant_max'] = self._reset_range(layer_quant_config[tensor_type]['quant_max'],
                                                                                                             layer_quant_config[tensor_type]['bit_width'],
                                                                                                             layer_quant_config[tensor_type]['symmetric_mode'],
                                                                                                             layer_quant_config[tensor_type]['signed'],
                                                                                                             layer_quant_config[tensor_type]['narrow_range'])
                if layer_quant_config[tensor_type].get('quant_min', None) is None:
                    layer_quant_config[tensor_type]['quant_min'] = -sys.maxsize
                self._qconfig['layer_name_config'][layer_name][tensor_type]['quant_min'] = self._reset_range(layer_quant_config[tensor_type]['quant_min'],
                                                                                                             layer_quant_config[tensor_type]['bit_width'],
                                                                                                             layer_quant_config[tensor_type]['symmetric_mode'],
                                                                                                             layer_quant_config[tensor_type]['signed'],
                                                                                                             layer_quant_config[tensor_type]['narrow_range'])

    # def _qconfig_handle_conflict(self):
    #     if self._qconfig['mix_bit']:
    #         if (self._qconfig['weights']['bit_width'] != self._qconfig['bias']['bit_width'] or  
    #             self._qconfig['weights']['bit_width'] != self._qconfig['activation']['bit_width']):
    #             raise TypeError("The bit_width configurations are conflict with the mix_bit configuration")
    #     if self._qconfig['weights']['method'] == 'percentile':
    #         if ((self._qconfig['weights'].get('percentage') == None) or 
    #             (not isinstance(self._qconfig['weights'].get('percentage'), float)) or 
    #             (self._qconfig['weights']['percentage'] <= 0.0) or 
    #             (self._qconfig['weights']['percentage'] > 100.0)):
    #             raise TypeError("Percentage should be larger than 0.0 and smaller than 100.0")
    #     if self._qconfig['bias']['method'] == 'percentile':
    #         if ((self._qconfig['bias'].get('percentage') == None) or 
    #             (not isinstance(self._qconfig['bias'].get('percentage'), float)) or 
    #             (self._qconfig['bias']['percentage'] <= 0.0) or 
    #             (self._qconfig['bias']['percentage'] > 100.0)):
    #             raise TypeError("Percentage should be larger than 0.0 and smaller than 100.0")
    #     if self._qconfig['activation']['method'] == 'percentile':
    #         if ((self._qconfig['activation'].get('percentage') == None) or 
    #             (not isinstance(self._qconfig['activation'].get('percentage'), float)) or 
    #             (self._qconfig['activation']['percentage'] <= 0.0) or 
    #             (self._qconfig['activation']['percentage'] > 100.0)):
    #             raise TypeError("Percentage should be larger than 0.0 and smaller than 100.0")
    #     if self._qconfig['input']['method'] == 'percentile':
    #         if ((self._qconfig['input'].get('percentage') == None) or 
    #             (not isinstance(self._qconfig['input'].get('percentage'), float)) or 
    #             (self._qconfig['input']['percentage'] <= 0.0) or 
    #             (self._qconfig['input']['percentage'] > 100.0)):
    #             raise TypeError("Percentage should be larger than 0.0 and smaller than 100.0")
    
    @staticmethod
    def _reset_range(qrange, num_bit, symmetric_mode, signed, narrow_range):
        if qrange == -sys.maxsize:
            if signed and symmetric_mode == 'symmetric':
                if not narrow_range:
                    qrange = -int(2 ** (num_bit - 1))
                else:
                    qrange = -int(2 ** (num_bit - 1)) + 1
            else:
                qrange = 0
        elif qrange == sys.maxsize:
            if signed and symmetric_mode == 'symmetric':
                qrange = int(2 ** (num_bit - 1)) - 1
            else:
                qrange = int(2 ** num_bit) - 1
        
        return qrange
        
    @property
    def qconfig(self):
        return self._qconfig
 
