

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

from nndct_shared.utils import option_util, NndctScreenLogger, QError, QWarning

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
    def parse_bit_width(self, name, key, config_value, config_use):
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
        #self._qconfig['quantizable_data_type'] = json_config['quantizable_data_type']
        self.set_tensor_quant_config(json_config, self._qconfig)
        self.set_layer_quant_config(json_config, self._qconfig)
        
        if bit_width_w:
            if self._qconfig['weights']['bit_width'] != bit_width_w:
                NndctScreenLogger().warning2user(QWarning.BITWIDTH_MISMATCH, f"Bitwidth of weights in configuration file is different from that passed from torch_quantizer api, use the bitwidth in configuration file")
            if self._qconfig['bias']['bit_width'] != bit_width_w:
                NndctScreenLogger().warning2user(QWarning.BITWIDTH_MISMATCH, f"Bitwidth of bias in configuration file is different from that passed from torch_quantizer api, use the bitwidth in configuration file")
        if bit_width_a:
            if self._qconfig['activation']['bit_width'] != bit_width_a:
                NndctScreenLogger().warning2user(QWarning.BITWIDTH_MISMATCH, f"Bitwidth of activation in configuration file is different from that passed from torch_quantizer api, use the bitwidth in configuration file")
            if self._qconfig['input']['bit_width'] != bit_width_a:
                NndctScreenLogger().warning2user(QWarning.BITWIDTH_MISMATCH, f"Bitwidth of input in configuration file is different from that passed from torch_quantizer api, use the bitwidth in configuration file")
        if mix_bit:
            if self._qconfig['mix_bit'] != mix_bit:
                NndctScreenLogger().warning2user(QWarning.BITWIDTH_MISMATCH, f"Mix_bit parameter in configuration file is different from that passed from torch_quantizer api, use mix_bit parameter in configuration file")
        
        self._compute_q_maxmin()
        self._qconfig_handle_conflict()
    
    @staticmethod
    def _keywords_legel(json_configs):
        model_config_keys = ["convert_relu6_to_relu", "convert_silu_to_hswish", "include_cle", "keep_first_last_layer_accuracy", 
                            "keep_add_layer_accuracy", "include_bias_corr", "target_device",
                            "change_concat_input_fix", "change_pool_input_fix",
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
                NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Unsupported keyword in quantization config: '{key_word}'. ")
                exit(2)
                
            if key_word == "overall_quantize_config":
                overall_config = json_configs["overall_quantize_config"]
                for key1_word, _ in overall_config.items():
                    if key1_word not in quant_param_keys:
                        NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Unsupported keyword in quantization config: '{key1_word}'.")
                        exit(2)
            
            if key_word == "tensor_quantize_config":
                tensor_config = json_configs["tensor_quantize_config"]
                for key1_word, _ in tensor_config.items():
                    if key1_word not in tensor_keys:
                        NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Unsupported keyword in quantization config: '{key1_word}'.")
                        exit(2)
                        
                    tensor_overall_config = tensor_config[key1_word]
                    for key2_word, _ in tensor_overall_config.items():
                        if key2_word not in quant_param_keys:
                            NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Unsupported keyword in quantization config: '{key2_word}'.")
                            exit(2)
            
            if key_word == "layer_quantize_config":
                layer_configs = json_configs["layer_quantize_config"]
                for layer_config in layer_configs:
                    for key_word1, _ in layer_config.items():
                        if key_word1 not in layer_quant_keys:
                            NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Unsupported keyword in quantization config: '{key_word1}'.")
                            exit(2)
                        
                        if key_word1 == "overall_quantize_config":
                            overall_config = layer_config["overall_quantize_config"]
                            for key1_word, _ in overall_config.items():
                                if key1_word not in quant_param_keys:
                                    NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Unsupported keyword in quantization config: '{key1_word}'.")
                                    exit(2)
                                    
                        if key_word1 == "tensor_quantize_config":
                            tensor_config = layer_config["tensor_quantize_config"]
                            for key1_word, _ in tensor_config.items():
                                if key1_word not in tensor_keys:
                                    NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Unsupported keyword in quantization config: '{key1_word}'.")
                                    exit(2)
                                    
                                tensor_overall_config = tensor_config[key1_word]
                                for key2_word, _ in tensor_overall_config.items():
                                    if key2_word not in quant_param_keys:
                                        NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Unsupported keyword in quantization config: '{key2_word}'.")
                                        exit(2)
            
                
    @staticmethod
    def _nndct_switch_option(json_configs):
        if (json_configs.get('convert_relu6_to_relu', None) is not None) and isinstance(json_configs['convert_relu6_to_relu'], bool):
            option_util.set_option_value("nndct_convert_relu6_to_relu", json_configs['convert_relu6_to_relu'])
        if (json_configs.get('convert_silu_to_hswish', None) is not None) and isinstance(json_configs['convert_silu_to_hswish'], bool):
            option_util.set_option_value("nndct_convert_silu_to_hswish", json_configs['convert_silu_to_hswish'])
        if (json_configs.get('include_cle', None) is not None) and isinstance(json_configs['include_cle'], bool):
            option_util.set_option_value("nndct_equalization", json_configs['include_cle'])
        if (json_configs.get('change_concat_input_fix', None) is not None) and isinstance(json_configs['change_concat_input_fix'], bool):
            option_util.set_option_value("nndct_change_concat_input_fix", json_configs['change_concat_input_fix'])
        if (json_configs.get('change_pool_input_fix', None) is not None) and isinstance(json_configs['change_pool_input_fix'], bool):
            option_util.set_option_value("nndct_change_pool_input_fix", json_configs['change_pool_input_fix'])
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
            for tensor_type in q_config["quantizable_data_type"]:
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
                        nndct_layer_config[tensor_type] = copy.deepcopy(self._qconfig[tensor_type])
                    
                    self.set_tensor_quant_config(layer_config, nndct_layer_config)
                    #q_config['layer_type_config'].append(nndct_layer_config)
                    q_config['layer_type_config'][torch_layer_type] = nndct_layer_config
                elif layer_config.get('layer_name', None):
                    nndct_layer_name = layer_config.get('layer_name')
                    layer_tensor_types = layer_config.get('quantizable_data_type')
                    nndct_layer_config = {'quantizable_data_type':layer_tensor_types}
                    for tensor_type in nndct_layer_config['quantizable_data_type']:
                        nndct_layer_config[tensor_type] = copy.deepcopy(self._qconfig[tensor_type])
                    self.set_tensor_quant_config(layer_config, nndct_layer_config)
                    #q_config['layer_type_config'].append(nndct_layer_config)
                    q_config['layer_name_config'][nndct_layer_name] = nndct_layer_config
    
    def _map_tensor_type_config(self, tensor_type, export_config, q_config):
        if q_config.get(tensor_type, None) is None:
            q_config[tensor_type] = {}
        config_to_use = q_config[tensor_type]
        q_config[tensor_type] = self._generate_config_from_export(tensor_type, export_config, config_to_use)
    
    def _generate_config_from_export(self, tensor_type, export_config, config_to_use):
        if export_config.get('bit_width', None):
            self.parse_bit_width(tensor_type, 'bit_width', export_config.get('bit_width'), config_to_use)
            #config_to_use['bit_width'] = export_config.get('bit_width')
            
        if (export_config.get('method', None)):
            if (export_config['method'] in self._legal_qconfigs[tensor_type]['method']):
                config_to_use['method'] = export_config.get('method')
            else:
                method_legels = self._legal_qconfigs[tensor_type]['method']
                NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"The method of {tensor_type} should be in the list {method_legels}")
                exit(2)
                
        if (export_config.get('round_mode', None)):
            if (export_config['round_mode'] in self._legal_qconfigs[tensor_type]['round_method']):
                config_to_use['round_method'] = export_config.get('round_mode')
            else:
                round_legels = self._legal_qconfigs[tensor_type]['round_method']
                NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"The round_mode of {tensor_type} should be in the list {round_legels}")
                exit(2)
                
        if (export_config.get('scale_type', None)):
            if (export_config['scale_type'] in self._legal_qconfigs[tensor_type]['scale_type']):
                config_to_use['scale_type'] = export_config.get('scale_type')
            else:
                scale_legels = self._legal_qconfigs[tensor_type]['scale_type']
                NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"The scale_type of {tensor_type} should be in the list {scale_legels}")
                exit(2)
            
        if (not export_config.get('symmetry', None) is None):
            if isinstance(export_config.get('symmetry'), bool):
                config_to_use['symmetric_mode'] = 'symmetric' if export_config.get('symmetry') else 'asymmetric'
            else:
                NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"The symmetry parameter of {tensor_type} should be a boolean")
                exit(2)
            
        if (not export_config.get('per_channel', None) is None):
            if isinstance(export_config.get('per_channel'), bool):
                config_to_use['granularity'] = 'per_channel' if export_config.get('per_channel') else 'per_tensor'
            else:
                NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"The per_channel parameter of {tensor_type} should be a boolean")
                exit(2)
                
        if (not export_config.get('signed', None) is None):
            if isinstance(export_config.get('signed'), bool):
                config_to_use['signed'] = export_config.get('signed')
            else:
                NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"The signed parameter of {tensor_type} should be a boolean")
                exit(2)
            
        if (not export_config.get('narrow_range', None) is None):
            if isinstance(export_config.get('narrow_range'), bool):
                config_to_use['narrow_range'] = export_config.get('narrow_range')
            else:
                NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"The narrow_range parameter of {tensor_type} should be a boolean")
                exit(2)

        if tensor_type in ['activation', 'input']:
            if (export_config.get('calib_statistic_method', None)):
                if (export_config['calib_statistic_method'] in self._legal_qconfigs[tensor_type]['calib_statistic_method']):
                    config_to_use['calib_statistic_method'] = export_config.get('calib_statistic_method')
                else:
                    calib_legels = self._legal_qconfigs[tensor_type]['calib_statistic_method']
                    NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"The calib_statistic_method of {tensor_type} should be in the list {calib_legels}")
                    exit(2)
        
        if config_to_use.get('method') == 'percentile':
            if export_config.get('percentage', None) is None:
                config_to_use['percentage'] = 99.99
            else:
                if ((not isinstance(export_config.get('percentage'), float)) or \
                    (export_config['percentage'] <= 0.0) or \
                    (export_config['percentage'] > 100.0)):
                    NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Percentage should be larger than 0.0 and smaller than 100.0")
                    exit(2)
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
    
    def _qconfig_handle_conflict(self):
        target_device = self._qconfig['target_device']
        for data_type in self._qconfig['quantizable_data_type']:
            self._handle_tensor_conflict(self._qconfig, data_type, target_device)
        for _, tensor_config in self._qconfig['layer_type_config'].items():
            for data_type in tensor_config['quantizable_data_type']:
                self._handle_tensor_conflict(tensor_config, data_type, target_device)
        for _, tensor_config in self._qconfig['layer_name_config'].items():
            for data_type in tensor_config['quantizable_data_type']:
                self._handle_tensor_conflict(tensor_config, data_type, target_device)

    @staticmethod
    def _handle_tensor_conflict(tensor_config, tensor_type, target_device):
        if target_device == 'DPU':
            QConfigBase._handle_dpu_tensor_conflict(tensor_config, tensor_type)
        elif target_device == 'CPU':
            QConfigBase._handle_cpu_gpu_tensor_conflict(tensor_config, tensor_type)
        elif target_device == 'GPU':
            QConfigBase._handle_cpu_gpu_tensor_conflict(tensor_config, tensor_type)
            
    @staticmethod
    def _handle_cpu_gpu_tensor_conflict(tensor_config, tensor_type):
        quant_param = tensor_config[tensor_type]
        granularity = quant_param.get("granularity")
        if granularity == "per_channel":
            if tensor_type != "weights":
                NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Only support per_channel quantization for weights for now")
                exit(2)
            scale_type = quant_param.get("scale_type")
            method = quant_param.get("method")
            if scale_type == "float":
                if method != 'maxmin':
                    NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Only support maxmin calibration method in per_channel float quantization")
                    exit(2)
            elif scale_type == "poweroftwo":
                if method not in ['diffs', 'maxmin']:
                    NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Only support diffs and maxmin calibration method in per_channel poweroftwo quantization")
                    exit(2)
                symmetric_mode = quant_param.get("symmetric_mode")
                if symmetric_mode == "asymmetric":
                    NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Not support asymmetric quantization in per_channel poweroftwo quantization")
                    exit(2)
                signed = quant_param.get("signed")
                if not signed:
                    NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Not support unsigned quantization in per_channel poweroftwo quantization")
                    exit(2)
                
        if granularity == "per_tensor":
            method = quant_param.get("method")
            scale_type = quant_param.get("scale_type")
            if scale_type == "float":
                if method not in ['maxmin', 'percentile', 'mse', 'entropy']:
                    NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Only support maxmin, percentile, mse and entropy calibration method in per_tensor float quantization")
                    exit(2)
                if quant_param.get("calib_statistic_method", None):
                    calib_statistic_method = quant_param.get("calib_statistic_method")
                    if calib_statistic_method not in ["max", "mean", "median"]:
                        NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Only support max, mean and median scale activation statistic method in per_tensor float quantization")
                        exit(2)
                symmetric_mode = quant_param.get("symmetric_mode")
                if symmetric_mode == "asymmetric":
                    if method in ['percentile', 'entropy', 'mse']:
                        NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Not support asymmetric quantization in percentile, entropy and mse calibration method")
                        exit(2)
            elif scale_type == "poweroftwo":
                if method not in ['diffs', 'maxmin']:
                    NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Only support diffs and maxmin calibration method in per_tensor poweroftwo quantization")
                    exit(2)
                symmetric_mode = quant_param.get("symmetric_mode")
                if symmetric_mode == "asymmetric":
                    NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Not support asymmetric quantization in per_tensor poweroftwo quantization")
                    exit(2)
                signed = quant_param.get("signed")
                if not signed:
                    NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Not support unsigned quantization in per_tensor poweroftwo quantization")
                    exit(2)
                if quant_param.get("calib_statistic_method", None):
                    calib_statistic_method = quant_param.get("calib_statistic_method")
                    if calib_statistic_method != "modal":
                        NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Only support modal scale activation statistic method in per_tensor poweroftwo quantization")
                        exit(2)
        
    @staticmethod
    def _handle_dpu_tensor_conflict(tensor_config, tensor_type):
        quant_param = tensor_config[tensor_type]
        
        symmetric_mode = quant_param.get("symmetric_mode")
        if symmetric_mode != "symmetric":
            NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Only support symmetric quantization in DPU device")
            exit(2)
        method = quant_param.get("method")
        if method not in ["diffs", "maxmin"]:
            NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Only support diffs and maxmin calibration method in DPU device")
            exit(2)
        granularity = quant_param.get("granularity")
        if granularity != "per_tensor":
            NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Only support per_tensor quantization in DPU device")
            exit(2)
        scale_type = quant_param.get("scale_type")
        if scale_type != "poweroftwo":
            NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Only support poweroftwo scale type in DPU device")
            exit(2)
        narrow_range = quant_param.get("narrow_range")
        if narrow_range:
            NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Not support narrow_range in DPU device")
            exit(2)
        if quant_param.get("calib_statistic_method", None):
            calib_statistic_method = quant_param.get("calib_statistic_method")
            if calib_statistic_method != "modal":
                NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Only support modal activation statistic method in DPU device")
                exit(2)
        round_method = quant_param.get("round_method")
        if tensor_type == "activation":
            if round_method != "half_up":
                NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Only support half_up round method in activation quantization of DPU device")
                exit(2)
        else:
            if round_method != "std_round":
                NndctScreenLogger().error2user(QError.QUANT_CONFIG, f"Only support std_round round method in weights, bias and input quantization of DPU device")
                exit(2)

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
 
