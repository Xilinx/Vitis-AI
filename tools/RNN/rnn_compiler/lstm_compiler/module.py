

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

import os
import glob
import json
from xir import Graph
from utils.data import *
from .cell import LstmCellCompiler
from instruction_generator.apis import instruction_generate

class LstmModuleCompiler():
    def __init__(self, xmodels, device='u50'):
        r"""
        Create one LstmmModuleCompiler object
        xmodels: directory which contains xmodel files of the LSTM model,
                         or list which contains xmodels of the LSTM model.
        device: device to deploy the LSTM model. Default is 'u50'
        """
        self.__device = device
        if isinstance(xmodels, list) or isinstance(xmodels, tuple):
            self.__xmodels = sorted(xmodels, lambda xmodel:xmodel.get_name())
        elif isinstance(xmodels, str):
            path_xmodel = xmodels + '/*.xmodel'
            xmodels_list = glob.glob(path_xmodel)
            self.__xmodels = sorted(xmodels_list)
        else:
            raise ValueError("xmodels type shoule be list, tuple or string")
        
        self.__generate_data_path = GLOBAL_VARIABLE.GENERATE_DATA_PATH
        if not os.path.exists(GLOBAL_VARIABLE.GENERATE_DATA_PATH):
            os.makedirs(GLOBAL_VARIABLE.GENERATE_DATA_PATH)
        
        #layer_node_edge, layer_node_info_dict, layer_cycle_ops, layer_name_list
        self.__layer_node_edge = {}
        self.__layer_node_info_dict = {}
        self.__layer_cycle_ops = {}
        self.__layer_name_list = []

    def compile(self):
        model_config_list = []
        node_cnt = 0
        
        for model in self.__xmodels:
            cell_compiler = LstmCellCompiler(xir_graph=model)
            print("[COMPILOR INFO] Layer {} computation graph generating...".format(cell_compiler.graph.get_name()))
            cell_compiler.compile()
            model_config = {'name':cell_compiler.graph.get_name()}
            #print('attrs: ', cell_compiler.graph.get_attrs())
            if cell_compiler.graph.has_attr('direction'):
                #model_config['direction']=getattr(cell_compiler.graph.metadata.get_attrs(), "get_attr_str")("direction")
                model_config['direction']=cell_compiler.graph.get_attr("direction")
            else:
                model_config['direction']='forward'
            model_config_list.append(model_config)
            
            self.__layer_node_edge[node_cnt] = cell_compiler.op_edge
            self.__layer_node_info_dict[node_cnt] = cell_compiler.op_info
            self.__layer_cycle_ops[node_cnt] = cell_compiler.cycle_ops
            self.__layer_name_list.append(model_config['name'])
            node_cnt = node_cnt + 1
        
        with open(self.__generate_data_path + 'model_config.json', 'w') as f_config:
            json.dump(model_config_list, f_config, indent=4)
        #embed()    
        instruction_generate(self.__layer_node_edge, 
                             self.__layer_node_info_dict, 
                             self.__layer_cycle_ops, 
                             self.__layer_name_list,
                             model_config_list,
                             self.__device)
        
        print("[COMPILOR INFO] Lstm model compiling completed")
        
