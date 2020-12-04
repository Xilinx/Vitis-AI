

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
import json
import copy
import numpy as np
from .operator import *
from lstm_compiler.parser import XirParser
from tqdm import tqdm

class LstmCellSimulator():
    def __init__(self, layer_name, nodes_info, nodes_edge, cycle_ops, reverse=False):
        self.__layer_name = layer_name
        self.__nodes_info = LstmCellSimulator.__load_params(nodes_info)
        self.__nodes_edge = LstmCellSimulator.__load_params(nodes_edge)
        self.__cycle_ops = LstmCellSimulator.__load_params(cycle_ops)
        self.__reverse = reverse
        
        # merge nodes_info and nodes_edge
        self.__nodes_temp = copy.deepcopy(self.__nodes_info[:-3])
        for node in self.__nodes_temp:
            node['inputs'] = XirParser.get_input_nodes(self.__nodes_edge, node)
        
        self.__simulator_data_path = GLOBAL_VARIABLE.SIMULATOR_DATA_PATH + self.__layer_name + '/'
        if not os.path.exists(self.__simulator_data_path):
            os.makedirs(self.__simulator_data_path)
 
    @staticmethod
    def __load_params(param_name):
        param_info = []
        if isinstance(param_name, str):
            with open(param_name, 'r') as f_param:
                param_info = json.load(f_param)
        elif isinstance(param_name, list):
            param_info = param_name
        else:
            raise ValueError('Not supported param type')
        return param_info

    '''
    def __get_input_nodes(self, node):
        input_nodes = []
        for edge in self.__nodes_edge:
            if edge[1] == node['name']:
                input_nodes.append(edge[0])
        return input_nodes
    '''
    
    def infer_shape(self):
        pass
    
    def run(self, inputs, frame_cnt):
        if len(inputs) != frame_cnt:
            raise ValueError('Inputs length does not match frames count')
        
        input_frames = copy.deepcopy(inputs[::-1]) if self.__reverse else copy.deepcopy(inputs)
        
        h_prev_name = self.__cycle_ops[0][0]
        h_prev_node = XirParser.get_node_from_name(self.__nodes_temp, h_prev_name)
        h_prev_node['data'] = np.zeros(h_prev_node['shape'], dtype = np.int64).tolist()
        
        c_prev_name = self.__cycle_ops[1][0]
        c_prev_node = XirParser.get_node_from_name(self.__nodes_temp, c_prev_name)
        c_prev_node['data'] = np.zeros(c_prev_node['shape'], dtype = np.int64).tolist()
        
        h_next_name = self.__cycle_ops[0][1]
        c_next_name = self.__cycle_ops[1][1]
        input_node = LstmCellSimulator.get_input_node(self.__nodes_temp)
        
        for frame in tqdm(range(frame_cnt), desc="Simulation Process"):
            frame_input = input_frames[frame]
            input_node['data'] = np.array(frame_input).reshape(input_node['shape']).tolist()
            params_data = LstmCellSimulator.__run_one_frame(self.__nodes_temp)

            c_prev_node['data'] = copy.deepcopy(params_data[c_next_name])
            h_prev_node['data'] = copy.deepcopy(params_data[h_next_name])
            
            LstmCellSimulator.__dump_data(self.__simulator_data_path, frame, params_data)
    
    @staticmethod
    def __run_one_frame(nodes_list):
        all_data = {}
        for node in nodes_list:
            input_nodes = []
            for input_name in node['inputs']:
                input_nodes.append(XirParser.get_node_from_name(nodes_list, input_name))
            node_config = {'node':node,
                           'input_nodes':input_nodes}
            
            node_op = make_operator(**node_config)
            node['shape'], node['data'] = node_op.run()
            all_data.update({node['name']: node['data']})
        return all_data

    @staticmethod
    def __dump_data(data_path, frame_cnt, params_data):
        data_frame_path = data_path + 'frame_' + str(frame_cnt) + '/'
        if not os.path.exists(data_frame_path):
            os.makedirs(data_frame_path)
        for (name, data) in params_data.items():
            params_path = data_frame_path + name + '.txt'
            np.savetxt(params_path, np.array(data), fmt='%d', delimiter='\n')
    
    '''       
    @staticmethod            
    def __get_node_from_name(nodes_info, name):
        for node in nodes_info:
            if node['name'] == name:
                return node
    '''
    
    @staticmethod
    def get_input_node(nodes_info):
        for node in nodes_info:
            if (node['type'] == 'vector') and ('input_0' in node['name']):
                return node 

    @property
    def simulator_data_path(self):
        return self.__simulator_data_path
