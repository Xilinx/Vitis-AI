

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
from tqdm import tqdm
from .cell_simulator import *
from lstm_compiler.parser import XirParser

class LstmModuleSimulator():
    def __init__(self, data_path):
        if not os.path.exists(data_path+'model_config.json'):
            raise IOError('model_config.json file does not exist')
        
        self.__data_path = data_path
        with open(data_path+'model_config.json', 'r') as f_model:
            self.__layers_info = json.load(f_model)
        
        self.__simulator_data_path = GLOBAL_VARIABLE.SIMULATOR_DATA_PATH
        if not os.path.exists(GLOBAL_VARIABLE.SIMULATOR_DATA_PATH):
            os.makedirs(GLOBAL_VARIABLE.SIMULATOR_DATA_PATH)

    def run(self, data_check = False):
        for layer_info in self.__layers_info:
            config_path = self.__data_path + '/' + layer_info['name'] + '/'
            with open(config_path+'nodes_info.json', 'r') as f_nodes_info:
                nodes_info = json.load(f_nodes_info)
            with open(config_path+'nodes_edge.json', 'r') as f_nodes_edge:
                nodes_edge = json.load(f_nodes_edge)
            with open(config_path+'cycle_ops.json', 'r') as f_cycle_ops:
                cycle_ops = json.load(f_cycle_ops)
            with open(config_path+'invalid_ops.json', 'r') as f_invalid_ops:
                invalid_ops = json.load(f_invalid_ops)
            #layer_reverse = True if layer_info['direction'] == 'backward' else False
            self.__remove_invalid_nodes(invalid_ops, nodes_info, nodes_edge)
            layer_reverse = False

            input_node = LstmCellSimulator.get_input_node(nodes_info)
            input_txt = input_node['name'] + '_fix.txt'
            
            dump_path = self.__data_path + '/deploy_check_data_int/' + layer_info['name'] + '/'
            dir_list = os.listdir(dump_path)
            dir_len = len(dir_list)
            
            input_frames = []
            for i in range(dir_len):
                index_name = 'frame_' + str(i) + '/' + input_txt
                input_frame=np.loadtxt(dump_path+index_name, dtype=int)
                input_frames.append(input_frame.tolist())
            
            cell_simulator = LstmCellSimulator(layer_name=layer_info['name'],
                                               nodes_info=nodes_info,
                                               nodes_edge=nodes_edge,
                                               cycle_ops=cycle_ops,
                                               reverse=layer_reverse)
            cell_simulator.run(input_frames, len(input_frames))
            
            if data_check:
                self.__check_data(dump_path, cell_simulator.simulator_data_path, layer_info['name'], nodes_info)
    
    def __remove_invalid_nodes(self, invalid_nodes, nodes_info, nodes_edge):
        for node in invalid_nodes:
            node_name = node['name']
            for node_edge in nodes_edge:
                if node_edge[1] == node_name:
                    input_edge = node_edge
                if node_edge[0] == node_name:
                    output_edge = node_edge
            output_index = nodes_edge.index(output_edge)
            nodes_edge[output_index] = [input_edge[0], output_edge[1]]
            nodes_edge.remove(input_edge)
            
            remove_node = XirParser.get_node_from_name(nodes_info, node_name)
            nodes_info.remove(remove_node)
    
    def __check_data(self, nndct_data_path, simulator_data_path, layer_name, nodes_info):
        dir_list = os.listdir(nndct_data_path)
        dir_len = len(dir_list)
        diff_info = '[DATA DIFFERENCES]: \n'
        find_diff = False
        for i in tqdm(range(dir_len), desc='Data checking'):
            fix_data_path = "%sframe_%d/*_fix.txt"%(nndct_data_path, i)
            simulator_frame_path = "%sframe_%d/"%(simulator_data_path, i)
            
            fix_path_list = glob.glob(fix_data_path)
            fix_txt_list = [fix_path.split('/')[-1] for fix_path in fix_path_list]
            simulator_list = os.listdir(simulator_frame_path)
            
            generate_data_list = [node['name'] for node in nodes_info if node['type'] != 'group']
            for data_name in generate_data_list:
                simulator_txt, fix_txt = LstmModuleSimulator.__get_nndct_and_compiler_name(data_name, 
                                                                                  generate_data_list,
                                                                                  fix_txt_list,
                                                                                  simulator_list)
                if simulator_txt:
                    simulator_txt_path = "%s%s"%(simulator_frame_path, simulator_txt)
                    fix_txt_path = "%sframe_%d/%s"%(nndct_data_path, i, fix_txt)
                    nndct_fix_data = np.loadtxt(fix_txt_path, dtype=int)
                    simulator_data = np.loadtxt(simulator_txt_path, dtype=int)
                    diff_data = nndct_fix_data - simulator_data
                    #print('***************************************************')
                    if not np.all(diff_data == 0):
                        add_info = 'Different in Layer %s, Frame %d, Parameter %s\n'%(layer_name, i, data_name)
                        diff_info = diff_info + add_info
                        find_diff = True
                else:
                    continue
        if find_diff:
            diff_txt_path = '%s%s/check_difference.txt'%(self.__simulator_data_path, layer_name)
            print('[DATA CHECK INFO] Have found difference in layer {}, check {}'.format(layer_name, diff_txt_path))
            with open(diff_txt_path, 'w') as f_diff:
                f_diff.write(diff_info) 
        else:
            print('[DATA CHECK INFO] Layer {} data is totally the same'.format(layer_name))
                
    @staticmethod
    def __get_nndct_and_compiler_name(data_name, name_list, fix_txt_list, simulator_list):
        txt_name = '%s.txt'%(data_name)
        fix_txt_name = '%s_fix.txt'%(data_name)
        if data_name.endswith('_matmul'):
            eltwise_name = '%s_eltwise'%(data_name[:-7])
            fix_txt_name = '%s_fix.txt'%(data_name[:-7])
            if (eltwise_name not in name_list) and (fix_txt_name in fix_txt_list) and (txt_name in simulator_list):
                return (txt_name, fix_txt_name)
            else:
                return (None, None)
        elif data_name.endswith('_eltwise'):
            fix_txt_name = '%s_fix.txt'%(data_name[:-8])
            if (fix_txt_name in fix_txt_list) and (txt_name in simulator_list):
                return (txt_name, fix_txt_name)
            else:
                return (None, None)
        elif (fix_txt_name in fix_txt_list) and (txt_name in simulator_list):
            return (txt_name, fix_txt_name)
        else:
            return (None, None)
    
    '''    
    def _check_data(self, nndct_data_path, simulator_data_path, layer_name):
        dir_list = os.listdir(nndct_data_path)
        dir_len = len(dir_list)
        diff_info = '[DATA DIFFERENCES]: \n'
        find_diff = False
        for i in tqdm(range(dir_len), desc='Data checking'):
            fix_data_path = "%sframe_%d/*_fix.txt"%(nndct_data_path, i)
            simulator_frame_path = "%sframe_%d/"%(simulator_data_path, i)
            
            fix_txt_list = glob.glob(fix_data_path)
            simulator_list = os.listdir(simulator_frame_path)
            
            for fix_txt in fix_txt_list:
                data_name = fix_txt.split('/')[-1][:-8]
                data_txt = data_name + '.txt'
                simulator_txt = LstmModuleSimulator.__find_txt_in_simulator_path(data_txt, simulator_list)
                if simulator_txt:
                    simulator_txt_path = "%s%s"%(simulator_frame_path, simulator_txt)
                    nndct_fix_data = np.loadtxt(fix_txt, dtype=int)
                    simulator_data = np.loadtxt(simulator_txt_path, dtype=int)
                    diff_data = nndct_fix_data - simulator_data
                    if not np.all(diff_data == 0):
                        add_info = 'Different in Layer %s, Frame %d, Parameter %s\n'%(layer_name, i, data_name)
                        diff_info = diff_info + add_info
                        find_diff = True
                else:
                    continue
        if find_diff:
            diff_txt_path = '%s%s/check_difference.txt'%(self.__simulator_data_path, layer_name)
            print('[DATA CHECK INFO] Have found difference in layer {}, check {}'.format(layer_name, diff_txt_path))
            with open(diff_txt_path, 'w') as f_diff:
                f_diff.write(diff_info) 
        else:
            print('[DATA CHECK INFO] Layer {} data is totally the same'.format(layer_name))
                    

    @staticmethod
    def __find_txt_in_simulator_path(txt_name, simulator_path):
        if txt_name in simulator_path:
            return txt_name
        eltwise_txt_name = txt_name.split('.')[0] + '_eltwise.txt'
        if eltwise_txt_name in simulator_path:
            return eltwise_txt_name
        matmul_txt_name = txt_name.split('.')[0] + '_matmul.txt'
        if matmul_txt_name in simulator_path:
            return matmul_txt_name
        return None
   ''' 
