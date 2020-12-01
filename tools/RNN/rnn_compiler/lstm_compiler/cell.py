

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
import numpy as np
#from pathlib import Path
from xir import Graph
#from .xir_builder import LstmXirBuilder
from .utils import *
from .rebuilder import *
from .parser import *
from utils.processing import *
from utils.data import *
from utils.tools import *
from algorithms.graph_algorithms import reorder_xir_graph
from algorithms.pattern_matcher import PatternType, GraphSearcher

class LstmCellCompiler():
    #def __init__(self, configs=None, data_configs=None, xir_graph=None, actv_configs=None):
    def __init__(self, configs=None, xir_graph=None):
        self.__configs = {}
        self.__fix_configs = {}
        self.__const_fix_data = {}
        self.__op_info = []
        self.__op_edge = []
        self.__cycle_ops = []
        self.__invalid_ops = []
        #self.__REBUILD_OPS_TYPE = ['fix','strided_slice']
        self._build_config(configs, xir_graph)
        
    def _build_config(self, configs=None, xir_graph=None):
        if isinstance(xir_graph, Graph):
            self.__graph = xir_graph
            #self._fuse_graph()
            self._rebuild_graph()
        elif isinstance(xir_graph, str):
            #graph_path = Path(xir_graph)
            self.__graph = Graph.deserialize(xir_graph)
            #self._fuse_graph()
            self._rebuild_graph()
        else:
            raise ValueError("xir_graph type must be xir.Graph or string")
            '''
            if isinstance(configs, dict):
                config_temp = configs
            elif isinstance(configs, str):
                with open(configs, 'r') as f:
                    config_temp = json.load(f)
            else:
                raise ValueError("Config type must be dict or file")
            
            cell_name = list(config_temp.keys())[0]
            self.__configs = config_temp[cell_name]
        
            xir_builder = LstmXirBuilder(cell_name, self.__configs)
            self.__graph, self.__fix_configs = xir_builder.build(data_dict)
            '''
        
        self.__generate_data_path = GLOBAL_VARIABLE.GENERATE_DATA_PATH + self.__graph.get_name() + '/'
        if not os.path.exists(self.__generate_data_path):
            os.makedirs(self.__generate_data_path)
        
        self.__configs['weights_reload'] = True
        self.__configs['bias_reload'] = True
        self.__configs['vector_reload'] = True
        
        self.__ordered_nodes = reorder_xir_graph(self.__graph)
        
    def __get_op_input(self, name):
        op_input_dict = {}
        for op in self.__graph.get_ops():
            if op.get_type() == name:
                #op_input_dict[op.get_name()] = get_input_ops_list(op)[0].get_name()
                op_input_dict[op.get_name()] = [node.get_name() for node in get_input_ops_list(op)]
        return op_input_dict
    
    def __get_rebuild_ops(self, type_name):
        rebuild_ops = []
        for op in self.__graph.get_ops():
            if op.get_type() == type_name:
                rebuild_ops.append(op.get_name())
        return rebuild_ops

    def __remove_invalid_const_data(self):
        for op in self.__graph.get_ops():
            if (op.get_type() == 'const') or (op.get_type() == 'data'):
                down_ops = get_down_ops(op, self.__graph)
                if len(down_ops) == 0:
                    self.__graph.remove_op(op)
    
    def __make_one_fix_data(self, const_op_name, fix_op_name):
        const_op = self.__graph.get_op(const_op_name)
        fix_op = self.__graph.get_op(fix_op_name)
        #const_data = const_op.get_attrs()['data']
        #const_data = const_op.get_attr('data')
        const_data = const_op_data(const_op)
        fix_attrs = fix_op.get_attrs()  
        self.__const_fix_data[const_op_name] = quantize_data2int(const_data, fix_attrs['bit_width'], fix_attrs['fix_point'])       
    '''
    def _fuse_graph(self):
        graph_searcher = GraphSearcher(self.__graph)
        #for fuse_ops in GLOBAL_VARIABLE.FUSE_OPS_TYPE:
        nodes_set = graph_searcher.find_nodes_from_type([PatternType(pattern=['matmul','relu'])])
        self.__conv_relu_pairs = nodes_set[0]
    
    def _fuse_fix_config(self):
        for node_pair in self.__conv_relu_pairs:
            self.__fix_configs[node_pair[0].get_name()] = self.__fix_configs[node_pair[1].get_name()]
    '''
    def _rebuild_graph(self):        
        for op_type in GLOBAL_VARIABLE.REBUILD_OPS_TYPE:
            #op_input = self.__get_op_input(op_type)
            rebuild_ops = self.__get_rebuild_ops(op_type)
            if (op_type == 'relu') and (len(rebuild_ops) > 0):                
                rebuild_configs = {'graph':self.__graph, 'rebuild_ops':rebuild_ops}
                rebuilder = ReluRebuilder(**rebuild_configs)
                rebuilder.rebuild(self.__graph, self.__fix_configs)
            
            elif (op_type == 'fix') and (len(rebuild_ops) > 0):
                for fix_op in rebuild_ops:
                    #input_ops = [op for op in self.__graph.get_op(fix_op).get_input_ops()]
                    input_ops = [op for op in get_input_ops_list(self.__graph.get_op(fix_op))]
                    #if self.__graph.get_op(op_input[fix_op][0]).get_type() == 'const':
                    if input_ops[0].get_type() == 'const':
                        self.__make_one_fix_data(input_ops[0].get_name(), fix_op)
                        #self.__make_one_fix_data(op_input[fix_op][0], fix_op)
                
                rebuild_configs = {'graph':self.__graph, 'rebuild_ops':rebuild_ops}
                rebuilder = FixRebuilder(**rebuild_configs)
                rebuilder.rebuild(self.__graph, self.__fix_configs)
                #self._fuse_fix_config()
                                
            elif (op_type == 'strided_slice') and (len(rebuild_ops) > 0):
                rebuild_configs = {'graph':self.__graph, 'rebuild_ops':rebuild_ops, 'fix_data':self.__const_fix_data}
                rebuilder = StridedSliceRebuilder(**rebuild_configs)
                rebuilder.rebuild(self.__graph, self.__fix_configs)
                
            elif (op_type in GLOBAL_VARIABLE.MATMUL_OPS) and (len(rebuild_ops) > 0):
                rebuild_configs = {'graph':self.__graph, 'rebuild_ops':rebuild_ops}
                rebuilder = MatmulRebuilder(**rebuild_configs)
                rebuilder.rebuild(self.__graph, self.__fix_configs)
            
        self.__remove_invalid_const_data()
        '''
        for op in self.__graph.get_ops():
            print(op.get_name())
            print(op.get_type())
        '''
    '''
    def __rebuild_graph(self):
        for op_type in GLOBAL_VARIABLE.REBUILD_OPS_TYPE:
            if op_type == 'fix':
                fix_op_input = self.__get_op_input(op_type)
    
                for fix_op in fix_op_input:
                    self.__fix_configs[fix_op_input[fix_op]] = self.__graph.get_op(fix_op).get_attrs()
                    if self.__graph.get_op(fix_op_input[fix_op]).get_type() == 'const':
                        self.__make_one_fix_data(fix_op_input[fix_op], fix_op)
                    fix_rebuild_config = self.__get_fix_rebuild_config(fix_op, fix_op_input)
                    fix_rebuild_config['graph'] = self.__graph
                    fix_rebuilder = FixRebuilder(fix_rebuild_config)
                    fix_rebuilder.rebuild()

            if op_type == 'strided_slice':
                #stridedslice_op_input = self.__get_stridedslice_op_input__()
                stridedslice_op_input = self.__get_op_input(op_type)
                print(stridedslice_op_input)
                
                while len(stridedslice_op_input) > 0:
                    any_dict_index = list(stridedslice_op_input.keys())[0]
                    stridedslice_rebuild_config = self.__get_stridedslice_rebuild_config(any_dict_index, stridedslice_op_input)
                    stridedslice_rebuild_config['graph'] = self.__graph
                    stridedslice_rebuild_config['fix_data'] = self.__const_fix_data
                    stridedslice_rebuilder = StridedSliceRebuilder(stridedslice_rebuild_config)
                    stridedslice_rebuilder.rebuild(self.__fix_configs)
        self.__remove_invalid_const_data()
        for op in self.__graph.get_ops():
            print(op.get_name())
            print(op.get_type())
    '''    

    def reset(self, configs, xir_graph=None):
        pass
    
    def compile(self):

        parser_configs = {'ordered_nodes':self.__ordered_nodes,
                          'graph':self.__graph,
                          'fix_configs':self.__fix_configs,
                          'fix_data':self.__const_fix_data}
        xir_parser = XirParser(**parser_configs)
    
        '''
        xir_parser = XirParser(ordered_nodes=self.__ordered_nodes,
                               graph=self.__graph,
                               fix_configs=self.__fix_configs,
                               fix_data=self.__const_fix_data)
        '''
        xir_parser.reorder_add_and_matmul()
        xir_parser.make_op_connection()
        xir_parser.extend_op_connection()
        xir_parser.get_cycle_ops()
        xir_parser.make_shift_attrs()
        xir_parser.add_data_to_concat()
        self.__op_info = xir_parser.op_info
        self.__op_edge = xir_parser.op_edge
        self.__cycle_ops = xir_parser.cycle_ops
        self.__invalid_ops = xir_parser.invalid_ops
        
        ''' 
        for node in op_info:
            print('name: {}\ntype: {}\nfp: {}\nshift: {}'.format(node['name'], node['type'], node['fp'], node['shift']))
        '''
     
        with open(self.__generate_data_path + 'nodes_edge.json', 'w') as f_edge:
            json.dump(self.__op_edge, f_edge, indent=4)
    
        with open(self.__generate_data_path + 'nodes_info.json', 'w') as f_info:
            json.dump(self.__op_info, f_info, indent=4)
    
        with open(self.__generate_data_path + 'cycle_ops.json', 'w') as f_cycle:
            json.dump(self.__cycle_ops, f_cycle, indent=4)
            
        with open(self.__generate_data_path + 'invalid_ops.json', 'w') as f_invalid:
            json.dump(self.__invalid_ops, f_invalid, indent=4)
        
        '''
        self.op_edge, self.op_info_list = make_op_connection(self.__ordered_nodes, 
                                        self.__graph, self.__fix_configs, self.__const_fix_data)
        extend_op_connection(self.op_edge, self.op_info_list, self.actv_configs)
        self.cycle_ops = get_cycle_ops(self.op_info_list)
        '''
        
    @property
    def graph(self):
        return self.__graph
    
    @property
    def ordered_nodes(self):
        return self.__ordered_nodes
    
    @property
    def op_info(self):
        return self.__op_info
    
    @property
    def op_edge(self):
        return self.__op_edge

    @property
    def cycle_ops(self):
        return self.__cycle_ops
		
    @property
    def invalid_ops(self):
        return self.__invalid_ops
