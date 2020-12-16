

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
from functools import reduce
import numpy as np
from xir import Graph
from xir import Op
from utils.processing import *
from utils.data import *
from utils.tools import *
from .utils import *

class XirParser():
    def __init__(self, *args, **kwargs):
        self.__nodes_list = kwargs['ordered_nodes']
        self.__graph = kwargs['graph']
        self.__fix_configs = kwargs['fix_configs']
        self.__fix_data = kwargs['fix_data']
        self.__op_edge = []
        self.__op_info = []
        self.__invalid_nodes = []
        self.__cycle_ops = []
        #self.DIM_UNCHANGE_OPS = ['eltwise', 'mul', 'relu', 'relu6', 'sigmoid', 'tanh', 'sub']
        #self.MATMUL_OPS = ['matmul', 'linear']
        self.__DATA_NODES = ['group', 'data', 'vector']
        self.__MULTI_NODES = ['matmul', 'matmul_relu']
        self.__EMUL_NODES = ['mul']
        self.__ACTI_NODES = ['sigmoid', 'tanh']
        self.__ADD_NODES = ['eltwise', 'sub', 'add', 'eltwise_relu']
        self.__CONCAT_NODES = ['concat']
        
    def reorder_add_and_matmul(self):
        for i in range(len(self.__nodes_list)):
            node = self.__nodes_list[i]
            if node.get_type() == 'add' and node.get_input_num() == 2:
                input_ops = get_input_ops_list(node)
                input_ops_set = set([input_op.get_type() for input_op in input_ops])
                if input_ops_set == set(["add", "matmul"]):
                    for input_op in input_ops:
                        if input_op.get_type() == "matmul":
                            op_temp = input_op
                            self.__nodes_list.pop(self.__nodes_list.index(input_op))
                            node_index = self.__nodes_list.index(node)
                            self.__nodes_list.insert(node_index, op_temp)
        
    #nodes_list, graph, fix_configs, fix_data
    def make_op_connection(self):
        for node in self.__nodes_list:
            node_info = self.__get_node_info(node)
            self.__op_info.append(node_info)
        
        op_names = [node['name'] for node in self.__op_info]
        '''
        def index(name):
            return op_names.index(name)
        '''
        for node in self.__nodes_list:
            if node.get_input_num() > 0:
                '''
                input_names = [input_op.get_name() for input_op in get_input_ops_list(node)]
                input_names.sort(key=index)
                for name in input_names:
                    self.__op_edge.append([name, node.get_name()])
                '''
                for op in get_input_ops_list(node):
                    self.__op_edge.append([op.get_name(), node.get_name()])
        
        self.__reorder_info_by_edge()
        self.__reorder_cross_in_graph()
        #self.__add_data_to_concat()
        
        for key in self.__fix_configs:
            for node in self.__op_info:
                if node['name'] == key:
                    node['bn'] = self.__fix_configs[key]['bit_width']
                    node['fp'] = self.__fix_configs[key]['fix_point']
                    node['signed'] = self.__fix_configs[key]['if_signed']
    
    def add_data_to_concat(self):
        concat_nodes_list = XirParser.get_nodes_from_type(self.__op_info, 'concat')
        if len(concat_nodes_list) == 0:
            return
        
        for concat_node in concat_nodes_list:
            input_nodes_name = XirParser.get_input_nodes(self.__op_edge, concat_node)
            for node_name in input_nodes_name:
                node = XirParser.get_node_from_name(self.__op_info, node_name)
                if node['type'] not in self.__DATA_NODES:
                    data_node = {}
                    data_node['name'] = node['name'] + '_data'
                    data_node['type'] = 'data'
                    if 'shape' in node.keys(): data_node['shape'] = node['shape']
                    if 'fp' in node.keys(): data_node['fp'] = node['fp']
                    if 'bn' in node.keys(): data_node['bn'] = node['bn']
                    if 'signed' in node.keys(): data_node['signed'] = node['signed']
                    if 'shift' in node.keys(): data_node['shift'] = node['shift']
                    XirParser.insert_node(self.__op_info, node, data_node)
                    concat_edge = [node['name'], concat_node['name']]
                    XirParser.insert_edge(self.__op_edge, concat_edge, data_node)
                    self.__invalid_nodes.append(data_node)
    
    def __reorder_info_by_edge(self):
        nodes_rev = self.__op_edge[::-1]
        nodes_1 = []
        nodes_2 = []
        for node in nodes_rev:
            node_temp = copy.deepcopy(node)
            if node_temp[1] in nodes_2:
                node_temp.pop(1)
            else:
                nodes_2.append(node_temp[1])
            nodes_1.append(node_temp)
        nodes_3 = nodes_1[::-1]
        nodes_reorder = [node for nodes_4 in nodes_3 for node in nodes_4]
        self.__op_info.sort(key=lambda x:nodes_reorder.index(x['name']))       
        
    
    # This function should rewrite later    
    def __reorder_cross_in_graph(self):
        engine_connection_dict = {}
        
        for edge in self.__op_edge:
            edge_node0 = XirParser.get_node_from_name(self.__op_info, edge[0])
            edge_node1 = XirParser.get_node_from_name(self.__op_info, edge[1])
            edge_node0_engine = GLOBAL_VARIABLE.TYPE_ENGINE_DICT[edge_node0['type']]
            edge_node1_engine = GLOBAL_VARIABLE.TYPE_ENGINE_DICT[edge_node1['type']]
            engine_connection_name = edge_node0_engine + '2' + edge_node1_engine
            if engine_connection_name not in engine_connection_dict.keys():
                engine_connection_dict[engine_connection_name] = [[edge_node0, edge_node1]]
            else:
                engine_connection_dict[engine_connection_name].append([edge_node0, edge_node1])
            
        for (name, edge_list) in engine_connection_dict.items():
            engine0_name, engine1_name = name.split('2')
            edge_len = len(edge_list)
            for i in range(edge_len):
                node_edge = edge_list[i]
                for item in edge_list[i+1::]:
                    cur_edge_node0_index = self.__op_info.index(node_edge[0])
                    next_edge_node0_index = self.__op_info.index(item[0])
                    if (cur_edge_node0_index < next_edge_node0_index) and ([item[1]['name'], node_edge[1]['name']] in self.__op_edge):
                        temp = node_edge[0]
                        self.__op_info[cur_edge_node0_index] = item[0]
                        self.__op_info[next_edge_node0_index] = temp
                    elif (cur_edge_node0_index > next_edge_node0_index) and ([node_edge[1]['name'], item[1]['name']] in self.__op_edge):
                        temp = node_edge[0]
                        self.__op_info[cur_edge_node0_index] = item[0]
                        self.__op_info[next_edge_node0_index] = temp
        info_name_list = [node['name'] for node in self.__op_info]
        self.__op_edge.sort(key=lambda x:info_name_list.index(x[1]))
        self.__reorder_info_by_edge()

    #node, info_list, graph, fix_data
    def __get_node_info(self, node):
        node_type = node.get_type()
        #print(self.__nodes_list)
        info_name = [temp['name'] for temp in self.__op_info]
        node_info = {'name':node.get_name(), 'fp':0, 'bn':16, 'signed':True}
        
        if node_type == 'const':           
            #node_info['shape'] = node.get_output_tensor().dims
            node_ndim = node.get_output_tensor().ndim
            down_ops = get_down_ops(node, self.__graph)
            if node.get_name() in self.__fix_data:
                const_data = self.__fix_data[node.get_name()]
            else:
                #const_data = node.get_attrs()['data']
                #const_data = node.get_attr('data')
                const_data = const_op_data(node)
            
            if node_ndim == 2:
                if (len(down_ops) > 0) and (any(op.get_type() in GLOBAL_VARIABLE.MATMUL_OPS for op in down_ops)):
                    node_info['type'] = 'group'
                    #node_info['shape'] = node.get_output_tensor().dims
                    #node_info['data'] = const_data.tolist()
                else:
                    node_info['type'] = 'data'
                    #node_info['shape'] = node.get_output_tensor().dims[::-1]
                    #node_info['data'] = const_data.transpose().tolist()
            elif node_ndim == 1:
                node_info['type'] = 'data'
                '''
                node_info['shape'] = [node.get_output_tensor().dims[0], 1]
                node_info['data'] = const_data[:,np.newaxis].tolist()
                '''
            node_info['shape'] = node.get_output_tensor().dims
            node_info['data'] = const_data.tolist()
                
        elif node_type == 'data':
            #node_info['shape'] = node.get_output_tensor().dims[::-1]
            #node_info['shape'] = node.get_output_tensor().dims
            node_info['shape'] = node.get_output_tensor().dims
            node_info['data'] = np.zeros(node_info['shape'], dtype=np.int64).tolist()
            down_ops = get_down_ops(node, self.__graph)
            if (len(down_ops) > 0) and (any(op.get_type() in GLOBAL_VARIABLE.MATMUL_OPS for op in down_ops)):
                node_info['type'] = 'vector'
            else:
                node_info['type'] = 'data'
                
        elif node_type in GLOBAL_VARIABLE.MATMUL_OPS:
            attr_dict = get_matmul_weights_vector(node)
            
            weights_name = attr_dict['weights']
            vector_name = attr_dict['vector']
            if (weights_name not in info_name) or (vector_name not in info_name):
                raise AttributeError('Have not this op yet')
            weight_dim = XirParser.get_shape_from_name(self.__op_info, weights_name)
            vector_dim = XirParser.get_shape_from_name(self.__op_info, vector_name)
            
            if node.get_type() == 'matmul':
                node_info['transpose_a'] = node.get_attr('transpose_a')
                node_info['transpose_b'] = node.get_attr('transpose_b')
            else:
                node_info['transpose_a'] = False
                node_info['transpose_b'] = True
            
            if node_info['transpose_a']:
                vector_dim = vector_dim[::-1]
            if node_info['transpose_b']:
                weight_dim = weight_dim[::-1]
                
            if node.has_attr('fuse_relu'):
                node_info['type'] = 'matmul_relu'
            else:
                node_info['type'] = 'matmul'
            node_info['shape'] = [vector_dim[0], weight_dim[1]]
            
        elif node_type in GLOBAL_VARIABLE.DIM_UNCHANGE_OPS:
            any_op_name = get_any_input_op(node)
            if any_op_name not in info_name:
                raise AttributeError('Have not this op yet')
            node_info['type'] = node.get_type()
            #print('self.__op_info', self.__op_info)
            #print('any_op_name', any_op_name)
            node_info['shape'] = XirParser.get_shape_from_name(self.__op_info, any_op_name)
            
            #node_info['data'] = np.zeros(node_info['shape'], dtype=np.int64).tolist()

        elif node_type in GLOBAL_VARIABLE.ADD_OPS:
            input_op_num = node.get_input_num()
            node_info['type'] = 'eltwise' if node_type == 'add' else node_type
            
            if node.has_attr('fuse_relu'):
                node_info['type'] = node_info['type'] + '_relu'
            
            if input_op_num > 0:
                input_ops = get_input_ops_list(node)
                ops_shape = []
                for op in input_ops:
                    op_shape = XirParser.get_shape_from_name(self.__op_info, op.get_name())
                    ops_shape.append(op_shape)
                ops_array = [np.ones(shape) for shape in ops_shape] 
                node_info['shape'] = reduce(lambda x,y: np.add(x,y), ops_array).shape
            
        elif node_type == 'mul':
            input_op_num = node.get_input_num()
            node_info['type'] = node.get_type()
            if input_op_num > 0:
                input_ops = get_input_ops_list(node)
                ops_shape = []
                for op in input_ops:
                    op_shape = XirParser.get_shape_from_name(self.__op_info, op.get_name())
                    ops_shape.append(op_shape)
                ops_array = [np.ones(shape) for shape in ops_shape]
                node_info['shape'] = reduce(lambda x,y: x*y, ops_array).shape
                
        elif node_type == 'concat':
            concat_axis = node.get_attrs()['axis']
            input_op_num = node.get_input_num()
            node_info['type'] = node.get_type()
            node_info['axis'] = concat_axis
            if input_op_num > 0:
                input_ops = get_input_ops_list(node)
                ops_shape = []
                for op in input_ops:
                    op_shape = XirParser.get_shape_from_name(self.__op_info, op.get_name())
                    ops_shape.append(op_shape)
                ops_array = [np.ones(shape) for shape in ops_shape]
                node_info['shape'] = reduce(lambda x,y: np.concatenate((x,y), axis=concat_axis), ops_array).shape
        else:
            raise KeyError('Node type is not supported')
        
        return node_info
    
    #node_edge, node_info_list, actv_configs
    def extend_op_connection(self):
        result_dict = {'name':self.__graph.get_name()+'__result', 'type':'data', 'fp':0, 'bn':16, 'signed':True}
        #result_dict['shape'] = self.__op_info[-1]['shape']
        result_dict['data'] = np.zeros(self.__op_info[-1]['shape'], dtype=np.int64).tolist()
        result_dict['shape'] = np.array(result_dict['data']).squeeze().shape
        if len(result_dict['shape'])==0: result_dict['shape'] = [1]
        result_dict['data'] = np.zeros(result_dict['shape'], dtype=np.int64).tolist()
        self.__op_info.append(result_dict)
        
        last_node_name = self.__op_info[-2]['name']
        self.__op_edge.append([last_node_name, result_dict['name']])
        
        sgmd_dict = {'name':'actv_sgmd', 'type':'data', 'fp':12, 'bn':16, 'signed':False}
        sgmd = np.array(GLOBAL_VARIABLE.SIGMOID_ACTV).reshape(2048)
        sgmd_dict['shape'] = sgmd.shape
        sgmd_dict['data'] = sgmd.tolist()
        self.__op_info.append(sgmd_dict)
        
        tanh_dict = {'name':'actv_tanh', 'type':'data', 'fp':13, 'bn':16, 'signed':True}
        tanh = np.array(GLOBAL_VARIABLE.TANH_ACTV).reshape(2048)
        tanh_dict['shape'] = tanh.shape
        tanh_dict['data'] = tanh.tolist()
        self.__op_info.append(tanh_dict)
        
    def make_shift_attrs(self):
        output_fp = self.__op_info[-4]['fp']
        
        for info_node in self.__op_info:
            if info_node['type'] in self.__DATA_NODES:
                info_node['shift'] = info_node['fp']
                
            elif info_node['type'] in self.__MULTI_NODES:
                info_node['shift'] = 0
                cur_op = self.__graph.get_op(info_node['name'])
                input_ops = get_input_ops_list(cur_op)
                for input_op in input_ops:
                    info_node['shift'] += XirParser.get_fp_from_name(self.__op_info, input_op.get_name())
                info_node['shift'] -= output_fp
            
            elif info_node['type'] in self.__EMUL_NODES:
                info_node['shift'] = 0
                cur_op = self.__graph.get_op(info_node['name'])
                input_ops = get_input_ops_list(cur_op)
                for input_op in input_ops:
                    if input_op.get_type() == 'sigmoid':
                        info_node['shift'] += GLOBAL_VARIABLE.SIGMOID_EXPO
                    elif input_op.get_type() == 'tanh':
                        info_node['shift'] += GLOBAL_VARIABLE.TANH_EXPO
                    else:
                        info_node['shift'] += XirParser.get_fp_from_name(self.__op_info, input_op.get_name())
                info_node['shift'] -= output_fp
 
            elif info_node['type'] in self.__ACTI_NODES:
                info_node['shift'] = output_fp
                
            elif info_node['type'] in self.__ADD_NODES:
                info_node['shift'] = 0
            
            elif info_node['type'] in self.__CONCAT_NODES:
                info_node['shift'] = 0
    ''' 
    def get_cycle_ops(self):
        h_prev_name = XirParser.get_op_from_name_and_type(self.__op_info, 'h_prev', 'vector')
        if h_prev_name is not None:
            h_next_name = XirParser.get_op_from_name_and_type(self.__op_info, 'h_next', None)
            if h_next_name is not None:
                self.__cycle_ops.append([h_prev_name, h_next_name])
        
        c_prev_name = XirParser.get_op_from_name_and_type(self.__op_info, 'c_prev', 'data')
        if c_prev_name is not None:
            c_next_name = XirParser.get_op_from_name_and_type(self.__op_info, 'c_next', 'eltwise')
            if c_next_name is not None:
                self.__cycle_ops.append([c_prev_name, c_next_name])
        return self.__cycle_ops
    '''
    
    def get_cycle_ops(self):
        if not self._XirParser__graph.has_attr('return_ops'):
            return
        #return_ops = getattr(self.__graph.metadata.get_attrs(), "get_attr_vstr")("return_ops")
        return_ops = self._XirParser__graph.get_attr('return_ops')
        h_prev_name = XirParser.get_op_from_name_and_type(self.__op_info, 'input_1', 'vector')
        if h_prev_name is not None:
            h_next_name = return_ops[0][:-4] if return_ops[0].endswith('_fix') else return_ops[0]
            self.__cycle_ops.append([h_prev_name, h_next_name])
        
        c_prev_name = XirParser.get_op_from_name_and_type(self.__op_info, 'input_2', 'data')
        if c_prev_name is not None:
            c_next_name = return_ops[1][:-4] if return_ops[1].endswith('_fix') else return_ops[1]
            self.__cycle_ops.append([c_prev_name, c_next_name])
        return self.__cycle_ops

    @staticmethod
    def get_node_from_name(node_info_list, name):
        for node in node_info_list:
            if node['name'] == name:
                return node
    
    @staticmethod
    def get_fp_from_name(node_info_list, name):
        for node in node_info_list:
            if node['name'] == name:
                return node['fp']
   
    @staticmethod
    def get_shape_from_name(info_list, name):
        for info in info_list:
            if info['name'] == name:
                return info['shape']
    
    @staticmethod
    def get_nodes_from_type(info_list, type_name):
        nodes_list = []
        for info in info_list:
            if info['type'] == type_name:
                nodes_list.append(info)
        return nodes_list

    @staticmethod
    def get_op_from_name_and_type(node_info_list, name, op_type):
        for node in node_info_list:
            if (name in node['name']) and ((op_type is None) or (op_type == node['type'])):
                return node['name']
        return None
    
    @staticmethod
    def get_input_nodes(node_edge_list, node):
        input_nodes = []
        for edge in node_edge_list:
            if edge[1] == node['name']:
                input_nodes.append(edge[0])
        return input_nodes
    
    @staticmethod
    def insert_node(node_info_list, node, data_node):
        node_index = node_info_list.index(node)
        node_info_list.insert(node_index+1, data_node)
    
    @staticmethod
    def insert_edge(node_edge_list, concat_edge, data_node):
        input_index = 0
        for node_edge in node_edge_list:
            input_index = input_index+1
            if node_edge[1] == concat_edge[0]:
                insert_index1 = input_index
        node_edge_list.insert(insert_index1, [concat_edge[0], data_node['name']])
        
        insert_index2 = node_edge_list.index(concat_edge)
        node_edge_list[insert_index2] = [data_node['name'], concat_edge[1]]
    
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
        return self.__invalid_nodes
