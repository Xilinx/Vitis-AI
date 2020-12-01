

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
import numpy as np
from functools import reduce
from .utils import *
from utils.data import *

NODE_OPERATOR_DICT = {'group':'DataOperator', 
                      'vector':'DataOperator', 
                      'data':'DataOperator', 
                      'matmul':'MatmulOperator', 
                      'eltwise':'AddOperator', 
                      'sigmoid':'SigmoidOperator', 
                      'tanh':'TanhOperator', 
                      'mul':'EmulOperator', 
                      'sub':'SubOperator',
                      'concat': 'ConcatOperator',
                      'matmul_relu': 'MatmulReluOperator',
                      'eltwise_relu': 'AddReluOperator'}

def generate_all_operator():

    class BaseOperator():
        def __init__(self, *args, **kwargs):
            self._node = kwargs['node']
            self._input_nodes = kwargs['input_nodes']
        
        def run(self):
            data_shape = np.array(self._node['data']).shape
            if not list(data_shape) == list(self._node['shape']):
                raise ValueError('Data shape does not match the shape attribute')
            return (self._node['shape'], self._node['data'])

    class DataOperator(BaseOperator):
        pass

    class MatmulOperator(BaseOperator):
        def run(self):
            #weights_shape = []
            #input_shape = []
            for node in self._input_nodes:
                if node['type'] == 'group':
                    weights_matrix = np.array(node['data'])
                    #weights_shape = node['shape']
                else:
                    input_matrix = np.array(node['data'])
                    #input_shape = node['shape']
            #if not weights_shape[1] == input_shape[0]:
            #    raise ValueError('Matmul node, shape of weights does not match shape of input')
            #matmul_shape = [weights_shape[0], input_shape[1]]
            #print('weights: ', weights_matrix) 
            #print('input: ', input_matrix) 
            #print('shift: ', self._node['shift'])
            if self._node['transpose_a']:
                input_matrix = input_matrix.transpose()
            if self._node['transpose_b']:
                weights_matrix = weights_matrix.transpose()
            
            matmul_matrix = mmul(input_matrix.astype(np.int64), weights_matrix.astype(np.int64), self._node['shift'])
            #matmul_matrix = mmul(weights_matrix, input_matrix, self._node['shift'])
            return (self._node['shape'], matmul_matrix.tolist())
        
    class EmulOperator(BaseOperator):
        def run(self):
            '''
            input0_shape = self._input_nodes[0]['shape']
            input1_shape = self._input_nodes[1]['shape']
            if not input0_shape == input1_shape:
                raise ValueError('Emul node, shape of inputs does not match')
            '''
            
            input0_matrix = self._input_nodes[0]['data']
            input1_matrix = self._input_nodes[1]['data']
            emul_matrix = emul(np.array(input0_matrix, dtype=np.int64), 
                               np.array(input1_matrix, dtype=np.int64),
                               self._node['shift'])
            return (self._node['shape'], emul_matrix.tolist())
            
    class TanhOperator(BaseOperator):
        def run(self):
            input_matrix = self._input_nodes[0]['data']
            tanh_matrix = mapping_tanh(np.array(input_matrix, dtype=np.int64), 
                                       np.array(GLOBAL_VARIABLE.TANH_ACTV, dtype=np.int64),
                                       self._node['shift'])
            return (copy.deepcopy(self._input_nodes[0]['shape']), tanh_matrix.tolist())

    class SigmoidOperator(BaseOperator):
        def run(self):
            input_matrix = self._input_nodes[0]['data']
            sigmoid_matrix = mapping_sigm(np.array(input_matrix, dtype=np.int64), 
                                          np.array(GLOBAL_VARIABLE.SIGMOID_ACTV, dtype=np.int64),
                                          self._node['shift'])
            return (copy.deepcopy(self._input_nodes[0]['shape']), sigmoid_matrix.tolist())

    class AddOperator(BaseOperator):
        def run(self):
            '''
            input0_shape = self._input_nodes[0]['shape']
            input1_shape = self._input_nodes[1]['shape']
            
            if not input0_shape == input1_shape:
                raise ValueError('Add node, shape of inputs does not match')
            '''
                
            input0_matrix = self._input_nodes[0]['data']
            input1_matrix = self._input_nodes[1]['data']
            add_matrix = add(np.array(input0_matrix, dtype=np.int64), np.array(input1_matrix, dtype=np.int64))
            return (self._node['shape'], add_matrix.tolist())
        
    class SubOperator(BaseOperator):
        def run(self):
            '''
            input0_shape = self._input_nodes[0]['shape']
            input1_shape = self._input_nodes[1]['shape']
            
            if not input0_shape == input1_shape:
                raise ValueError('Sub node, shape of inputs does not match')
            '''
                
            input0_matrix = self._input_nodes[0]['data']
            input1_matrix = self._input_nodes[1]['data']
            sub_matrix = sub(np.array(input0_matrix, dtype=np.int64), np.array(input1_matrix, dtype=np.int64))
            return (self._node['shape'], sub_matrix.tolist())
    
    class ConcatOperator(BaseOperator):
        def run(self):
            concat_axis = self._node['axis']
            
            input0_matrix = self._input_nodes[0]['data']
            input1_matrix = self._input_nodes[1]['data']
            #concat_matrix = concat(np.array(input0_matrix, dtype=np.int64), np.array(input1_matrix, dtype=np.int64), concat_axis)
            
            concat_matrix = reduce(lambda x,y: np.concatenate((np.array(x['data']),np.array(y['data'])), axis=concat_axis), self._input_nodes)
            #reduce(lambda x,y: np.concatenate((x,y), axis=concat_axis), ops_array)
            return (self._node['shape'], concat_matrix.tolist())

    class MatmulReluOperator(MatmulOperator):
        def run(self):
            _, matmul_matrix = super().run()
            relu_matrix = np.where(np.array(matmul_matrix)>0, np.array(matmul_matrix), 0)
            return (self._node['shape'], relu_matrix.tolist())
        
    class AddReluOperator(AddOperator):
        def run(self):
            _, add_matrix = super().run()
            relu_matrix = np.where(np.array(add_matrix)>0, np.array(add_matrix), 0)
            return (self._node['shape'], relu_matrix.tolist())

    return locals()

all_generated_operators = generate_all_operator()

def make_operator(*args, **kwargs):
    cur_node = kwargs['node']
    node_class = all_generated_operators[NODE_OPERATOR_DICT[cur_node['type']]]
    return node_class(*args, **kwargs)
    
