

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

#from xir.wrapper import *
import struct
from functools import reduce
import numpy as np

actv_op = ['relu', 'relu6']

def get_input_ops_list(op):
    input_ops_list = []
    for name, temp in op.get_input_ops().items():
        input_ops_list = input_ops_list + temp
    return input_ops_list

BYTES_TYPE_BYTES = {'INT8':1, 'INT16':2, 'INT32':4,
                     'FLOAT32':4, 'FLOAT64':8,
                     'int8':1, 'int16':2, 'int32':4,
                     'float32':4, 'float64':8}

BYTES_TYPE_FORMAT = {'INT8':'b', 'INT16':'h', 'INT32':'i',
                     'FLOAT32':'f', 'FLOAT64':'d',
                     'int8':'b', 'int16':'h', 'int32':'i',
                     'float32':'f', 'float64':'d'}

def const_op_data(const_op):
    if const_op.get_type() != 'const':
        raise ValueError('Op type must be const')
    const_bytes = const_op.get_attr('data')
    bytes_type = const_op.get_attr('data_type')
    data_shape = const_op.get_attr('shape')
    data_len = reduce(lambda x,y: x*y, data_shape)
    bytes_len = BYTES_TYPE_BYTES[bytes_type]
    
    data_temp_list = []
    for i in range(data_len):
        data_temp_list.append(struct.unpack(BYTES_TYPE_FORMAT[bytes_type], 
                              const_bytes[i*bytes_len:(i+1)*bytes_len])[0])
    data_array = np.array(data_temp_list).reshape(data_shape)
    return data_array

def get_matmul_bias(op):
    input_op_num = op.get_input_num()
    if input_op_num == 3:
        input_ops = get_input_ops_list(op)
        for aop in input_ops:
            if (aop.get_type() == 'const') and ('bias' in aop.get_name()):
                return aop
    return None

def get_matmul_weights_data(op):
    attr_dict = {}
    input_op_num = op.get_input_num()
    if input_op_num >= 2:
        input_ops = get_input_ops_list(op)
        for aop in input_ops:
            if (aop.get_type() == 'const') and ('weights' in aop.get_name()):
                attr_dict['weights'] = aop
            elif aop.get_type() == 'data':
                attr_dict['vector'] = aop
    return attr_dict



'''
def extend_graph_op_dims(ordered_ops):
    for aop in ordered_ops:
        aop_name = aop.get_type()
        if aop_name is 'matmul':
            input_op_num = aop.get_input_num()
            if input_op_num == 2:
                attr_dict = get_matmul_weights_data(aop)
                aop_dim = [attr_dict['weights'].get_output_tensor().dims[0], 
                           attr_dict['vector'].get_output_tensor().dims[1]]
                aop_tensor = aop.get_output_tensor()
                aop_tensor = 
            elif input_op_num == 3:
                
                
        elif aop_name in actv_op:
            
        elif aop_name is 'mul':
        
        elif aop_name is 'eltwise':
        
        else:
            pass
'''
