

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

import numpy as np
from xir import Graph
from xir import Op
from utils.processing import *
from utils.tools import *

def get_matmul_weights_vector(op):
    attr_dict = {}
    input_op_num = op.get_input_num()
    if input_op_num >= 2:
        input_ops = get_input_ops_list(op)
        for aop in input_ops:
            if (aop.get_type() == 'const') and ('weight' in aop.get_name()):
                attr_dict['weights'] = aop.get_name()
            #elif aop.get_type() == 'data':
            elif aop.get_type() != 'const':
                attr_dict['vector'] = aop.get_name()
    if 'weights' not in attr_dict:
        raise KeyError('Matmul op must contain weights op input')
    if 'vector' not in attr_dict:
        raise KeyError('Matmul op must contain vector op input')
    return attr_dict

def get_any_input_op(op):
    input_op_num = op.get_input_num()
    if input_op_num > 0:
        any_op = get_input_ops_list(op)[0]
    else:
        raise ValueError('This op must have at least one input op')
    return any_op.get_name()

def get_down_ops(op, graph):
    down_ops = []
    for aop in graph.get_ops():
        if (aop.get_input_num() > 0):# and (op.get_name() in [bop.get_name for bop in get_input_ops_list(aop)]):
            aop_input_names = [bop.get_name() for bop in get_input_ops_list(aop)]
            if op.get_name() in aop_input_names:
                down_ops.append(aop)
    return down_ops

def max(data,name='',quantizer=None):
    return data.max()

def min(data,name='',quantizer=None):
    return data.min()

#for quantization process
def __amplify_data(data,max,amp,method=2):
    #1 for floor, 2 for dpu round; use number, not amplified
    '''
    if method==1:
        data=np.floor(data*amp)
        data=np.clip(data, -max, max-1)
    elif method==2:
        data=data*amp
        data=np.clip(data, -max, max-1)
        data=np.where(np.logical_and(data<0,(data-np.floor(data))==0.5),np.ceil(data),np.round(data))
    '''
    data=data*amp
    return data

def normal_quant_neuron(data,maxamps=[[32768], [2048]],strides=[-1],round_method=2,
    keep_scale=True,name='',quantizer=None,on_gpu=True,as_int=False):
    #integer need not keep scale as precondition
    if as_int:
        keep_scale=False
    if len(strides)==1:
        data=__amplify_data(data,maxamps[0][0],maxamps[1][0],method=round_method)
        if keep_scale:
            data=data/maxamps[1][0]
    else:
        org_shape=data.shape
        flatten_data=data.flatten()
        pos=0
        for idx,s in enumerate(strides):
            flatten_data[pos:pos+s]=__amplify_data(flatten_data[pos:pos+s],
                maxamps[0][idx],maxamps[1][idx],method=round_method)
            if keep_scale:
                flatten_data[pos:pos+s]=flatten_data[pos:pos+s]/maxamps[1][idx]
            pos+=s
        data=flatten_data.reshape(org_shape)
    #return integer or origional dtype
    if as_int:
        assert all(m==maxamps[0][0] for m in maxamps[0]),"all max limitation should be the same"
        if maxamps[0][0]==2**7:
            return data.astype(np.int8)
        elif maxamps[0][0]==2**15:
            return data.astype(np.int16)
        else:
            raise TypeError("unexpected max found "+str(maxamps[0][0]))
    else:
        return data

def quantize_data2int(data,bn,fp,method=2):
    return normal_quant_neuron(data,maxamps=[[2**(bn-1)],[2**fp]],round_method=method,as_int=True)
