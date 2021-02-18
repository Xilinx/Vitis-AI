

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

from xir import Graph
import numpy as np
from typing import Any, Dict, List
from abc import abstractmethod
from .utils import *
from utils.data import *
from utils.tools import *

class BaseRebuilder():
    def __init__(self, *args, **kwargs):
        self.op_graph = kwargs['graph']
        self.rebuild_ops = kwargs['rebuild_ops']
    
    @abstractmethod
    def rebuild(self, configs=None):
        pass
    
    '''
    @abstractmethod
    def __get_rebuild_config(self, index, op_dict):
        pass
    
    def get_down_ops(self, op):
        down_ops = []
        for aop in self.op_graph.get_ops():
            if (aop.get_input_num() > 0):# and (op.get_name() in [bop.get_name for bop in get_input_ops_list(aop)]):
                aop_input_names = [bop.get_name() for bop in get_input_ops_list(aop)]
                if op.get_name() in aop_input_names:
                    down_ops.append(aop)
        return down_ops
    '''
class ReluRebuilder(BaseRebuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def rebuild(self, graph, configs=None):
        if not isinstance(configs, dict):
            raise ValueError('Relu op rebuild configs should be dict')
        
        for relu_op in self.rebuild_ops:
            #input_ops = [node for node in self.op_graph.get_op(relu_op).get_input_ops()]
            input_ops = [node for node in get_input_ops_list(self.op_graph.get_op(relu_op))]
            relu_rebuild_config = self.__get_rebuild_config(relu_op)
            
            #if relu_rebuild_config['relu_up_op'].get_type() != 'fix':
            input_ops[0].set_attr('fuse_relu', True)
            for down_op in relu_rebuild_config['relu_down_ops']:
                down_op.replace_input_ops(relu_rebuild_config['relu_op'], input_ops[0])
            graph.remove_op(relu_rebuild_config['relu_op'])
        
    def __get_rebuild_config(self, index):
        rebuild_config = {}
        rebuild_config['relu_op'] = self.op_graph.get_op(index)
        #rebuild_config['relu_up_op'] = self.op_graph.get_op(input_ops[0])
        
        relu_down_ops = get_down_ops(rebuild_config['relu_op'], self.op_graph)
        rebuild_config['relu_down_ops'] = relu_down_ops
                        
        return rebuild_config

class FixRebuilder(BaseRebuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def rebuild(self, graph, configs=None):
        if not isinstance(configs, dict):
            raise ValueError('Fix op rebuild configs should be dict')
        
        #print(self.rebuild_ops)
        for fix_op in self.rebuild_ops:
            #print(fix_op)
            #input_ops = [node for node in self.op_graph.get_op(fix_op).get_input_ops()]
            input_ops = [node for node in get_input_ops_list(self.op_graph.get_op(fix_op))]
            configs[input_ops[0].get_name()] = self.op_graph.get_op(fix_op).get_attrs()
            fix_rebuild_config = self.__get_rebuild_config(fix_op)
        
            for down_op in fix_rebuild_config['fix_down_ops']:
                down_op.replace_input_ops(fix_rebuild_config['fix_op'], input_ops[0])
            graph.remove_op(fix_rebuild_config['fix_op'])
        
    def __get_rebuild_config(self, index):
        rebuild_config = {}
        rebuild_config['fix_op'] = self.op_graph.get_op(index)
        #rebuild_config['fix_up_op'] = self.op_graph.get_op(op_dict[index][0])
        
        fix_down_ops = get_down_ops(rebuild_config['fix_op'], self.op_graph)
        rebuild_config['fix_down_ops'] = fix_down_ops
                        
        return rebuild_config


class MatmulRebuilder(BaseRebuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def rebuild(self, graph, configs=None):
        if not isinstance(configs, dict):
            raise ValueError('Matmul op rebuild configs should be dict')
        for matmul_op in self.rebuild_ops:
            if not self.__should_rebuild(matmul_op):
                continue
            #input_ops = [node for node in self.op_graph.get_op(matmul_op).get_input_ops()]
            input_ops = [node for node in get_input_ops_list(self.op_graph.get_op(matmul_op))]
            matmul_rebuild_config = self.__get_rebuild_config(matmul_op, input_ops)
            self.__rebuild_one_op(graph, matmul_rebuild_config, configs)
            
    def __should_rebuild(self, op):
        #if len(self.rebuild_ops[op]) == 3:
        if self.op_graph.get_op(op).get_input_num() == 3:
            return True
        else:
            return False
    
    def __get_rebuild_config(self, index, input_ops):
        rebuild_config = {}
        rebuild_config['matmul_op'] = self.op_graph.get_op(index)
        
        for node in input_ops:
            #node = self.op_graph.get_op(name)
            '''
            if node.get_type() == 'data':
                rebuild_config['input'] = node
            '''
            if (node.get_type() == 'const') and (node.get_output_tensor().ndim == 2):
                rebuild_config['weights'] = node
            elif (node.get_type() == 'const') and (node.get_output_tensor().ndim == 1):
                rebuild_config['bias'] = node
            else:
                rebuild_config['input'] = node
        
        matmul_down_ops = get_down_ops(rebuild_config['matmul_op'], self.op_graph)
        rebuild_config['matmul_down_ops'] = matmul_down_ops
        return rebuild_config
    
    def __rebuild_one_op(self, graph, rebuild_configs, fix_configs):
        matmul_op = rebuild_configs['matmul_op']
        
        wx_xt_ops: Dict[str, List[Op]] = {}
        wx_xt_ops['input'] = [rebuild_configs['input']]
        wx_xt_ops['input'].append(rebuild_configs['weights'])
        
        attrs: Dict[str, Any] = {}
        if matmul_op.get_type() == 'matmul':
            attrs['transpose_a'] = matmul_op.get_attr('transpose_a')
            attrs['transpose_b'] = matmul_op.get_attr('transpose_b')
        else:
            attrs['transpose_a'] = False
            attrs['transpose_b'] = True
        
        wx_xt_name = matmul_op.get_name() + '_matmul'
        wx_xt_mmul = graph.create_op(wx_xt_name, 'matmul', attrs=attrs, input_ops=wx_xt_ops)
        fix_configs[wx_xt_name] = fix_configs[matmul_op.get_name()]
        
        add_ops: Dict[str, List[Op]] = {}
        add_ops['input'] = [wx_xt_mmul]
        add_ops['input'].append(rebuild_configs['bias'])
        
        add_name = matmul_op.get_name() + '_eltwise'
        #add_name = matmul_op.get_name()
        #add_mmul = graph.create_op(add_name, 'eltwise', input_ops=add_ops)
        add_mmul = graph.create_op(add_name, 'add', input_ops=add_ops)
        fix_configs[add_name] = fix_configs[matmul_op.get_name()]
        
        if matmul_op.has_attr('fuse_relu'):
            add_mmul.set_attr('fuse_relu', True)
        
        for down_op in rebuild_configs['matmul_down_ops']:
            down_op.replace_input_ops(matmul_op, add_mmul)
        graph.remove_op(matmul_op)


class StridedSliceRebuilder(BaseRebuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__fix_data = kwargs['fix_data']
        self.DATA_OPS = ['const']
        self.OPERATION_OPS = ['eltwise', 'matmul', 'linear', 'add']
    
    def rebuild(self, graph, configs=None):
        if not isinstance(configs, dict):
            raise ValueError('Strided_slice op rebuild configs should be dict')
        
        while len(self.rebuild_ops) > 0:
            #any_dict_index = list(self.rebuild_ops.keys())[0]
            any_index = self.rebuild_ops[0]
            stridedslice_rebuild_type, stridedslice_rebuild_config = self.__get_rebuild_config(any_index, self.rebuild_ops)
            self.__rebuild_one_op(graph, stridedslice_rebuild_type, stridedslice_rebuild_config, configs)
            
    def __get_rebuild_config(self, index, op_dict):
        rebuild_config = {}
        
        strided_op_list = []
        #input_op_name = op_dict[index][0]
        #input_ops = [node for node in self.op_graph.get_op(index).get_input_ops()]
        input_ops = [node for node in get_input_ops_list(self.op_graph.get_op(index))]
        input_op_name = input_ops[0].get_name()
        
        for key in op_dict:
            #key_input_ops = [node for node in self.op_graph.get_op(key).get_input_ops()]
            key_input_ops = [node for node in get_input_ops_list(self.op_graph.get_op(key))]
            #if op_dict[key][0] == input_op_name:
            if key_input_ops[0].get_name() == input_op_name:
                strided_op_list.append(self.op_graph.get_op(key))
        rebuild_config['stridedslice_ops'] = strided_op_list
        
        stridedslice_up_op = self.op_graph.get_op(input_op_name)
        rebuild_config['stridedslice_up_op'] = stridedslice_up_op
        rebuild_type, rebuild_config['params_ops'] = self.__get_legal_stridedslice_param(stridedslice_up_op)
        
        stridedslice_down_ops = {}
        for op in strided_op_list:
            down_ops = get_down_ops(op, self.op_graph)
            stridedslice_down_ops[op] = down_ops
        rebuild_config['stridedslice_down_ops'] = stridedslice_down_ops
            
        for op in strided_op_list:
            #del op_dict[op.get_name()]
            op_dict.remove(op.get_name())
                        
        return rebuild_type, rebuild_config
    
    def __get_operation_stridedslice_param(self, up_op):
        param_ops_dict = {'input':None, 'weights':None, 'bias':None}
        
        input_num = up_op.get_input_num()
        #if (input_num == 2) and (up_op.get_type() == 'eltwise'):
        if (input_num == 2) and (up_op.get_type() == 'add'):
            input_ops = get_input_ops_list(up_op)
            input_ops_type = [input_op.get_type() for input_op in input_ops]
            if ('const' in input_ops_type) and (('matmul' in input_ops_type) or ('linear' in input_ops_type)):
            #if set(['const', 'matmul']) == set([input_op.get_type() for input_op in input_ops]):
                for op in input_ops:
                    if op.get_type() == 'const':
                        param_ops_dict['bias'] = op
                    elif op.get_type() in GLOBAL_VARIABLE.MATMUL_OPS:
                        if op.get_input_num() != 2:
                            raise ValueError('Can not handle this kind of strided_slice op')
                        for matmul_op in get_input_ops_list(op):
                            if op.get_type() == 'data':
                                param_ops_dict['input'] = matmul_op
                            else:
                                param_ops_dict['weights'] = matmul_op
                        
            else:
                raise ValueError('Can not handle this kind of strided_slice op')
            
        elif (input_num > 1) and (up_op.get_type() in GLOBAL_VARIABLE.MATMUL_OPS):
            param_ops = get_input_ops_list(up_op)
            for op in param_ops:
                if op.get_type() == 'data':
                    param_ops_dict['input'] = op
                elif op.get_output_tensor().ndim == 2:
                    param_ops_dict['weights'] = op
                elif op.get_output_tensor().ndim == 1:
                    param_ops_dict['bias'] = op
        else:
            raise ValueError('Can not handle this kind of strided_slice op')
        return param_ops_dict
        

    def __get_legal_stridedslice_param(self, up_op):
        if up_op.get_type() in self.DATA_OPS:
            stridedslice_type = 1
            param_ops_dict = {'const':up_op}
        elif up_op.get_type() in self.OPERATION_OPS:
            stridedslice_type = 2
            param_ops_dict = self.__get_operation_stridedslice_param(up_op)
        else:
            raise ValueError('Can not handle this kind of strided_slice op')
        return stridedslice_type, param_ops_dict
    
    def __rebuild_one_operation_op(self, graph, rebuild_configs, fix_configs):
        num = 0
        
        stridedslice_ops = rebuild_configs['stridedslice_ops']    
        stridedslice_up_op = rebuild_configs['stridedslice_up_op']
        params_ops = rebuild_configs['params_ops']
        param_input = params_ops['input']
        param_weights = params_ops['weights']
        param_bias = params_ops['bias']
        stridedslice_down_ops = rebuild_configs['stridedslice_down_ops']
        
        for op in stridedslice_ops:
            op_attrs = op.get_attrs()
            begin = op_attrs['begin']
            end = op_attrs['end']
            end_mask = op_attrs['end_mask']
            if param_weights.get_name() in self.__fix_data:
                weights_data = self.__fix_data[param_weights.get_name()]
            else:
                #weights_data = param_weights.get_attrs()['data']
                #weights_data = param_weights.get_attr('data')
                weights_data = const_op_data(param_weights)

            if end_mask == 1:
                weights_tensor = weights_data[begin[1]:end[1], :]
            else:
                weights_tensor = weights_data[begin[1]:end[1], begin[0]:end[0]]
            
            weight_op_name = param_weights.get_name()+'_split_'+str(num)
            #weights_op = graph.create_op(name=weight_op_name, kind=param_weights.get_type(), tensor=weights_tensor)
            weights_op = graph.create_const_op(weight_op_name, weights_tensor)
            fix_configs[weight_op_name] = fix_configs[param_weights.get_name()]           
 
            wx_xt_ops: Dict[str, List[Op]] = {}
            wx_xt_ops['input'] = [param_input]
            #wx_xt_ops['input'].append(param_input)
            wx_xt_ops['input'].append(weights_op)
            
            attrs: Dict[str, Any] = {}
            if stridedslice_up_op.get_type() == 'matmul':
                attrs['transpose_a'] = stridedslice_up_op.get_attr('transpose_a')
                attrs['transpose_b'] = stridedslice_up_op.get_attr('transpose_b')
            else:
                attrs['transpose_a'] = False
                attrs['transpose_b'] = True
            
            wx_xt_name = op.get_name() + '_matmul'
            wx_xt_mmul = graph.create_op(wx_xt_name, 'matmul', attrs=attrs, input_ops=wx_xt_ops)
            fix_configs[wx_xt_name] = fix_configs[op.get_name()]
            
            if param_bias is not None:
                if param_bias.get_name() in self.__fix_data:
                    bias_data = self.__fix_data[param_bias.get_name()]
                else:
                    #bias_data = param_bias.get_attrs()['data']
                    #bias_data = param_bias.get_attr('data')
                    bias_data = const_op_data(param_bias)

                bias_tensor =  bias_data[begin[1]:end[1]]
                bias_op_name = param_bias.get_name()+'_split_'+str(num)
                #bias_op = graph.create_op(name=bias_op_name, kind=param_bias.get_type(), tensor=bias_tensor)
                bias_op = graph.create_const_op(bias_op_name, bias_tensor)
                fix_configs[bias_op_name] = fix_configs[param_bias.get_name()]
                
                add_ops: Dict[str, List[Op]] = {}
                add_ops['input'] = [wx_xt_mmul]
                add_ops['input'].append(bias_op)

                add_name = op.get_name() + '_eltwise'
                #add_mmul = graph.create_op(add_name, 'eltwise', input_ops=add_ops)
                add_mmul = graph.create_op(add_name, 'add', input_ops=add_ops)
                fix_configs[add_name] = fix_configs[op.get_name()]
            
            for down_op in stridedslice_down_ops[op]:
                if param_bias is not None:
                    down_op.replace_input_ops(op, add_mmul)
                else:
                    down_op.replace_input_ops(op, wx_xt_mmul)
            graph.remove_op(op)
            num = num + 1
        graph.remove_op(stridedslice_up_op)
        
    def __rebuild_one_const_op(self, graph, rebuild_configs, fix_configs):
        num = 0
        
        stridedslice_ops = rebuild_configs['stridedslice_ops']    
        stridedslice_up_op = rebuild_configs['stridedslice_up_op']
        const_op = rebuild_configs['const']
        stridedslice_down_ops = rebuild_configs['stridedslice_down_ops']
        
        for op in stridedslice_ops:
            op_attrs = op.get_attrs()
            begin = op_attrs['begin']
            end = op_attrs['end']
            end_mask = op_attrs['end_mask']
            if const_op.get_name() in self.__fix_data:
                const_data = self.__fix_data[const_op.get_name()]
            else:
                #const_data = const_op.get_attrs()['data']
                #const_data = const_op.get_attr('data')
                const_data = const_op_data(const_op)

            if end_mask == 1:
                const_tensor = const_data[begin[1]:end[1], :]
            else:
                const_tensor = const_data[begin[1]:end[1], begin[0]:end[0]]
            
            split_op_name = const_op.get_name()+'_split_'+str(num)
            #split_op = graph.create_op(name=split_op_name, kind=const_op.get_type(), tensor=const_tensor)
            split_op = graph.create_const_op(split_op_name, const_tensor)
            fix_configs[split_op_name] = fix_configs[const_op.get_name()]           
 
            for down_op in stridedslice_down_ops[op]:
                down_op.replace_input_ops(op, split_op)
            graph.remove_op(op)
            num = num + 1
        graph.remove_op(stridedslice_up_op)
        
    
    def __rebuild_one_op(self, graph, rebuild_type, rebuild_configs, fix_configs):
        if rebuild_type == 1:
            self.__rebuild_one_const_op(graph, rebuild_configs, fix_configs)
        elif rebuild_type == 2:
            self.__rebuild_one_operation_op(graph, rebuild_configs, fix_configs)
        
        
'''
class StridedSliceRebuilder(Rebuilder):
    def __init__(self, *args, **kwargs):
        super(StridedSliceRebuilder).__init__(*args, **kwargs)
        
        
        self.graph = configs['graph']
        self.fix_data = configs['fix_data']
        self.stridedslice_ops = configs['stridedslice_ops']    
        self.stridedslice_up_op = configs['stridedslice_up_op']
        params_ops = configs['params_ops']
        self.param_input = params_ops['input']
        self.param_weights = params_ops['weights']
        self.param_bias = params_ops['bias']
        self.stridedslice_down_ops = configs['stridedslice_down_ops']
    
    def rebuild(self, fix_configs):
        num = 0
        for op in self.stridedslice_ops:
            op_attrs = op.get_attrs()
            begin = op_attrs['begin']
            end = op_attrs['end']
            end_mask = op_attrs['end_mask']
            if self.param_weights.get_name() in self.fix_data:
                weights_data = self.fix_data[self.param_weights.get_name()]
            else:
                #weights_data = self.param_weights.get_attrs()['data']
                weights_data = self.param_weights.get_attr('data')

            if end_mask == 1:
                weights_tensor = weights_data[begin[1]:end[1], :]
            else:
                weights_tensor = weights_data[begin[1]:end[1], begin[0]:end[0]]
            
            weight_op_name = self.param_weights.get_name()+'_split_'+str(num)
            weights_op = self.graph.create_op(name=weight_op_name, 
                                              kind=self.param_weights.get_type(), tensor=weights_tensor)
            fix_configs[weight_op_name] = fix_configs[self.param_weights.get_name()]           
 
            wx_xt_ops: Dict[str, List[Op]] = {}
            wx_xt_ops['input'] = [self.param_input]
            wx_xt_ops['input'].append(weights_op)
            
            attrs: Dict[str, Any] = {}
            attrs['transpose_a'] = False
            attrs['transpose_b'] = True
            
            wx_xt_name = op.get_name() + '_matmul'
            wx_xt_mmul = self.graph.create_op(wx_xt_name, 'matmul', attrs=attrs, input_ops=wx_xt_ops)
            fix_configs[wx_xt_name] = fix_configs[op.get_name()]
            
            if self.param_bias is not None:
                if self.param_bias.get_name() in self.fix_data:
                    bias_data = self.fix_data[self.param_bias.get_name()]
                else:
                    #bias_data = self.param_bias.get_attrs()['data']
                    bias_data = self.param_bias.get_attr('data')

                bias_tensor =  bias_data[begin[1]:end[1]]
                bias_op_name = self.param_bias.get_name()+'_split_'+str(num)
                bias_op = self.graph.create_op(name=bias_op_name, 
                                               kind=self.param_bias.get_type(), tensor=bias_tensor)
                fix_configs[bias_op_name] = fix_configs[self.param_bias.get_name()]
                
                add_ops: Dict[str, List[Op]] = {}
                add_ops['input'] = [wx_xt_mmul]
                add_ops['input'].append(bias_op)
                
                add_shape = wx_xt_mmul.get_output_tensor().dims
                add_tensor = np.zeros(add_shape, dtype=np.int64)
                
                add_name = op.get_name() + '_eltwise'
				#add_mmul = self.graph.create_op(add_name, 'eltwise', input_ops=add_ops, tensor=add_tensor)
                add_mmul = self.graph.create_op(add_name, 'add', input_ops=add_ops, tensor=add_tensor)
                fix_configs[add_name] = fix_configs[op.get_name()]
            
            for down_op in self.stridedslice_down_ops[op]:
                if self.param_bias is not None:
                    down_op.replace_input_ops(op, add_mmul)
                else:
                    down_op.replace_input_ops(op, wx_xt_mmul)
            self.graph.remove_op(op)
            num = num + 1
        self.graph.remove_op(self.stridedslice_up_op)
        

class FixRebuilder():
    def __init__(self, configs):
        self.graph = configs['graph']
        self.fix_op = configs['fix_op']    
        self.fix_up_op = configs['fix_up_op']
        self.fix_down_ops = configs['fix_down_ops']
    
    def rebuild(self):
        for down_op in self.fix_down_ops:
            down_op.replace_input_ops(self.fix_op, self.fix_up_op)
        self.graph.remove_op(self.fix_op)
'''
