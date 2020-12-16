

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
import networkx as nx
import random
import math
import struct
import binascii
import os
import json
import re
import os
from . import dctc
from . import dctc_backend
import copy

sub_matrix_w=32
sub_matrix_h=32

group_bank=32
group_id=0
vector_id=1

sgmd_num=2048
tanh_num=2048
actv_init=1

mmul_save=1
mmul_id=0

data_bit16=16
data_size=data_bit16//8
share_mem_width=512
share_mem_size=share_mem_width//data_bit16

data_op=['group', 'vector', 'data']
com_op=['matmul', 'mul', 'eltwise','sub', 'tanh', 'sigmoid', 'add', 'actv']

class Mult_bank(object):
    def __init__(self, num, size):
        self.data={}
        self.size=size
        for i in range(num):
            self.data[str(i)]=str(size*'0')

    def memory_alloc(self, num, size):
        #print('Alloc number =%d, size =%d'%(num, size))
        mem_list=self.data
        mem_zero=size*'0'
        mem_flag=size*'1'
        id_addr=[[0, 0] for i in range(num)]
        bank_mem_id=list(mem_list.keys())
        for i in range(num):
            for j in (bank_mem_id):
                addr_temp=mem_list[j].find(mem_zero)
                if(addr_temp != -1):
                    bank_mem_id.remove(j)# next not participate in cycle
                    id_addr[i][0]=int(j)
                    id_addr[i][1]=addr_temp
                    list_mem=list(mem_list[j])
                    list_mem[addr_temp:addr_temp+size]=mem_flag
                    self.data[j]=''.join(list_mem)
                    break
                else:
                    pass
                    #print(bank_mem_id)
        #print(id_addr)
        # print(self.data)
        for i, j in self.data.items():
            if(num == 0):
                continue
            else:
                self.data.pop(i)
                self.data[i]=j
            num=num-1

        return id_addr
    
    def memory_free(self, id_num, addr, size):
        mem_list=list(self.data[str(id_num)])
        mem_zero=size*'0'
        mem_list[addr : addr+size]=mem_zero
        print('Memory free id=%d, start addr=%d, size=%d'%(id_num, addr, size))
        print(self.data)
        self.data[str(id_num)]=''.join(mem_list)
        print(self.data)

    def memory_dump(self):
        print(self.data)
    
    def memory_zero(self):
        num=len(self.data)
        size=len(self.data[0])
        self.data=[size*'0' for i in range(num)]


def proc_init(edges, op_dict):
    if(op_dict[edges[0]]['type'] in data_op and op_dict[edges[1]]['type'] in com_op):
        if(op_dict[edges[0]]['type'] =='group' and op_dict[edges[1]]['type'] =='matmul'):
            op_dict[edges[1]]['m']=op_dict[edges[0]]['m']
            op_dict[edges[1]]['n']=op_dict[edges[0]]['n']
        elif(op_dict[edges[0]]['type'] =='data'):
            op_dict[edges[1]]['m']=op_dict[edges[0]]['m']
            op_dict[edges[1]]['n']=1

    #if(op_dict[edges[0]]['type'] in com_op and op_dict[edges[1]]['type'] in com_op):
    else:
        op_dict[edges[1]]['m']=op_dict[edges[0]]['m']
        op_dict[edges[1]]['n']=1

def id_add_index(bank_addr, index):
    
    bank_addr[0][0]=bank_addr[0][0]+index
    return bank_addr

def bank_memory(Group_mem, vector_mem, node_edge, op_dict, bank_loop_op):
    load_mem = Mult_bank(1, 512*32)#0
    matmul_mem = Mult_bank(2, 512*32)#1-2
    eltwise_mem = Mult_bank(1, 512*32)#0 or 3
    actv_mem = Mult_bank(2, 512*32)#4-5
    mul_mem = Mult_bank(2, 512*32)#6-7

    bank_op_loop_dict={}
    for i in bank_loop_op:
        bank_op_loop_dict[i[1]]=i[0]

    G=nx.DiGraph()
    G.add_edges_from(node_edge)
    #malloc all memory
    for i in range(len(node_edge)):
        proc_init(node_edge[i], op_dict)

    add_num=0

    # malloc memory
    #com_op=['matmul', 'mul', 'eltwise','tanh', 'sigmoid']
    for edges_temp in node_edge:

        output_edge = list(G.out_edges(edges_temp[0]))

        if(edges_temp[1] != output_edge[0][1]):
            print('skip edge:', edges_temp)
            op_dict[edges_temp[1]]['input_id_addr'].append(op_dict[edges_temp[0]]['output_id_addr'][0])
            continue

        l_op_type = op_dict[edges_temp[0]]
        r_op_type = op_dict[edges_temp[1]]
        print(edges_temp)
        

        if(edges_temp[0] in bank_op_loop_dict):
            op_dict[edges_temp[0]]['output_id_addr'].append(op_dict[bank_op_loop_dict[edges_temp[0]]]['output_id_addr'][0])
            op_dict[edges_temp[1]]['input_id_addr'].append(op_dict[bank_op_loop_dict[edges_temp[0]]]['output_id_addr'][0])
            continue

        if(l_op_type['type'] in data_op and r_op_type['type'] in com_op):
            print('data -> com')
            if(l_op_type['type']== 'data'):
                bank_addr = load_mem.memory_alloc(1, op_dict[edges_temp[0]]['m']*op_dict[edges_temp[0]]['n'])
                print(bank_addr[0])
                op_dict[edges_temp[0]]['output_id_addr'].append(bank_addr[0])
                op_dict[edges_temp[1]]['input_id_addr'].append(bank_addr[0])
            elif(l_op_type['type']== 'group'):
                bank_addr = Group_mem.memory_alloc(op_dict[edges_temp[0]]['m']*op_dict[edges_temp[0]]['n'])
                print(bank_addr)
                op_dict[edges_temp[0]]['output_id_addr'].append({'group': bank_addr})
                op_dict[edges_temp[1]]['input_id_addr'].append({'group': bank_addr})
                
            elif(l_op_type['type']== 'vector'):
                #bank_addr = vector_mem.memory_alloc(op_dict[edges_temp[0]]['m']*op_dict[edges_temp[0]]['n'])
                bank_addr = vector_mem.memory_alloc(320)
                print(bank_addr)
                op_dict[edges_temp[0]]['output_id_addr'].append({'vector': bank_addr})
                op_dict[edges_temp[1]]['input_id_addr'].append({'vector': bank_addr})
                

        elif(l_op_type['type'] in com_op and r_op_type['type'] in com_op):
            print('com -> com')
            if(l_op_type['type']== 'matmul'):
                bank_addr = matmul_mem.memory_alloc(1, op_dict[edges_temp[0]]['m'])
                bank_addr=id_add_index(bank_addr, 1)
                print(bank_addr[0])
                op_dict[edges_temp[0]]['output_id_addr'].append(bank_addr[0])
                op_dict[edges_temp[1]]['input_id_addr'].append(bank_addr[0])

            elif(l_op_type['type']== 'eltwise' or l_op_type['type']== 'sub'):
                add_num+=1
                if(add_num%2 == 0):
                    bank_addr = eltwise_mem.memory_alloc(1, op_dict[edges_temp[0]]['m']*op_dict[edges_temp[0]]['n'])
                    bank_addr=id_add_index(bank_addr, 3)
                else:
                    bank_addr = load_mem.memory_alloc(1, op_dict[edges_temp[0]]['m']*op_dict[edges_temp[0]]['n'])
                print(bank_addr[0])
                op_dict[edges_temp[0]]['output_id_addr'].append(bank_addr[0])
                op_dict[edges_temp[1]]['input_id_addr'].append(bank_addr[0])

            elif(l_op_type['type']== 'tanh' or l_op_type['type']== 'sigmoid'):
                bank_addr = actv_mem.memory_alloc(1, op_dict[edges_temp[0]]['m']*op_dict[edges_temp[0]]['n'])
                bank_addr=id_add_index(bank_addr, 4)
                print(bank_addr[0])
                op_dict[edges_temp[0]]['output_id_addr'].append(bank_addr[0])
                op_dict[edges_temp[1]]['input_id_addr'].append(bank_addr[0])           

            elif(l_op_type['type']== 'mul'):
                bank_addr = mul_mem.memory_alloc(1, op_dict[edges_temp[0]]['m']*op_dict[edges_temp[0]]['n'])
                bank_addr=id_add_index(bank_addr, 6)
                print(bank_addr[0])
                op_dict[edges_temp[0]]['output_id_addr'].append(bank_addr[0])
                op_dict[edges_temp[1]]['input_id_addr'].append(bank_addr[0])

        elif(l_op_type['type'] in com_op and r_op_type['type'] in data_op):
            print('com -> data')
            bank_addr = load_mem.memory_alloc(1, op_dict[edges_temp[0]]['m']*op_dict[edges_temp[0]]['n'])
            print(bank_addr[0])
            op_dict[edges_temp[0]]['output_id_addr'].append(bank_addr[0])
            op_dict[edges_temp[1]]['input_id_addr'].append(bank_addr[0])

        else:
            print('none')

    #op_dict=add_complement_id(op_dict)
    return op_dict
def vector_memory(vector_mem, node_edge, op_dict):

    G=nx.DiGraph()
    G.add_edges_from(node_edge)

    for edges_temp in node_edge:
        output_edge = list(G.out_edges(edges_temp[0]))
        l_op_type = op_dict[edges_temp[0]]
        r_op_type = op_dict[edges_temp[1]]
        if(edges_temp[1] != output_edge[0][1]):
            # print('skip edge:', edges_temp)
            if(l_op_type['type'] == 'vector'):
                op_dict[edges_temp[1]]['input_id_addr'][0]['vector'] = op_dict[edges_temp[0]]['output_id_addr'][0]['vector']
            continue      


        if(l_op_type['type'] in data_op and r_op_type['type'] in com_op):
            print('data -> com')
                
            if(l_op_type['type']== 'vector'):
                #bank_addr = vector_mem.memory_alloc(op_dict[edges_temp[0]]['m']*op_dict[edges_temp[0]]['n'])
                bank_addr = vector_mem.memory_alloc(320)
                print(bank_addr)
                #op_dict[edges_temp[0]]['output_id_addr'].append({'vector': bank_addr})
                print(type(bank_addr))
                op_dict[edges_temp[0]]['output_id_addr'][0]['vector'] = bank_addr
                #op_dict[edges_temp[1]]['input_id_addr'].append({'vector': bank_addr})
                op_dict[edges_temp[1]]['input_id_addr'][0]['vector'] = bank_addr
                
        else:
            print('none')
    return op_dict
