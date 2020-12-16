

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
import copy
#import numba
#from numba import jit

sub_matrix_w=32
sub_matrix_h=32
com_matrix_w=64

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

class Bank:
    def __init__(self, size, start=0):
        self.data=size*'0'
        self.size=size
        self.unuse_addr=[[start, start+size-1]]

    def old_memory_alloc(self, size):
        mem_str=self.data
        mem_zero=size*'0'
        mem_flag=size*'1'
        addr=mem_str.find(mem_zero)
        if(addr == -1):
            print('not have enougth meory')
            exit()
        list_mem=list(mem_str)
        list_mem[addr : addr+size]=mem_flag
        self.data=''.join(list_mem)

        #print(('size: %d addr: %d')%(size, addr))
        return addr
    def memory_alloc(self, size):
        addr=''
        for segment_addr in self.unuse_addr:
            if(segment_addr[1]-segment_addr[0] < (size-1)):
                break
            else:
                addr = segment_addr[0]
                print('Alloc size: ', size,'Alloc addr start: ',addr)
                segment_addr[0] = segment_addr[0]+size
        #print(self.unuse_addr)
        return addr

    def memory_free(self, addr, size):
        list_mem=list(self.data)
        mem_zero=size*'0'
        list_mem[addr : addr+size]=mem_zero
        self.data=''.join(list_mem)

    def memory_dump(self):
        print(self.data)

    def memory_zero(self):
        size=len(self.data)
        self.data=size*'0'


def json_parser(path):
    layer_node_edge={}
    layer_node_info={}
    layer_cycle_ops={}
    layer_name_list=[]
    with open(path+'model_config.json', 'r') as f:
        layer_info = json.load(f)
    for layers_num in range(len(layer_info)):
        layer_name_list.append({'name':layer_info[layers_num]['name'], 'direction':layer_info[layers_num]['direction']})
        #print('Layer name:', layers_num,layer_info['layer'][layers_num])
        with open(path+layer_info[layers_num]['name']+'/'+'nodes_edge.json', 'r') as f:
            layer_node_edge[layers_num] = json.load(f)
        with open(path+layer_info[layers_num]['name']+'/'+'nodes_info.json', 'r') as f:
            layer_node_info[layers_num] = json.load(f)
        with open(path+layer_info[layers_num]['name']+'/'+'cycle_ops.json', 'r') as f:
            layer_cycle_ops[layers_num] = json.load(f)

    layer_node_info_dict={}
    for i in range(len(layer_node_info)):
        temp_node_info={}
        for j in range(len(layer_node_info[i])):
            temp =  layer_node_info[i][j]
            temp_node_info[layer_node_info[i][j]['name']]=temp
        layer_node_info_dict[i]=temp_node_info

    return layer_node_edge, layer_node_info_dict, layer_cycle_ops, layer_name_list

def print_dict(op_dict):
    for i, j in op_dict.items():
        print(("%s : %s") % (i, j))

def parser_nodes(node_info):
    op_dict={}
    for i in node_info:
        for key ,value in node_info[i].items():
            #op_dict[key]={'type':node_info[key]['type'],'l_count':0, 'r_count':0, 'address':0, 'm':0, 'n':0}
            #matrix_m=math.ceil(node_info[i][key]['shape'][0]/sub_matrix_w)*sub_matrix_w
            if(node_info[i][key]['type'] == 'group'):
                matrix_m=math.ceil(node_info[i][key]['shape'][0]/com_matrix_w)*com_matrix_w
            else:
                matrix_m=math.ceil(node_info[i][key]['shape'][0]/sub_matrix_w)*sub_matrix_w

            if(node_info[i][key]['shape'][1] == 1):
                matrix_n =1
            else:
                matrix_n=math.ceil(node_info[i][key]['shape'][1]/sub_matrix_h)*sub_matrix_h
            if(node_info[i][key]['type'] in data_op):
                op_dict[key]={'op_name': key, 'type':node_info[i][key]['type'],'m':matrix_m, 'n':matrix_n, 'input_id_addr':[], 'output_id_addr':[], 'data': node_info[i][key]['data'],'fix':node_info[i][key]['shift'], 'fp':node_info[i][key]['fp']}     
            else:
                op_dict[key]={'op_name': key, 'type':node_info[i][key]['type'],'m':matrix_m, 'n':matrix_n, 'input_id_addr':[], 'output_id_addr':[], 'fix':node_info[i][key]['shift'], 'fp':node_info[i][key]['fp']}     
            
    return op_dict

def ddr_loop(node_info_dict, node_edge, cycle_ops):
    loop_op=[]
    for j in range(len(node_info_dict)):
        G=nx.DiGraph()
        G.add_edges_from(node_edge[j])    
        for i in cycle_ops[j]:
            if(node_info_dict[j][i[0]]['type'] in data_op):
                if(node_info_dict[j][i[1]]['type'] in com_op):
                    out_edge = list(G.out_edges(i[1]))
                    if(node_info_dict[j][out_edge[0][1]]['type'] in data_op):
                        loop_op.append([i[0], out_edge[0][1]])
        
    print('ddr  loop op:', loop_op)

    return loop_op
    
def bank_loop(node_info_dict, node_edge, cycle_ops):
    loop_op=[]
    for j in range(len(node_info_dict)):
        G=nx.DiGraph()
        G.add_edges_from(node_edge[j])    
        for i in cycle_ops[j]:
            if(node_info_dict[j][i[0]]['type'] in data_op):
                if(node_info_dict[j][i[1]]['type'] in com_op):
                    out_edge = list(G.out_edges(i[1]))
                    if(node_info_dict[j][out_edge[0][1]]['type'] in com_op):
                        loop_op.append([i[0], i[1]])
    print('bank loop op:', loop_op)
    return loop_op

def malloc_ddr(DDR_mem, ddr_loop_op, op_dict):
    #print(op_dict)
    loop_op=[]
    input_size=[]
    for i in ddr_loop_op:
        loop_op.append(i[1])

    for i, j in op_dict.items():

        if (j['type'] == 'group') or (j['type']=='data'):
            if(i in loop_op):
                continue#跳过循环op
            else:
                op_dict[i]['input_id_addr'].append({'ddr_addr': DDR_mem.memory_alloc(op_dict[i]['m']*op_dict[i]['n'])})
            print(i)
            print(op_dict[i]['input_id_addr'])

    for i, j in op_dict.items():

        if j['type'] == 'vector':
            op_dict[i]['input_id_addr'].append({'ddr_addr':DDR_mem.memory_alloc(1000*op_dict[i]['m']*1)})
            print(i)
            print('input size: ', op_dict[i]['m'])
            input_size.append(op_dict[i]['m'])
            print(op_dict[i]['input_id_addr'])

    for i in ddr_loop_op:#给重复的loop赋值地址
        
        if(op_dict[i[1]]['type'] in data_op):
            print('com -> data:', i[1], '->ddr',i)
            op_dict[i[1]]['output_id_addr'].append({'ddr_addr': op_dict[i[0]]['input_id_addr'][0]['ddr_addr']})

    return op_dict

def ddr_address_dict(op_dict):
    ddr_address={}
    for op_index, j in op_dict.items():
        if(j['type']=='data' or j['type']=='group' or j['type']=='vector'):
            #print('%s ddr address: %d')%(op_index, j['address'])
            if(re.search('result', op_index)):
                ddr_address[op_index]=j['output_id_addr'][0]['ddr_addr']
            else:
                ddr_address[op_index]=j['input_id_addr'][0]['ddr_addr']
            
    ddr_address_list =(sorted(ddr_address.items(), key = lambda kv:(kv[1], kv[0])))
    for i in ddr_address_list:
        print(i)
    return ddr_address_list

def read_xt_and_data( op_dict, node_info_dict, ddr_loop_op):
    vector_op=[]
    input_frames={}
    for i ,j in node_info_dict.items():
        if(j['type']=='vector'):
            for loop_index in ddr_loop_op:
                if(i != loop_index[0]):
                    vector_op.append(i)
    print('xt =:', vector_op[0])
    ht_name = ddr_loop_op[0][0]
    print('ht =:', ht_name)

    xt_temp=np.zeros((op_dict[vector_op[0]]['m'], op_dict[vector_op[0]]['n']),dtype=np.int)
    # with open(os.path.join(path, 'model_config.json'), 'r') as f:
    #     layer_info = json.load(f)
    # input_path=os.path.join(path, 'deploy_check_data_int', layer_info[0]['name'])
    # dir_list=os.listdir(input_path)
    # for i in range(len(dir_list)):
    #     index_name='/frame_'+str(i)+'/'+vector_op[0]+'_fix.txt'
    #     input_frame=np.loadtxt(input_path+index_name, dtype=int)
    #     input_frames[i]=input_frame.reshape(1, len(input_frame))

    # xt_temp=input_frames[0]
    # for i in range(1, len(input_frames)):
    #     xt_temp=np.vstack((xt_temp, input_frames[i]))

    return vector_op[0], ht_name, xt_temp
    

################################################################################################################
def pack_ddr_data(DEVICE, ddr_address, op_dict, node_info_dict, ddr_loop_op):
    data_temp={}
    del_vector_name=[]
    layers_ht_name=[]
    xt_name, ht_name, xt_data_origin=read_xt_and_data(op_dict, node_info_dict[0], ddr_loop_op)
    xt_data = div_matrix(op_dict[xt_name]['n']*1000, op_dict[xt_name]['m'], xt_data_origin)
    for i ,j in op_dict.items():
        if(j['type'] == 'vector'):
            del_vector_name.append(i)
    for i in ddr_loop_op:
        layers_ht_name.append(i[0])
        del_vector_name.append(i[1])
    print(del_vector_name)
    del_vector_name.remove(xt_name)
    del_vector_name.remove(ht_name)
    print(del_vector_name)

    for i, j in ddr_address:
        if(i in del_vector_name):
            continue
        elif(i == xt_name ):
            data_temp[xt_name]=xt_data
            print('pack xt')
        elif(i ==ht_name):
            data_temp[ht_name]=np.zeros((op_dict[ht_name]['m']*1000, op_dict[ht_name]['n']),dtype=np.int)
            print('pack ht')

        else:
            print(i)
            temp_data = np.array(op_dict[i]['data'], int)
            print('pack ddr', i, temp_data.shape)

            if(len(temp_data.shape) == 1):
                print(i)
                temp_data=temp_data.reshape(temp_data.shape[0],1)
                data_temp[i]=div_matrix(op_dict[i]['m'], op_dict[i]['n'], temp_data)
            elif(op_dict[i]['type'] == 'group'):
                temp_data=div_matrix(op_dict[i]['m'], op_dict[i]['n'], temp_data)
                if(DEVICE == 'u25'):
                    data_temp[i]=rearrange_ddr_data(temp_data)
                elif(DEVICE == 'u50'):
                    data_temp[i]=interleave(temp_data)
            else:
                if(temp_data.shape[0] == 1):
                    temp_data=temp_data.reshape(temp_data.shape[1], 1)
                data_temp[i]=div_matrix(op_dict[i]['m'], op_dict[i]['n'], temp_data)
             
    out_temp=np.zeros((1,8), dtype=np.int)
    for i, j in ddr_address:
        if(i in del_vector_name):
            continue        
        if(j !=(out_temp.shape[0]*out_temp.shape[1]-8)):
            print(i, j)
            print('pack ddr error')

        out_temp=np.vstack((out_temp, data_temp[i].reshape(int(data_temp[i].shape[0]*data_temp[i].shape[1]/8), 8)))
    out_temp=np.delete(out_temp, 0, axis=0)
    return out_temp, xt_name, layers_ht_name


def rearrange_ddr_data(data):
    #print(data.shape)
    m=data.shape[0]
    n=data.shape[1]
    if(m==32):
        return data
    temp=np.zeros((m, n),dtype=np.int)
    loop_num = m//32
    for i in range(group_bank):
        for j in range(m//group_bank):
            #print(i+j*group_bank)
            #data[i+j*group_bank]
            temp[i*loop_num+j]=data[i+j*group_bank]
    return temp

def interleave(data):
    (row, col) = data.shape

    inter_data = np.empty((int(row/2), 2*col), dtype = data.dtype)
    for i in range(int(row/2)):
      inter_data[i, 0::2] = data[64*int(i/32)+i%32, :]
      inter_data[i, 1::2] = data[64*int(i/32)+i%32+32, :]

      #print("interleave array index: %d source array index is: %d %d" %(i, 64*int(i/32)+i%32, 64*int(i/32)+i%32+32))
    print('new shape:', inter_data.shape)
    inter_data = rearrange_ddr_data(inter_data)
    return inter_data

def div_matrix(m, n, matrix):
    #print(matrix.shape)
    w=matrix.shape[0]
    h=matrix.shape[1]
    temp=np.zeros((m, n),dtype=np.int)
    for i in range(w):
        temp[i, :matrix.shape[1]] = matrix[i]
    return temp

def string_to_hex_str(matrix): 
    w=matrix.shape[0]
    h=matrix.shape[1]
    print('DDR size:',matrix.shape)
    result_line=''
    for hbm_bank in range(16):
        #bank_addr=hbm_bank*1024*1024*256
        bank_addr=(hbm_bank)*1024*1024*256
        for i in range(w):
            for j in range(h):
                if(j == 0):
                    temp=binascii.b2a_hex(struct.pack('>h', matrix[i][0]))
                    result = temp.decode('utf-8','ignore')
                else:
                    temp=binascii.b2a_hex(struct.pack('>h', matrix[i][j]))
                    temp = temp.decode('utf-8','ignore')        
                    result=temp+result
            line_num=i*(j+1)*2+bank_addr
            line=(binascii.b2a_hex(struct.pack('>Q',line_num ))).decode('utf-8','ignore')  
            result_line=result_line+('0x'+line+' '+':'+' '+result.upper()+'\r\n')

    return result_line

def string_to_hex_str_orign(matrix): 
    w=matrix.shape[0]
    h=matrix.shape[1]
    print('DDR size:',matrix.shape)
    result_line=''
    for i in range(w):
        for j in range(h):
            if(j == 0):
                temp=binascii.b2a_hex(struct.pack('>h', matrix[i][0]))
                result = temp.decode('utf-8','ignore')
            else:
                temp=binascii.b2a_hex(struct.pack('>h', matrix[i][j]))
                temp = temp.decode('utf-8','ignore')        
                result=temp+result
        line_num=i*(j+1)*2
        line=(binascii.b2a_hex(struct.pack('>I',line_num ))).decode('utf-8','ignore')  
        result_line=result_line+('0x'+line+' '+':'+' '+result.upper()+'\r\n')

    return result_line

def write_dot(list_lstm):
    lstm_graph=nx.DiGraph()
    lstm_graph.add_edges_from(list_lstm)
    nx.draw(lstm_graph, with_labels=True)
    nx.drawing.nx_pydot.write_dot(lstm_graph, 'data/temp.dot')

if __name__ == "__main__":
    PATH='example/lstm_25_tf/'
    #PATH='example/AlternatingLSTM_xmodel/'
    layers_node_edge, layers_node_info_dict, layers_node_cycle_ops, layer_name_list = json_parser(PATH)
    DDR_mem=Bank(512*30720)

    #write_dot(layers_node_edge[0])
    node_info_dict=copy.deepcopy(layers_node_info_dict)
    node_edge     =copy.deepcopy(layers_node_edge)
    cycle_ops     =copy.deepcopy(layers_node_cycle_ops)

    op_dict = parser_nodes(node_info_dict)
    ddr_loop_op = ddr_loop(node_info_dict, node_edge, cycle_ops)
    op_dict = malloc_ddr(DDR_mem, ddr_loop_op, op_dict)
    ddr_address=ddr_address_dict(op_dict)
    ddr_init_str, ht_name =pack_ddr_data(PATH, ddr_address, op_dict, node_info_dict, ddr_loop_op)
    np.savetxt('data/ddr_init_str.txt', ddr_init_str, fmt='%d')
    ddr_init_txt=string_to_hex_str(ddr_init_str)
    with open('data/ddr_init.txt','w',encoding='utf-8') as f:
        f.write(ddr_init_txt)
