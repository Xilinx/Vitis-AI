

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
import os
import json
from . import dctc
from . import dctc_backend
from . import dctc_backend_u25
from . import dctc_memory
from . import dctc_draw
from . import dctc_config
from . import ddr_malloc
from . import bank_malloc
from . import bank_malloc_u25
import copy
import sys
from functools import reduce
from ctypes import *
from . import bin_gen
from . import u25_bin_gen
from . import read_instr
from utils.data import *

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

def parse_info(layer_node_info, layer_direction_dict):
    
    layer_node_info_dict={}
    for i in range(len(layer_node_info)):
        temp_node_info={}
        for j in range(len(layer_node_info[i])):
            temp =  layer_node_info[i][j]
            temp_node_info[layer_node_info[i][j]['name']]=temp
            if(layer_node_info[i][j]['type'] == 'vector'):
                temp_node_info[layer_node_info[i][j]['name']]['shape'].reverse()
            elif(layer_node_info[i][j]['type'] == 'data'):
                mult_temp = temp_node_info[layer_node_info[i][j]['name']]['shape']
                temp_node_info[layer_node_info[i][j]['name']]['shape'] = [reduce(lambda x,y:x * y,mult_temp), 1]

        layer_node_info_dict[i]=temp_node_info
    
    layer_direction = []
    for layers_num in range(len(layer_direction_dict)):
        layer_direction.append(layer_direction_dict[layers_num]['direction'])
    
    layer_direction_index = list(range(len(layer_direction)))    
    layer_direction_index[0] = 1 if layer_direction[0] == 'forward' else 0
    for i in range(len(layer_direction)-1):
        if layer_direction[i+1] == layer_direction[i]:
            layer_direction_index[i+1] = 1
        else:
            layer_direction_index[i+1] = 0
    
    return layer_node_info_dict, layer_direction_index

def pack_ins(ins_dict, op_dict):
    com_dict={}
    for i, j in ins_dict.items():
        if(j['name'] == 'load'):
            com_dict[i]=copy.deepcopy(dict(j, **op_dict[j['edge_name'][0]]))
            com_dict[i]['bank_size']=com_dict[i]['m']*com_dict[i]['n']
            del com_dict[i]['data']
            
        elif(j['name'] in com_op):
            #combine 2 dict
            com_dict[i]=copy.deepcopy(dict(j, **op_dict[j['edge_name'][1]]))
            if(j['name'] == 'add'):
                
                com_dict[i]['add_count']=len(com_dict[i]['input_id_addr'])
                
        else:
            #save op
            com_dict[i]=copy.deepcopy(dict(j, **op_dict[j['edge_name'][1]]))
            com_dict[i]['bank_size']=com_dict[i]['m']*com_dict[i]['n']
        
    return com_dict

def add_dpon_dpby(ins_list):
    
    for ins_dict in ins_list:
        print(ins_dict[1]['dpon'])
        print(ins_dict[len(ins_dict)]['dpby'])
        if(ins_dict[1]['dpon'] == ['none'] and ins_dict[len(ins_dict)]['dpby'] == ['none']):
            print('dpon and dpby check pass')
        else:
            print('dpon and dpby check False')
            exit
    for i in range(len(ins_list)-1):
        ins_list[i][len(ins_list[i])]['dpby'][0] = ins_list[i+1][1]['name']
        ins_list[i+1][1]['dpon'][0] = ins_list[i][len(ins_list[i])]['name']
    
    return ins_list

def print_dict(op_dict):
    for i, j in op_dict.items():
        print(("%s : %s") % (i, j))

def write_dot(list_lstm, save_name):
    for i in range(len(list_lstm)):
        for j in range(len(list_lstm[i])):
            list_lstm[i][j]=list_lstm[i][j].replace(save_name, '',1)
            print(list_lstm[i][j])
    lstm_graph=nx.DiGraph()
    lstm_graph.add_edges_from(list_lstm)
    nx.draw(lstm_graph, with_labels=True)
    save_name=save_name+'.dot'
    nx.drawing.nx_pydot.write_dot(lstm_graph, os.path.join('data', save_name))

def save_fp_json(input_fp, output_fp, save_path):
    fp_json=[{'lstm_in_float2fix': input_fp, 'lstm_out_fix2float': output_fp}]
    fp_json_write = json.dumps(fp_json, indent=1)
    with open(save_path, encoding='utf-8', mode='w') as f:
        f.write(fp_json_write)

def load_static(ins,ht_name):
    for i in range(len(ins)):
        if(ins[i+1]['op_name'] == ht_name):
            ins[i+1]['load_static'] = 1

def instruction_generate(layers_node_edge, 
                         layers_node_info_dict, 
                         layers_node_cycle_ops, 
                         layer_name_list,
                         layer_direction_dict,
                         device='u50'):
    SAVE_PATH=GLOBAL_VARIABLE.INSTRUCTIONS_PATH
    if not (os.path.exists(SAVE_PATH) and os.path.isdir(SAVE_PATH)):
        os.makedirs(SAVE_PATH)
    DEVICE = device
    
    layers_node_info_dict1, layer_direction = parse_info(layers_node_info_dict, layer_direction_dict)
    
    #DDR_mem=dctc_memory.Bank(512*30720)
    DDR_mem=dctc_memory.Bank(1024*1024*32)
    Group_mem=dctc_memory.Bank(65536*32)
    vector_mem=dctc_memory.Bank(512*32)
    actv_mem=dctc_memory.Mult_bank(1, 512*32)
    #write_dot(layers_node_edge[0], layer_name_list[0])
    node_info_dict =copy.deepcopy(layers_node_info_dict1)
    node_edge      =copy.deepcopy(layers_node_edge)
    cycle_ops      =copy.deepcopy(layers_node_cycle_ops)
    op_dict =       ddr_malloc.parser_nodes(node_info_dict)
    ddr_loop_op =   ddr_malloc.ddr_loop(node_info_dict, node_edge, cycle_ops)
    bank_loop_op=   ddr_malloc.bank_loop(node_info_dict, node_edge, cycle_ops)
    op_dict = ddr_malloc.malloc_ddr(DDR_mem, ddr_loop_op, op_dict)
    ddr_address=    ddr_malloc.ddr_address_dict(op_dict)
    ddr_init_str, xt_name, ht_name =ddr_malloc.pack_ddr_data(DEVICE, ddr_address, op_dict, node_info_dict, ddr_loop_op)
    save_fp_json(op_dict[xt_name]['fp'], op_dict[ht_name[-1]]['fp'], os.path.join(SAVE_PATH, 'fp.json'))
    
    # ddr_init_txt=   ddr_malloc.string_to_hex_str(ddr_init_str)
    # with open(os.path.join(SAVE_PATH, 'ddr_init.txt'),'w',encoding='utf-8') as f:
    #     f.write(ddr_init_txt)
        
    ddr_init_txt_orign = ddr_malloc.string_to_hex_str_orign(ddr_init_str)
    with open(os.path.join(SAVE_PATH, 'ddr_init_orign.txt'),'w',encoding='utf-8') as f:
        f.write(ddr_init_txt_orign)
    
    np.savetxt(os.path.join(SAVE_PATH, 'ddr_init_str.txt'), ddr_init_str, fmt='%d')

    if(DEVICE == 'u50'):
        instr_list=[]
        input_size={}
        for layer_index in range(len(layer_name_list)):
            input_size[layer_index]={}
            Group_mem.memory_zero()
            vector_mem.memory_zero()
            
            bank_malloc.init_mn(node_edge[layer_index], op_dict)
            op_dict = bank_malloc.group_vector_memory(Group_mem, vector_mem, node_edge[layer_index], op_dict, layer_direction[layer_index], ht_name[layer_index], input_size[layer_index])
            
            com_dict_init= dctc.process_edges_init(node_edge[layer_index], node_info_dict[layer_index])
            ins_dict_init = pack_ins(com_dict_init, op_dict)
            
            ins_temp, all_time = dctc_draw.ins_fetch({0:ins_dict_init})

            op_dict = bank_malloc.bank_new_memory(node_edge[layer_index], node_info_dict[layer_index], layer_index, op_dict, bank_loop_op, ins_temp, all_time)
            ins_dict_init = pack_ins(com_dict_init, op_dict)
            load_static(ins_dict_init, ht_name[layer_index])
            instr_list.append(copy.deepcopy(ins_dict_init))
            
            com_dict_run = dctc.process_edges_run(node_edge[layer_index], node_info_dict[layer_index])
            ins_dict_run = pack_ins(com_dict_run, op_dict)
            instr_list.append(copy.deepcopy(ins_dict_run))

        instr_list.append({1: {'name': 'end', 'dpon': ['none'], 'dpby': ['none']}})
        add_dpon_dpby(instr_list)
        # for i in range(len(instr_list)):
        #     print_dict(instr_list[i])
        # exit()
        instr_list_4=copy.deepcopy(instr_list)
        instr_list_3=copy.deepcopy(instr_list)

        intsr_txt = ''
        line_number_4 = []
        for i in range(len(instr_list_4)):
            print_dict(instr_list_4[i])
            temp_instr  = dctc_backend.dict_to_ins(instr_list_4[i], 4)
            line_number_4.append(temp_instr.count('\r\n'))
            intsr_txt  += temp_instr
        
        with open(os.path.join(SAVE_PATH, 'instr_ac_4.txt'),'w',encoding='utf-8') as f:
            f.write(intsr_txt)

        intsr_txt = ''
        line_number_3 = []
        for i in range(len(instr_list_3)):
            print_dict(instr_list_3[i])
            temp_instr  = dctc_backend.dict_to_ins(instr_list_3[i], 3)
            line_number_3.append(temp_instr.count('\r\n'))
            intsr_txt  += temp_instr

        with open(os.path.join(SAVE_PATH, 'instr_ac_3.txt'),'w',encoding='utf-8') as f:
            f.write(intsr_txt)
            
        line_number_4[-3]=line_number_4[-3]-4
        line_number_4[-1]=line_number_4[-1]+4
        line_number_3[-3]=line_number_3[-3]-3
        line_number_3[-1]=line_number_3[-1]+3
        print(input_size)
        dctc_config.save_config(SAVE_PATH, input_size, line_number_4, line_number_3)
        print('line_number_batch4 -> ', line_number_4)
        print('line_number_batch3 -> ', line_number_3)
        bin_gen.gen_bin(layer_name_list, line_number_3, line_number_4, SAVE_PATH)
        read_instr.read_is(layer_name_list, line_number_3, line_number_4, 0x7000000, SAVE_PATH)
    
    elif(DEVICE == 'u25'):
        instr_list=[]
        instr_actv_dict = dctc_backend.init_actv(op_dict, actv_mem)

        for layer_index in range(len(layer_name_list)):
                Group_mem.memory_zero()
                op_dict = bank_malloc_u25.bank_memory(Group_mem, vector_mem, node_edge[layer_index], op_dict, bank_loop_op)
                com_dict_init= dctc.process_edges_init(node_edge[layer_index], node_info_dict[layer_index], ht_name[layer_index])
                ins_dict_init = pack_ins(com_dict_init, op_dict)
                instr_list.append(copy.deepcopy(ins_dict_init))
                #print_dict(ins_dict_init)
                op_dict = bank_malloc_u25.vector_memory(vector_mem, node_edge[layer_index], op_dict)
                com_dict_run = dctc.process_edges_run(node_edge[layer_index], node_info_dict[layer_index])
                ins_dict_run = pack_ins(com_dict_run, op_dict)
                instr_list.append(copy.deepcopy(ins_dict_run))

        instr_list.insert(0, instr_actv_dict)
        instr_list.append({1: {'name': 'end', 'dpon': ['none'], 'dpby': ['none']}})
        add_dpon_dpby(instr_list)
        
        intsr_txt = ''
        for i in range(len(instr_list)):
            intsr_txt  += dctc_backend_u25.dict_to_ins(instr_list[i])
            print_dict(instr_list[i])

        with open(os.path.join(SAVE_PATH, 'instr_ac.txt'),'w',encoding='utf-8') as f:
            f.write(intsr_txt)

        u25_bin_gen.gen_bin(layer_name_list, SAVE_PATH)
