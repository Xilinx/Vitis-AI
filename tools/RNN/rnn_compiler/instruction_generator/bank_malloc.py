

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
import re
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
        print(id_addr)
        # print(self.data)
        for i, j in self.data.items():
            if(num == 0):
                continue
            else:
                self.data.pop(i)
                self.data[i]=j
            num=num-1

        return id_addr

    def unuse_memory_alloc(self, unuse, num, size):
        #print('Alloc number =%d, size =%d'%(num, size))
        if((num+len(unuse)) > len(self.data)):
            print('unuse + num > all bank num')
            exit()
        mem_list=self.data
        mem_zero=size*'0'
        mem_flag=size*'1'
        id_addr=[[0, 0] for i in range(num)]
        bank_mem_id=list(mem_list.keys())
        for del_index in unuse:
            bank_mem_id.remove(str(del_index))

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
        print(id_addr)
        # print(self.data)
        for i, j in self.data.items():
            if(num == 0):
                continue
            else:
                self.data.pop(i)
                self.data[i]=j
            num=num-1

        return id_addr

    def spec_memory_alloc(self, num, size):
        #print('Alloc number =%d, size =%d'%(num, size))
        mem_list=self.data
        mem_zero=size*'0'
        mem_flag=size*'1'
        addr=mem_list[num].find(mem_zero)
        if(addr == -1):
            print('not have enougth memory')
            exit()
        list_mem=list(mem_list[num])
        list_mem[addr : addr+size]=mem_flag
        self.data[num]=''.join(list_mem)

        print(('size: %d addr: %d')%(size, addr))
        return addr
        
    
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

def init_mn(node_edge, op_dict):
    for i in range(len(node_edge)):
        proc_init(node_edge[i], op_dict)

def bank_memory(Group_mem, vector_mem, node_edge, op_dict, bank_loop_op):

    load_save_mem = Mult_bank(2, 512*32) #0  load bias save result
    #add_mem = Mult_bank(1, 512*32) #1  mmul+bias or mmul+result
    
    matmul_mem = Mult_bank(1, 512*32)#2-3
    actv_emul_mem = Mult_bank(2, 512*32)#4-5
    eltwise_mem = Mult_bank(2, 512*32)#1-6-7
    
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
                
                if(re.search('input_2', l_op_type['op_name'])):# load ct-1
                    bank_addr = eltwise_mem.memory_alloc(1, op_dict[edges_temp[0]]['m']*op_dict[edges_temp[0]]['n'])
                    bank_addr=id_add_index(bank_addr, 6)
                else:
                    bank_addr = load_save_mem.memory_alloc(1, op_dict[edges_temp[0]]['m']*op_dict[edges_temp[0]]['n'])
                    #bank_addr=id_add_index(bank_addr, 3)
                
                print(bank_addr)
                op_dict[edges_temp[0]]['output_id_addr'].append(bank_addr[0])
                op_dict[edges_temp[1]]['input_id_addr'].append(bank_addr[0])
            elif(l_op_type['type']== 'group'):
                bank_addr = Group_mem.memory_alloc(op_dict[edges_temp[0]]['m']*op_dict[edges_temp[0]]['n'])
                print(bank_addr)
                op_dict[edges_temp[0]]['output_id_addr'].append({'group': bank_addr})
                op_dict[edges_temp[1]]['input_id_addr'].append({'group': bank_addr})
                
            elif(l_op_type['type']== 'vector'):
                bank_addr = vector_mem.memory_alloc(op_dict[edges_temp[0]]['m']*op_dict[edges_temp[0]]['n'])
                #bank_addr = vector_mem.memory_alloc(320)
                print(bank_addr)
                op_dict[edges_temp[0]]['output_id_addr'].append({'vector': bank_addr})
                op_dict[edges_temp[1]]['input_id_addr'].append({'vector': bank_addr})
                

        elif(l_op_type['type'] in com_op and r_op_type['type'] in com_op):
            print('com -> com')
            if(l_op_type['type']== 'matmul'):
                bank_addr = matmul_mem.memory_alloc(1, op_dict[edges_temp[0]]['m']//2)
                bank_addr1 = copy.deepcopy(bank_addr)
                bank_addr2 = copy.deepcopy(bank_addr)
                bank_addr1=id_add_index(bank_addr1, 2)
                bank_addr2=id_add_index(bank_addr2, 3)
                print(bank_addr1)
                print(bank_addr2)
                op_dict[edges_temp[0]]['output_id_addr'].append(bank_addr1[0])
                op_dict[edges_temp[0]]['output_id_addr'].append(bank_addr2[0])
                op_dict[edges_temp[1]]['input_id_addr'].append(bank_addr1[0])
                op_dict[edges_temp[1]]['input_id_addr'].append(bank_addr2[0])

            elif(l_op_type['type']== 'eltwise' or l_op_type['type']== 'sub'):
                if(len(op_dict[edges_temp[0]]['input_id_addr']) == 2):
                    print('!!', l_op_type['op_name'])
                    bank_addr = eltwise_mem.memory_alloc(1, op_dict[edges_temp[0]]['m']*op_dict[edges_temp[0]]['n'])
                    bank_addr=id_add_index(bank_addr, 6)
                    
                else:
                    #bank_addr = add_mem.memory_alloc(1, op_dict[edges_temp[0]]['m']*op_dict[edges_temp[0]]['n'])
                    bank_addr = load_save_mem.memory_alloc(1, op_dict[edges_temp[0]]['m']*op_dict[edges_temp[0]]['n'])
                    #bank_addr1=id_add_index(bank_addr, 1)
                print(bank_addr[0])
                op_dict[edges_temp[0]]['output_id_addr'].append(bank_addr[0])
                op_dict[edges_temp[1]]['input_id_addr'].append(bank_addr[0])

            elif(l_op_type['type']== 'tanh' or l_op_type['type']== 'sigmoid'):
                bank_addr = actv_emul_mem.memory_alloc(1, op_dict[edges_temp[0]]['m']*op_dict[edges_temp[0]]['n'])
                bank_addr=id_add_index(bank_addr, 4)
                print(bank_addr[0])
                op_dict[edges_temp[0]]['output_id_addr'].append(bank_addr[0])
                op_dict[edges_temp[1]]['input_id_addr'].append(bank_addr[0])           

            elif(l_op_type['type']== 'mul'):
                bank_addr = actv_emul_mem.memory_alloc(1, op_dict[edges_temp[0]]['m']*op_dict[edges_temp[0]]['n'])
                bank_addr=id_add_index(bank_addr, 4)
                print(bank_addr[0])
                op_dict[edges_temp[0]]['output_id_addr'].append(bank_addr[0])
                op_dict[edges_temp[1]]['input_id_addr'].append(bank_addr[0])

        elif(l_op_type['type'] in com_op and r_op_type['type'] in data_op):
            print('com -> data')
            #bank_addr = load_mem.memory_alloc(1, op_dict[edges_temp[0]]['m']*op_dict[edges_temp[0]]['n'])
            bank_addr = load_save_mem.memory_alloc(1, op_dict[edges_temp[0]]['m']*op_dict[edges_temp[0]]['n'])
            print(bank_addr[0])
            op_dict[edges_temp[0]]['output_id_addr'].append(bank_addr[0])
            op_dict[edges_temp[1]]['input_id_addr'].append(bank_addr[0])

        else:
            print('none')

    #op_dict=add_complement_id(op_dict)
    return op_dict

def group_vector_memory(Group_mem, vector_mem, node_edge, op_dict, layer_direction, ht_name, temp_input_size):
    
    G=nx.DiGraph()
    G.add_edges_from(node_edge)

    # malloc memory
    #com_op=['matmul', 'mul', 'eltwise','tanh', 'sigmoid']
    for edges_temp in node_edge:
        if(op_dict[edges_temp[0]]['type'] not in data_op):
            continue
        output_edge = list(G.out_edges(edges_temp[0]))

        if(edges_temp[1] != output_edge[0][1]):

            print('skip edge:', edges_temp)
            op_dict[edges_temp[1]]['input_id_addr'].append(op_dict[edges_temp[0]]['output_id_addr'][0])
            continue
        l_op_type = op_dict[edges_temp[0]]
        r_op_type = op_dict[edges_temp[1]]
        print(edges_temp)
        
        if(l_op_type['type'] in data_op and r_op_type['type'] in com_op):
            print('data -> com')
            
            if(l_op_type['type']== 'group'):
                bank_addr = Group_mem.memory_alloc(op_dict[edges_temp[0]]['m']*op_dict[edges_temp[0]]['n'])
                print(bank_addr)
                op_dict[edges_temp[0]]['output_id_addr'].append({'group': bank_addr})
                op_dict[edges_temp[1]]['input_id_addr'].append({'group': bank_addr})
                
            elif(l_op_type['type']== 'vector'):
                bank_addr = vector_mem.memory_alloc(op_dict[edges_temp[0]]['m']*op_dict[edges_temp[0]]['n'])
                #bank_addr = vector_mem.memory_alloc(320)
                print(bank_addr)
                if(edges_temp[0] == ht_name):
                    op_dict[edges_temp[0]]['dir']=1
                else:
                    op_dict[edges_temp[0]]['dir'] = layer_direction
                
                temp_input_size[edges_temp[0]]={'dir':op_dict[edges_temp[0]]['dir'], 'size':op_dict[edges_temp[0]]['m']*op_dict[edges_temp[0]]['n']}
                op_dict[edges_temp[0]]['load_static']=0
                op_dict[edges_temp[0]]['output_id_addr'].append({'vector': bank_addr})
                op_dict[edges_temp[1]]['input_id_addr'].append({'vector': bank_addr})
                
        else:
            print('none')

    #op_dict=add_complement_id(op_dict)
    return op_dict

def bank_new_memory(node_edge, node_info_dict, layer_index, op_dict, bank_loop_op, ins_temp, all_time):
    group0_mem = Mult_bank(2, 4096*32)#0-1
    matmul_mem = Mult_bank(1, 4096*32)#2-3
    group1_mem = Mult_bank(4, 4096*32)#0-3

    Ins_group={}
    for nodes_temp in node_info_dict:
            Ins_group[nodes_temp]={'group':[], 'addr':[], 'type':None, 'size':None} #init dict Ins_group

    G=nx.DiGraph()
    G.add_edges_from(node_edge)
    for nodes_temp in node_info_dict:
        Ins_group[nodes_temp]['type'] = node_info_dict[nodes_temp]['type']
        if(node_info_dict[nodes_temp]['type'] == 'matmul'):
            Ins_group[nodes_temp]['size'] = op_dict[nodes_temp]['m']//2

        else:
            Ins_group[nodes_temp]['size'] = op_dict[nodes_temp]['m']*op_dict[nodes_temp]['n']
        if(node_info_dict[nodes_temp]['type'] == 'matmul'):
            #Ins_group[nodes_temp].append('group0')
            output_edge = list(G.out_edges(nodes_temp))
            print(nodes_temp,'->', output_edge)
            for edge_temp in output_edge:
                in_edge_temp = list(G.in_edges(edge_temp[1]))
                print(in_edge_temp)
                for in_temp in in_edge_temp:
                    print(in_temp[0])
                    Ins_group[in_temp[0]]['group'].append('group0')
        if(node_info_dict[nodes_temp]['type'] in ['group', 'vector']):
            Ins_group[nodes_temp]['group'].append('none')
    
    for nodes_temp in node_info_dict:

        if(Ins_group[nodes_temp]['group'] == []):
            #pass
            Ins_group[nodes_temp]['group'].append('group1')
        
        if(len(Ins_group[nodes_temp]['group']) == 2):
            print('nodes_temp')
            exit()

    
    
    ins_squence_time = copy.deepcopy(ins_temp)
    
    temp_set=[]
    for time_index in range(ins_temp['save'][-1][1]):
        
        dict_temp={}
        for ins_index in(ins_temp):
            for single_ins_time in ins_temp[ins_index]:
                if(single_ins_time[0]<=time_index and single_ins_time[1]>=time_index):
                    dict_temp[single_ins_time[2]]=[]

        temp_set.append(list(dict_temp.keys()))
    new_list=[]
    for i in temp_set:
        if(i not in new_list):
            new_list.append(i)

    print(new_list)
    for same_time_ins in new_list:
        group0=[]
        group1=[]
        for ins_single in same_time_ins:
            if(Ins_group[ins_single]['group'][0] == 'group0'):
                group0.append(ins_single)
            elif(Ins_group[ins_single]['group'][0] == 'group1'):
                group1.append(ins_single)

        print('In group0-> ',group0)
        group0_mem_malloc(G, group0, Ins_group, group0_mem, matmul_mem)
        print('In group1-> ',group1)
        group1_mem_malloc(G, group1, Ins_group, group1_mem)

    

    Ins_group[bank_loop_op[layer_index][1]]['addr'] = Ins_group[bank_loop_op[layer_index][0]]['addr']

    
    for edges_temp in node_edge:# save to ddr
        output_edge = list(G.out_edges(edges_temp[0]))
        l_op_type = op_dict[edges_temp[0]]
        r_op_type = op_dict[edges_temp[1]]
        if(l_op_type['type'] in com_op and r_op_type['type'] in data_op):
            addr = group0_mem.spec_memory_alloc('0', Ins_group[edges_temp[0]]['size'])
            print(edges_temp[0])
            Ins_group[edges_temp[0]]['addr'] = [[0, addr]]


    
    for i in Ins_group:
        if(Ins_group[i]['addr'] == []):
            continue
        if(Ins_group[i]['type'] in com_op or Ins_group[i]['type'] == 'data'):
            print(i)
            op_dict[i]['output_id_addr']+=(Ins_group[i]['addr'])
            output_edge_list = list(G.out_edges(i))
            for j in output_edge_list:
                op_dict[j[1]]['input_id_addr']+=(Ins_group[i]['addr'])

    for i in Ins_group:
        if(Ins_group[i]['type'] == 'sub'):
            op_dict[i]['input_id_addr']=[]
            input_edge_list = list(G.in_edges(i))
            for j in input_edge_list:
                op_dict[i]['input_id_addr']+=Ins_group[j[0]]['addr']
            
        

    
    return op_dict

def group0_mem_malloc(G, group0, Ins_group, group0_mem, matmul_mem):
    if(group0 == []):
        return 
    temp_bank_id=[]
    temp_ins=[]
    for i in group0:
        if(Ins_group[i]['type'] == 'matmul' and Ins_group[i]['addr'] == []):
            print(Ins_group[i]['size'])
            matmul_addr = matmul_mem.memory_alloc(1, Ins_group[i]['size'])
            bank_addr1 = copy.deepcopy(matmul_addr)
            bank_addr2 = copy.deepcopy(matmul_addr)
            addr1 = id_add_index(bank_addr1, 2) 
            addr2 = id_add_index(bank_addr2, 3) 
            Ins_group[i]['addr'] = addr1 + addr2
        if(Ins_group[i]['addr'] != [] and Ins_group[i]['type'] != 'matmul'):
            temp_bank_id.append(Ins_group[i]['addr'][0][0])
        if(Ins_group[i]['addr'] == [] and Ins_group[i]['type'] != 'matmul'):
            temp_ins.append(i)
    if(temp_ins == []):
        return 
    else:
        for i in temp_ins:
            addr = group0_mem.unuse_memory_alloc(temp_bank_id, 1, Ins_group[i]['size'])
            print('group0 addr ->', addr)
            Ins_group[i]['addr']  = addr
    


def group1_mem_malloc(G, group1, Ins_group, group1_mem):
    if(group1 == []):
        return 
    temp_bank_id=[]
    temp_ins=[]
    for i in group1:
        if(Ins_group[i]['addr'] != []):
            temp_bank_id.append(Ins_group[i]['addr'][0][0])
        if(Ins_group[i]['addr'] == []):
            temp_ins.append(i)
    if(temp_ins == []):
        return 
    else:
        for i in temp_ins:
            used_bankid=[]
            used_bankid = group1_check_unuse_mem(G, i, Ins_group)
            print('used_bankid->', temp_bank_id + used_bankid)
            addr = group1_mem.unuse_memory_alloc(temp_bank_id + used_bankid, 1, Ins_group[i]['size'])
            addr = id_add_index(addr, 4)
            print('group1 addr ->', addr)
            Ins_group[i]['addr']  = addr

def group1_check_unuse_mem(G, node, Ins_group):
    used_bankid=[]
    output_edge = list(G.out_edges(node))
    print(node,' check_unuse_mem ->', output_edge)
    for edge_temp in output_edge:
        in_edge_temp = list(G.in_edges(edge_temp[1]))
        print(in_edge_temp)
        for in_temp in in_edge_temp:
            print(in_temp[0])
            if(Ins_group[in_temp[0]]['addr'] == []):
                used_bankid+=[]
            else:
                print(Ins_group[in_temp[0]]['addr'][0][0])
                used_bankid.append(Ins_group[in_temp[0]]['addr'][0][0]-4)
    
    return used_bankid
            
    
if __name__ == "__main__":
    
    mem = Mult_bank(4, 512*32)
    a= mem.memory_alloc(1, 32)
    a= mem.memory_alloc(1, 512)
    a= mem.spec_memory_alloc('0', 32)
    a= mem.spec_memory_alloc('0', 32)
    a= mem.unuse_memory_alloc([1, 2, 3], 1, 32)
    a= mem.unuse_memory_alloc([1, 2, 3], 1, 32)
    a= mem.unuse_memory_alloc([0], 2, 32)
    a= mem.unuse_memory_alloc([], 2, 32)
    a= mem.unuse_memory_alloc([], 2, 32)
    a= mem.memory_alloc(1, 32)
    a= mem.memory_alloc(1, 512)