

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
import copy
import math

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

load_list=[]
for j in range(4):
    for i in range(8) :
        temp= i*4+j
        load_list.append(temp)

data_bit16=16
data_size=data_bit16//8
share_mem_width=512
ddr_mem_width = 256
share_mem_size=share_mem_width//data_bit16
bank_mem_size = ddr_mem_width//data_bit16

kernel_num=3
batch4 = [1, 2, 3, 6, 8, 10, 12, 14]
batch3 = [17, 18, 19, 21, 23, 25, 27, 29]


def add_complement_id(edge): 
    #temp_num =[ i for i in range(8)]
    group0 = [0, 1, 2, 3]
    group1 = [4, 5, 6, 7]
    temp_num =[]
    
    if(edge['type'] == 'eltwise' or edge['type'] == 'sub'):
        #print('id: ', edge['input_id_addr'][0][0], edge['input_id_addr'][1][0])
        if(edge['input_id_addr'][0][0] in group0 and edge['input_id_addr'][1][0] in group0):
            temp_num = group0
        elif(edge['input_id_addr'][0][0] in group1 and edge['input_id_addr'][1][0] in group1):
            temp_num = group1
        else:
            print('name: ',edge['op_name'])
            print('add bank id num not in same bank')
            print('input_id_addr', edge['input_id_addr'][0][0], edge['input_id_addr'][1][0])
            exit()

        for index_id in edge['input_id_addr']:
            temp_num.remove(index_id[0])
        
        for surplus_num  in range(4-edge['add_count']):
            edge['input_id_addr'].append([temp_num[surplus_num], 0])
    return edge

def dict_to_ins(ins_dict, kernel_num):
    instr=''
    vector_index={'index':0}
    for id_num, ins_temp in ins_dict.items():
        ins_temp['dpon']=dpendency_trans(ins_temp['dpon'])
        ins_temp['dpby']=dpendency_trans(ins_temp['dpby'])
        #print(ins_temp)
        if(ins_temp['name']=='load'):
            instr+=load_ins(ins_temp, vector_index, kernel_num)
        elif(ins_temp['name']=='matmul'):
            instr+=mmul_ins(ins_temp)
        elif(ins_temp['name']=='actv'):
            instr+=actv_ins(ins_temp)
        elif(ins_temp['name']=='add'):
            instr+=add_ins(ins_temp)
        elif(ins_temp['name']=='mul'):
            instr+=emul_ins(ins_temp)
        elif(ins_temp['name']=='save'):
            instr+=save_ins(ins_temp, kernel_num)
        elif(ins_temp['name']=='end'):
            instr+=end_ins(ins_temp)
    return instr

def dpendency_trans(dp):
    for i in range(len(dp)):
        if(dp[i] == 'sigmoid' or dp[i]== "tanh"):
            dp[i]='actv'
        elif(dp[i] == 'matmul'):
            dp[i]='mmul'
        elif(dp[i] == 'mul'):
            dp[i]='emul'
        elif(dp[i] == 'eltwise' or dp[i] == 'sub'):
            dp[i]='add'
    temp='+'
    dp = temp.join(dp)
    return dp  


def load_ins(ins_temp, vector_index, kernel_num):
    instr=''
    if(kernel_num == 4):
        batch_bank=batch4
    elif(kernel_num == 3):
        batch_bank=batch3
    elif(kernel_num == 1):
        batch_bank=batch4
    if(ins_temp['type']=='group'):
        #for i in range(group_bank):#group_bank
        for i in load_list:#group_bank
            hbm_bank_addr= 1024*1024*256*(batch_bank[i//4])  
            if(i==0):#load who
                instr+=load_module(ins_temp['dpon'], 'none', 0, (ins_temp['input_id_addr'][0]['ddr_addr']+i*(ins_temp['bank_size']//32))*data_size+hbm_bank_addr, i, ins_temp['output_id_addr'][0]['group']//32//bank_mem_size, ins_temp['bank_size']//32//bank_mem_size)
            elif(i==(group_bank-1)):
                instr+=load_module('none', ins_temp['dpby'], 0, (ins_temp['input_id_addr'][0]['ddr_addr']+i*(ins_temp['bank_size']//32))*data_size+hbm_bank_addr, i, ins_temp['output_id_addr'][0]['group']//32//bank_mem_size, ins_temp['bank_size']//32//bank_mem_size)
            else:
                instr+=load_module('none', 'none', 0, (ins_temp['input_id_addr'][0]['ddr_addr']+i*(ins_temp['bank_size']//32))*data_size+hbm_bank_addr, i, ins_temp['output_id_addr'][0]['group']//32//bank_mem_size, ins_temp['bank_size']//32//bank_mem_size)
    elif(ins_temp['type']=='vector'):
        if(ins_temp['load_static']== 1):
            instr+=load_vector_static_module(ins_temp['dpon'], ins_temp['dpby'], 0, ins_temp['input_id_addr'][0]['ddr_addr']*data_size, 32, ins_temp['output_id_addr'][0]['vector']//bank_mem_size, ins_temp['bank_size']//bank_mem_size, kernel_num)
        else:
            instr+=load_module_dy(ins_temp['dpon'], ins_temp['dpby'], 0, ins_temp['input_id_addr'][0]['ddr_addr']*data_size, 32, ins_temp['output_id_addr'][0]['vector']//bank_mem_size, ins_temp['bank_size']//bank_mem_size, ins_temp['dir'], vector_index['index'], kernel_num)
            vector_index['index']+=1
    else:
        if(kernel_num == 4):
            base_bank = 6
        elif(kernel_num == 3):
            base_bank = 21
        elif(kernel_num == 1):
            base_bank = 6
        addr_base=1024*1024*256*base_bank# load bias bank6
        instr+=load_share_module(ins_temp['dpon'], ins_temp['dpby'], 0, ins_temp['input_id_addr'][0]['ddr_addr']*data_size+addr_base, ins_temp['output_id_addr'][0][0], ins_temp['output_id_addr'][0][1]//bank_mem_size, ins_temp['bank_size']//bank_mem_size, kernel_num)
    return instr

def mmul_input_addr(ins_temp, data_type):
    for i in ins_temp['input_id_addr']:
        if(data_type in i):
            return i[data_type]

def mmul_ins(ins_temp):
    instr=''
    loop_mmul=ins_temp['m']//64
    #bias_addr=share_mem.memory_alloc(1, mmul_input_addr(ins_temp, 'vector'))
    #print(bias_addr)
    for i in range(loop_mmul):
        if(loop_mmul==1):#//1024=//32//32
            instr+=mmul_module(ins_temp['dpon'], ins_temp['dpby'], vector_id, mmul_input_addr(ins_temp, 'vector')//16, ins_temp['n']//16, group_id, mmul_input_addr(ins_temp, 'group')//1024+i*ins_temp['n']//16, 32, 3, 0, ins_temp['output_id_addr'][0][0]+2, ins_temp['output_id_addr'][0][1]//32+ i, ins_temp['fix'], mmul_id, mmul_save)
        elif(loop_mmul==2):
            if(i==0):
                instr+=mmul_module(ins_temp['dpon'], 'none', vector_id, mmul_input_addr(ins_temp, 'vector')//16, ins_temp['n']//16, group_id, mmul_input_addr(ins_temp, 'group')//1024+i*ins_temp['n']//16, 32, 3, 0, ins_temp['output_id_addr'][0][0]+2, ins_temp['output_id_addr'][0][1]//32+ i, ins_temp['fix'], mmul_id, mmul_save)
            elif(i==1):
                instr+=mmul_module('none', ins_temp['dpby'], vector_id, mmul_input_addr(ins_temp, 'vector')//16, ins_temp['n']//16, group_id, mmul_input_addr(ins_temp, 'group')//1024+i*ins_temp['n']//16, 32, 3, 0, ins_temp['output_id_addr'][0][0]+2, ins_temp['output_id_addr'][0][1]//32+ i, ins_temp['fix'], mmul_id, mmul_save)
        elif(loop_mmul>2):
            if(i==0):
                instr+=mmul_module(ins_temp['dpon'], 'none', vector_id, mmul_input_addr(ins_temp, 'vector')//16, ins_temp['n']//16, group_id, mmul_input_addr(ins_temp, 'group')//1024+i*ins_temp['n']//16, 32, 3, 0, ins_temp['output_id_addr'][0][0]+2, ins_temp['output_id_addr'][0][1]//32+ i, ins_temp['fix'], mmul_id, mmul_save)
            elif(i==(loop_mmul-1)):
                instr+=mmul_module('none', ins_temp['dpby'], vector_id, mmul_input_addr(ins_temp, 'vector')//16, ins_temp['n']//16, group_id, mmul_input_addr(ins_temp, 'group')//1024+i*ins_temp['n']//16, 32, 3, 0, ins_temp['output_id_addr'][0][0]+2, ins_temp['output_id_addr'][0][1]//32+ i, ins_temp['fix'], mmul_id, mmul_save)
            else:
                instr+=mmul_module('none', 'none', vector_id, mmul_input_addr(ins_temp, 'vector')//16, ins_temp['n']//16, group_id, mmul_input_addr(ins_temp, 'group')//1024+i*ins_temp['n']//16, 32, 3, 0, ins_temp['output_id_addr'][0][0]+2, ins_temp['output_id_addr'][0][1]//32+ i, ins_temp['fix'], mmul_id, mmul_save)
    
    return instr

def actv_ins(ins_temp):
    if('init' in ins_temp):
        instr=actv_module(ins_temp['dpon'], ins_temp['dpby'], ins_temp['type'], 0, ins_temp['input_id_addr'][0][0]+2, ins_temp['input_id_addr'][0][1]//share_mem_size, ins_temp['m']*ins_temp['n']//share_mem_size, ins_temp['output_id_addr'][0][0]+2, ins_temp['output_id_addr'][0][1]//share_mem_size, 1, 0)
    else:
        instr=actv_module(ins_temp['dpon'], ins_temp['dpby'], ins_temp['type'], 0, ins_temp['input_id_addr'][0][0]+2, ins_temp['input_id_addr'][0][1]//share_mem_size, ins_temp['m']*ins_temp['n']//share_mem_size, ins_temp['output_id_addr'][0][0]+2, ins_temp['output_id_addr'][0][1]//share_mem_size, 0, ins_temp['fix'])
    return instr

def del_bank(ins, bank_id):
    ins_temp=copy.deepcopy(ins)
    print('bank ',ins_temp['input_id_addr'])
    for i in ins_temp['input_id_addr']:
        if(i[0] == bank_id):
            ins_temp['input_id_addr'].remove(i)
    print('del_bank ', ins_temp['input_id_addr'])
    ins_temp['add_count'] = 2
    if(ins_temp['input_id_addr'][0][0]>ins_temp['input_id_addr'][1][0]):
        ins_temp['input_id_addr'].reverse()
    return ins_temp

#add_module(dpon, dpby, add_count, add_id, type_id, add_size, a_id, a_addr, b_id, b_addr, c_id, c_addr, d_id, d_addr, result_id, result_addr,fix)
def add_ins(ins_temp):
    #add_loop = ins_temp['m']*ins_temp['n']//64  #mmul out 32*2
    add_loop = math.ceil(ins_temp['m']*ins_temp['n']/64)
    dpon_by = ['none']*add_loop*4
    dpon_by[0] = ins_temp['dpon']
    dpon_by[-1] = ins_temp['dpby']
    #mmul_add_size = add_loop//2
    mmul_add_size = 1
    instr=''
    if(ins_temp['add_count'] == 2):
        ins_temp=add_complement_id(ins_temp)
        instr=add_module(ins_temp['dpon'], ins_temp['dpby'], ins_temp['add_count'], 0, ins_temp['type'], ins_temp['m']*ins_temp['n']//share_mem_size, ins_temp['input_id_addr'][0][0]+2, ins_temp['input_id_addr'][0][1]//share_mem_size, ins_temp['input_id_addr'][1][0]+2, ins_temp['input_id_addr'][1][1]//share_mem_size, ins_temp['input_id_addr'][2][0]+2, ins_temp['input_id_addr'][2][1]//share_mem_size, ins_temp['input_id_addr'][3][0]+2, ins_temp['input_id_addr'][3][1]//share_mem_size, ins_temp['output_id_addr'][0][0]+2, ins_temp['output_id_addr'][0][1]//share_mem_size, ins_temp['fix'])
    elif(ins_temp['add_count'] == 3):
        for i in range(add_loop):
            ins_temp1=del_bank(ins_temp, 3)# bank_id 2
            ins_temp1=add_complement_id(ins_temp1)
            instr+=add_module(dpon_by[2*(2*i)], dpon_by[2*(2*i)+1], ins_temp1['add_count'], 0, ins_temp1['type'], mmul_add_size, ins_temp1['input_id_addr'][0][0]+2, ins_temp1['input_id_addr'][0][1]//share_mem_size + 2*i*mmul_add_size, ins_temp1['input_id_addr'][1][0]+2, ins_temp['input_id_addr'][1][1]//share_mem_size + i*mmul_add_size, ins_temp1['input_id_addr'][2][0]+2, ins_temp1['input_id_addr'][2][1]//share_mem_size, ins_temp1['input_id_addr'][3][0]+2, ins_temp1['input_id_addr'][3][1]//share_mem_size, ins_temp1['output_id_addr'][0][0]+2, ins_temp1['output_id_addr'][0][1]//share_mem_size + 2*i*mmul_add_size, ins_temp1['fix'])
            ins_temp2=del_bank(ins_temp, 2)# bank_id 3
            ins_temp2=add_complement_id(ins_temp2)
            instr+=add_module(dpon_by[2*(2*i+1)], dpon_by[2*(2*i+1)+1], ins_temp2['add_count'], 0, ins_temp2['type'], mmul_add_size, ins_temp2['input_id_addr'][0][0]+2, ins_temp2['input_id_addr'][0][1]//share_mem_size+ (2*i+1)*mmul_add_size, ins_temp2['input_id_addr'][1][0]+2, ins_temp2['input_id_addr'][1][1]//share_mem_size + i*mmul_add_size, ins_temp2['input_id_addr'][2][0]+2, ins_temp2['input_id_addr'][2][1]//share_mem_size, ins_temp2['input_id_addr'][3][0]+2, ins_temp2['input_id_addr'][3][1]//share_mem_size, ins_temp2['output_id_addr'][0][0]+2, ins_temp2['output_id_addr'][0][1]//share_mem_size + (2*i+1)*mmul_add_size, ins_temp2['fix'])
        
    return instr

def emul_ins(ins_temp):
    instr=emul_module(ins_temp['dpon'], ins_temp['dpby'], 0, ins_temp['input_id_addr'][0][0]+2, ins_temp['input_id_addr'][0][1]//share_mem_size, ins_temp['m']*ins_temp['n']//share_mem_size, ins_temp['input_id_addr'][1][0]+2, ins_temp['input_id_addr'][1][1]//share_mem_size, ins_temp['output_id_addr'][0][0]+2, ins_temp['output_id_addr'][0][1]//share_mem_size, ins_temp['fix'])
    return instr

def save_ins(ins_temp, kernel_num):
    instr=save_module(ins_temp['dpon'], ins_temp['dpby'], 0, ins_temp['output_id_addr'][0]['ddr_addr']*data_size, ins_temp['input_id_addr'][0][0], ins_temp['input_id_addr'][0][1]//bank_mem_size, ins_temp['m']*ins_temp['n']//bank_mem_size, kernel_num)
    return instr

def end_ins(ins_temp):
    instr=end_module(ins_temp['dpon'], ins_temp['dpby'])
    return instr

def init_actv(op_dict, share_mem):
    instr_dict={}
    sgmd_init_addr=share_mem.memory_alloc(1, op_dict['actv_sgmd']['m']*op_dict['actv_sgmd']['n'])
    sgmd_init_addr[0][0]=1
    tanh_init_addr=share_mem.memory_alloc(1, op_dict['actv_tanh']['m']*op_dict['actv_tanh']['n'])
    tanh_init_addr[0][0]=1
    instr_dict[1]={'name':'load', 'dpon': ['none'], 'dpby': ['actv'], 'type': 'data','input_id_addr': op_dict['actv_sgmd']['input_id_addr'], 'output_id_addr': sgmd_init_addr,'bank_size': op_dict['actv_sgmd']['m']*op_dict['actv_sgmd']['n']}
    instr_dict[2]={'name':'actv', 'dpon': ['load'], 'dpby': ['none'], 'type': 'sigmoid','m':op_dict['actv_sgmd']['m'],'n':op_dict['actv_sgmd']['n'],'input_id_addr': sgmd_init_addr, 'output_id_addr': sgmd_init_addr, 'fix': 0, 'init':True}
    instr_dict[3]={'name':'load', 'dpon': ['none'], 'dpby': ['actv'], 'type': 'data','input_id_addr': op_dict['actv_tanh']['input_id_addr'], 'output_id_addr': tanh_init_addr,'bank_size': op_dict['actv_tanh']['m']*op_dict['actv_tanh']['n']}
    instr_dict[4]={'name':'actv', 'dpon': ['load'], 'dpby': ['none'], 'type': 'tanh','m':op_dict['actv_tanh']['m'],'n':op_dict['actv_tanh']['n'],'input_id_addr': tanh_init_addr, 'output_id_addr': tanh_init_addr, 'fix': 0, 'init':True}

    return instr_dict


def load_module(dpon, dpby, ddr_channel, ddr_addr, bank_id, bank_addr, bank_size):
    load_instr='LOAD dpon %s dpby %s hp_id %d bank_id %d bank_addr %d bank_size %d ddr_addr %d addr_type 0\r\n'%(dpon, dpby, ddr_channel, bank_id, bank_addr, bank_size-1, ddr_addr )
    return (load_instr)

def load_vector_static_module(dpon, dpby, ddr_channel, ddr_addr, bank_id, bank_addr, bank_size, kernel_num):
    
    dpon_by = ['none']*2*kernel_num
    dpon_by[0] = dpon
    dpon_by[-1] = dpby
    load_instr=''
    for i in range(kernel_num):
        load_instr+='LOAD dpon %s dpby %s hp_id %d bank_id %d bank_addr %d bank_size %d ddr_addr %d addr_type 0\r\n'%(dpon_by[2*i], dpon_by[2*i+1], ddr_channel, bank_id+i, bank_addr, bank_size-1, ddr_addr )
    return (load_instr)

def load_share_module(dpon, dpby, ddr_channel, ddr_addr, bank_id, bank_addr, bank_size, kernel_num):
    
    dpon_by = ['none']*2*kernel_num
    dpon_by[0] = dpon
    dpon_by[-1] = dpby
    load_instr=''
    for i in range(kernel_num):
        load_instr+='LOAD dpon %s dpby %s hp_id %d bank_id %d bank_addr %d bank_size %d ddr_addr %d addr_type 0\r\n'%(dpon_by[2*i], dpon_by[2*i+1], ddr_channel, bank_id+((4+i)<<4), bank_addr, bank_size-1, ddr_addr )
    return (load_instr)

def load_module_dy(dpon, dpby, ddr_channel, ddr_addr_dy, bank_id_dy, bank_addr_dy, bank_size, load_dir, vector_index=0, kernel_num =4):
    print('load vector: ddr_addr %d bank_id %d bank_addr %d '%(ddr_addr_dy, bank_id_dy, bank_addr_dy))
    print('bank_id_dy ->:', bank_id_dy+32)
    print('vector_index: ',vector_index)
    print(load_dir)
    src_reg_op =load_dir
    
    dpon_by = ['none']*2*kernel_num
    dpon_by[0] = dpon
    dpon_by[-1] = dpby
    load_instr=''
    for i in range(kernel_num):
        dest_kernel_index=i
        dest_reg_index=vector_index
        dest_reg_op=1#add
        #dest_reg_step=bank_size*share_mem_size
        dest_reg_step=0
        bank_id=(dest_kernel_index<<4)+(dest_reg_index<<2)+dest_reg_op
        bank_addr=dest_reg_step

        src_kernel_index=i
        src_reg_index=vector_index
        #src_reg_op=1 #add
        src_reg_step=bank_size*32
        #src_reg_step=bank_size*data_size
        ddr_addr=(src_kernel_index<<20)+(src_reg_index<<18)+(src_reg_op<<16)+src_reg_step

        load_instr+='LOAD dpon %s dpby %s hp_id %d bank_id %d bank_addr %d bank_size %d ddr_addr %d addr_type 1\r\n'%(dpon_by[2*i], dpon_by[2*i+1], ddr_channel, bank_id, bank_addr, bank_size-1, ddr_addr )
    return (load_instr)

def save_module(dpon, dpby, ddr_channel, ddr_addr_dy, bank_id, bank_addr, bank_size, kernel_num =4):
    print('save ddr addr:', ddr_addr_dy)
    dpon_by = ['none']*2*kernel_num
    dpon_by[0] = dpon
    dpon_by[-1] = dpby
    save_instr=''
    for i in range(kernel_num):
        dest_kernel_index=i
        dest_reg_index=0
        dest_reg_op=1#add
        dest_reg_step=bank_size*32
        ddr_addr=(dest_kernel_index<<20)+(dest_reg_index<<18)+(dest_reg_op<<16)+dest_reg_step
        save_instr+='SAVE dpon %s dpby %s hp_id %d ddr_addr %d bank_size %d bank_id %d bank_addr %d addr_type 1\r\n'%(dpon_by[2*i], dpon_by[2*i+1], ddr_channel, ddr_addr, bank_size-1, bank_id+((4+i)<<4), bank_addr)
    return save_instr

def mmul_module(dpon, dpby, vector_id, vector_addr, vector_size, matrix_id, matrix_addr, matrix_row, bias_id, bias_addr, result_id, result_addr, fix, mmul_id, result_save):
    mmul_instr="MMUL dpon %s dpby %s vector_len %d vector_bank_id %d vector_bank_addr %d weight_bank_id %d weight_bank_addr %d weight_rows 0 \
 wgt_size %d result_bank_id %d result_bank_addr %d trunc %d relu 0 addr_type 0\r\n"\
    %(dpon, dpby, vector_size-1, 1, vector_addr, matrix_id, matrix_addr, 2*vector_size-1, result_id,  result_addr, fix)
#%(dpon, dpby, vector_size-1, 1, vector_addr, matrix_id, matrix_addr, matrix_row-1, bias_id, bias_addr, result_id, result_addr, fix, mmul_id, result_save)
    return mmul_instr

def add_module(dpon, dpby, add_count, add_id, type_id, add_size, a_id, a_addr, b_id, b_addr, c_id, c_addr, d_id, d_addr, result_id, result_addr,fix):
    if(type_id =='eltwise'):
        type_id =0
    elif(type_id =='sub'):
        type_id =1
    add_instr='ADD  dpon %s dpby %s addends %d trunc %d add_id %d sign %d len %d a_bank_id %d a_bank_addr %d b_bank_id %d b_bank_addr %d c_bank_id %d c_bank_addr %d out_type 0 relu 0 result_bank_id %d result_bank_addr %d\r\n'\
        %(dpon, dpby, add_count-1, fix, add_id, type_id, add_size-1, a_id, a_addr, b_id, b_addr, c_id, c_addr, result_id, result_addr)
         #dpon, dpby, add_count-1, fix, add_id, add_size-1, a_id
    return add_instr

def actv_module(dpon, dpby, type_id, actv_id, src_id, src_addr, src_size, dest_id, dest_addr, init, fix ):
    if(type_id =='sigmoid'):
        type_id =0
    elif(type_id =='tanh'):
        type_id =1
    actv_instr='ACTV dpon %s dpby %s activ_type %d actv_id %d len %d in_bank_id %d in_bank_addr %d result_bank_id %d result_bank_addr %d trunc %d init %d\r\n'\
        %(dpon, dpby, type_id, actv_id, src_size-1, src_id, src_addr, dest_id, dest_addr, fix, init)
    
    return actv_instr

def emul_module(dpon, dpby, emul_id, a_id, a_addr, size, b_id, b_addr, result_id, result_addr, fix):
    if(a_id == b_id):
        print('emul same: ', a_id, b_id)
        exit()
    emul_instr='EMUL dpon %s dpby %s trunc %d emul_id %d len %d vector1_bank_id %d vector1_bank_addr %d vector2_bank_id %d vector2_bank_addr %d result_bank_id %d result_bank_addr %d\r\n'\
        %(dpon, dpby, fix, emul_id, size-1, a_id, a_addr, b_id, b_addr, result_id, result_addr)
    
    return emul_instr

def end_module(dpon, dpby):
    end_instr='END dpon %s dpby %s\r\n'%(dpon, dpby)
    return end_instr

