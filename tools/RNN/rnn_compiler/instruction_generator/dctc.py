

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

import networkx as nx
import json
import copy

data_op=['group', 'vector', 'data']
com_op=['matmul', 'mul', 'eltwise','sub', 'tanh', 'sigmoid', 'add', 'actv']

def json_parser(path):
    node_info_dict={}
    with open(path+'nodes_edge.json', 'r') as f:
        node_edge = json.load(f)
    with open(path+'nodes_info.json', 'r') as f:
        node_info = json.load(f)
    with open(path+'cycle_ops.json', 'r') as f:
        cycle_ops = json.load(f)


    for i in range(len(node_info)):
        temp =  node_info[i]
        node_info_dict[node_info[i]['name']]=temp

    return node_edge, node_info_dict

def cal_num(ins_dict):
    num_id=list(ins_dict.items())[-1][0] + 1
    return num_id

def print_dict(op_dict):
    print('\n')
    for i, j in op_dict.items():
        print(("%s : %s") % (i, j))

def update_ins_dict(ins_dict, name, dpon=None, dpby=None, edges_temp=None):
    num_id=cal_num(ins_dict)
    ins_dict.update({num_id:{'name':name,'dpon':dpon, 'dpby':dpby, 'edge_name':edges_temp}})
    #print(num_id, ins_dict[num_id])
    #print('\n')
            
def dpon_ins(node_info_dict, edge):
    ins_temp=[]
    if(edge == []):
        ins_temp.append('none') 
    else:
        for i in edge:
            if(node_info_dict[i[0]]['type'] in data_op):
                ins_temp.append('load')
            else:
                ins_temp.append(node_info_dict[i[0]]['type'])
    return ins_temp

def dpby_ins(node_info_dict, edge):
    ins_temp=[]
    if(edge == []):
        ins_temp.append('none') 
    else:
        for i in edge:
            if(node_info_dict[i[1]]['type'] in data_op):
                ins_temp.append('save')
            else:
                ins_temp.append(node_info_dict[i[1]]['type'])
    return ins_temp

def find_duplicate(dpon_or_dpby, node_dict, dp_list):
    temp_dict={}
    new_dict={}
    same_op_list=[]
    node_name=''
    if(dpon_or_dpby == 'dpon'):
        node_name=dp_list[0][1]
        for i in dp_list:
            temp_dict[i[0]]=node_dict[i[0]]['type']
        for k, v in temp_dict.items():
            new_dict.setdefault(v, []).append(k)
        for k, v in new_dict.items():
            if(len(v)>1):
                same_op_list.append(v)
        
    elif(dpon_or_dpby == 'dpby'):
        node_name=dp_list[0][0]
        for i in dp_list:
            temp_dict[i[1]]=node_dict[i[1]]['type']
        for k, v in temp_dict.items():
            new_dict.setdefault(v, []).append(k)
        for k, v in new_dict.items():
            if(len(v)>1):
                same_op_list.append(v)
    del_list=[]
    for dup_list in same_op_list:
        temp_dict={}
        for i in dup_list:
            temp_dict[i]=abs(node_dict[node_name]['num']-node_dict[i]['num'])
        sorted_list= list({k: v for k, v in sorted(temp_dict.items(), key=lambda item: item[1])}.keys())
        sorted_list.remove(sorted_list[0])
        del_list=del_list+sorted_list
    
    del_edge=[]
    for i in del_list:
        if(dpon_or_dpby == 'dpon'):
            tup=(i, node_name)
        elif(dpon_or_dpby == 'dpby'):
            tup=(node_name, i)
        del_edge.append(tup)

    return del_edge

def del_edge(edge, dup_edge):
    temp_list = edge
    for i in dup_edge:
        if(i in edge):
            temp_list.remove(i)
    return temp_list
            

def remove_duplicate(node_dict, op_dp):
    for i ,j in op_dp.items():
        del_dpon_list=[]
        del_dpby_list=[]
        if(j['dpon'] != []):
            del_dpon_list = find_duplicate('dpon', node_dict, j['dpon'])#find the same type op

        if(j['dpby'] != []):
            del_dpby_list = find_duplicate('dpby', node_dict, j['dpby'])

        print(i, j)
        print('    ', 'del -> dpon: ',del_dpon_list, 'dpby:',del_dpby_list)
        for k, v in op_dp.items():
            if(node_dict[i]['num']>node_dict[k]['num']):
                op_dp[k]['dpby'] = del_edge(op_dp[k]['dpby'], del_dpon_list)
            if(node_dict[i]['num']<node_dict[k]['num']):
                op_dp[k]['dpon'] = del_edge(op_dp[k]['dpon'], del_dpby_list)
            if(node_dict[i]['num'] == node_dict[k]['num']):
                op_dp[k]['dpon'] = del_edge(op_dp[k]['dpon'], del_dpon_list)
                op_dp[k]['dpby'] = del_edge(op_dp[k]['dpby'], del_dpby_list)
    
    return op_dp

def process_edges_init(node_edge_input, node_info_dict, del_vector=''):#后置操作，即是否进行操作取决于edges_temp[1]的op类型
    node_info=copy.deepcopy(node_info_dict)
    node_edge = copy.deepcopy(node_edge_input)
    node_edge_load_data = copy.deepcopy(node_edge_input)
    ins_dict={0:None}

    G0=nx.DiGraph()
    G0.add_edges_from(node_edge_load_data)
    for edges_temp in node_edge_load_data:
        l_op_type = node_info[edges_temp[0]]
        r_op_type = node_info[edges_temp[1]]
        l_op_out_edge = list(G0.out_edges(l_op_type['name']))
        r_op_in_edge = list(G0.in_edges(r_op_type['name']))

        if(l_op_type['type'] in data_op and r_op_type['type'] in com_op):
            if(r_op_type['name'] == l_op_out_edge[0][1]):
                if(l_op_type['type'] == 'data' or l_op_type['type'] == 'group'):
                    update_ins_dict(ins_dict, 'load', ['none'], ['none'], edges_temp)

    temp_edge=[]
    for i in range(len(node_edge)):
        if(node_info[node_edge[i][0]]['name'] == del_vector or node_info[node_edge[i][0]]['type'] == 'data' or node_info[node_edge[i][0]]['type'] == 'group'):
            print('edge:', node_edge[i])
            node_edge[i][0] = ''
            print('del:', node_edge[i])
        else:
            temp_edge.append(node_edge[i])
    node_edge = temp_edge

    G=nx.DiGraph()
    G.add_edges_from(node_edge)
    
    op_dp={}
    for index_op, data in node_info.items():
        dpby = list(G.out_edges(index_op))
        dpon = list(G.in_edges(index_op))
        op_dp.update({index_op:{'dpon':dpon, 'dpby':dpby}})
        if(data['type'] == 'sub'):
            node_info[index_op]['type']='add'

        elif(data['type'] == 'eltwise'):
            node_info[index_op]['type']='add'

        elif(data['type'] == 'sigmoid'):
            node_info[index_op]['type']='actv'

        elif(data['type'] == 'tanh'):
            node_info[index_op]['type']='actv'
    print_dict(op_dp)

    index_node=0
    node_dict={}
    for i, j in node_info.items():
        index_node+=1
        if(j['type'] in data_op):
            node_dict[i]={'num':index_node, 'type': 'data'}
        else:
            node_dict[i]={'num':index_node, 'type': j['type']}
    #print_dict(node_dict)
    dp_dict=remove_duplicate(node_dict, op_dp)
    
    for edges_temp in node_edge:
        l_op_type = node_info[edges_temp[0]]
        r_op_type = node_info[edges_temp[1]]
        l_op_out_edge = list(G.out_edges(l_op_type['name']))
        r_op_out_edge = list(G.out_edges(r_op_type['name']))
        l_op_in_edge = list(G.in_edges(l_op_type['name']))
        r_op_in_edge = list(G.in_edges(r_op_type['name']))

        print(edges_temp)
        if(l_op_type['type'] in data_op and r_op_type['type'] in com_op):

            if(r_op_type['name'] == l_op_out_edge[0][1]):
                print('load')
                if(l_op_type['name'] == r_op_in_edge[-1][0]):
                    update_ins_dict(ins_dict, 'load', dpon_ins(node_info, dp_dict[edges_temp[0]]['dpon']), dpby_ins(node_info, dp_dict[edges_temp[0]]['dpby']), edges_temp)
                    print('com')
                    update_ins_dict(ins_dict, r_op_type['type'], dpon_ins(node_info, dp_dict[edges_temp[1]]['dpon']), dpby_ins(node_info, dp_dict[edges_temp[1]]['dpby']), edges_temp)
                else:
                    update_ins_dict(ins_dict, 'load', dpon_ins(node_info, dp_dict[edges_temp[0]]['dpon']), dpby_ins(node_info, dp_dict[edges_temp[0]]['dpby']), edges_temp)
            elif(l_op_type['name'] == r_op_in_edge[-1][0]):
                print('com')
                update_ins_dict(ins_dict, r_op_type['type'], dpon_ins(node_info, dp_dict[edges_temp[1]]['dpon']), dpby_ins(node_info, dp_dict[edges_temp[1]]['dpby']), edges_temp)

        elif(l_op_type['type'] in com_op and r_op_type['type'] in com_op):
            if(l_op_in_edge == []):
                print('first com')
                update_ins_dict(ins_dict, l_op_type['type'], dpon_ins(node_info, dp_dict[edges_temp[0]]['dpon']), dpby_ins(node_info, dp_dict[edges_temp[0]]['dpby']), ['', edges_temp[0]])
            
            if(l_op_type['name'] == r_op_in_edge[-1][0]):
                print('com->com')
                update_ins_dict(ins_dict, r_op_type['type'], dpon_ins(node_info, dp_dict[edges_temp[1]]['dpon']), dpby_ins(node_info, dp_dict[edges_temp[1]]['dpby']), edges_temp)
            else:
                pass
                #update_ins_dict(ins_dict, 'output address = input', [], [], edges_temp)###

        elif(l_op_type['type'] in com_op and r_op_type['type'] in data_op):
            print('com -> data')
            update_ins_dict(ins_dict, 'save', dpon_ins(node_info, dp_dict[edges_temp[1]]['dpon']), dpby_ins(node_info, dp_dict[edges_temp[1]]['dpby']), edges_temp)
        
        
        else:
            print('error unknow op type, exit......')
            print(l_op_type['type'] )
            exit
            
    del ins_dict[0]
    return ins_dict


def process_edges_run(node_edge, node_info_dict):#后置操作，即是否进行操作取决于edges_temp[1]的op类型
    node_info=copy.deepcopy(node_info_dict)

    temp_edge=[]
    for i in range(len(node_edge)):
        if(node_info[node_edge[i][0]]['type'] == 'group' or node_info[node_edge[i][0]]['type'] == 'data'):
            node_edge[i][0] = ''
        else:
            temp_edge.append(node_edge[i])
    node_edge=temp_edge

    for i in node_edge:
        print(i)

    G=nx.DiGraph()
    G.add_edges_from(node_edge)
    ins_dict={0:None}

    op_dp={}
    for index_op, data in node_info.items():
        dpby = list(G.out_edges(index_op))
        dpon = list(G.in_edges(index_op))
        op_dp.update({index_op:{'dpon':dpon, 'dpby':dpby}})
        if(data['type'] == 'sub'):
            node_info[index_op]['type']='add'

        elif(data['type'] == 'eltwise'):
            node_info[index_op]['type']='add'

        elif(data['type'] == 'sigmoid'):
            node_info[index_op]['type']='actv'

        elif(data['type'] == 'tanh'):
            node_info[index_op]['type']='actv'
    print_dict(op_dp)

    index_node=0
    node_dict={}
    for i, j in node_info.items():
        index_node+=1
        if(j['type'] in data_op):
            node_dict[i]={'num':index_node, 'type': 'data'}
        else:
            node_dict[i]={'num':index_node, 'type': j['type']}
    #print_dict(node_dict)
    dp_dict=remove_duplicate(node_dict, op_dp)
    
    for edges_temp in node_edge:
        l_op_type = node_info[edges_temp[0]]
        r_op_type = node_info[edges_temp[1]]
        l_op_out_edge = list(G.out_edges(l_op_type['name']))
        r_op_out_edge = list(G.out_edges(r_op_type['name']))
        l_op_in_edge = list(G.in_edges(l_op_type['name']))
        r_op_in_edge = list(G.in_edges(r_op_type['name']))

        print(edges_temp)
        if(l_op_type['type'] in data_op and r_op_type['type'] in com_op):
            #print('data -> com')
            #print('\033[1;35m output \033[0m!: ',l_op_out_edge)
            #print('\033[1;35m input  \033[0m!: ',r_op_in_edge)
            #r_op_type['input'][l_op_type['type']]=l_op_type['output']

            if(r_op_type['name'] == l_op_out_edge[0][1]):
                print('load')
                if(l_op_type['name'] == r_op_in_edge[-1][0]):
                    update_ins_dict(ins_dict, 'load', dpon_ins(node_info, dp_dict[edges_temp[0]]['dpon']), dpby_ins(node_info, dp_dict[edges_temp[0]]['dpby']), edges_temp)
                    print('com')
                    update_ins_dict(ins_dict, r_op_type['type'], dpon_ins(node_info, dp_dict[edges_temp[1]]['dpon']), dpby_ins(node_info, dp_dict[edges_temp[1]]['dpby']), edges_temp)
                else:
                    update_ins_dict(ins_dict, 'load', dpon_ins(node_info, dp_dict[edges_temp[0]]['dpon']), dpby_ins(node_info, dp_dict[edges_temp[0]]['dpby']), edges_temp)
            elif(l_op_type['name'] == r_op_in_edge[-1][0]):
                print('com')
                update_ins_dict(ins_dict, r_op_type['type'], dpon_ins(node_info, dp_dict[edges_temp[1]]['dpon']), dpby_ins(node_info, dp_dict[edges_temp[1]]['dpby']), edges_temp)

        elif(l_op_type['type'] in com_op and r_op_type['type'] in com_op):
            if(l_op_type['name'] == r_op_in_edge[-1][0]):
                print('com->com')
                update_ins_dict(ins_dict, r_op_type['type'], dpon_ins(node_info, dp_dict[edges_temp[1]]['dpon']), dpby_ins(node_info, dp_dict[edges_temp[1]]['dpby']), edges_temp)
            else:
                pass
                #update_ins_dict(ins_dict, 'output address = input', [], [], edges_temp)###

        elif(l_op_type['type'] in com_op and r_op_type['type'] in data_op):
            print('com -> data')
            update_ins_dict(ins_dict, 'save', dpon_ins(node_info, dp_dict[edges_temp[1]]['dpon']), dpby_ins(node_info, dp_dict[edges_temp[1]]['dpby']), edges_temp)
        else:
            print('error unknow op type, exit......')
            print(l_op_type['type'] )
            exit
            
    del ins_dict[0]
    return ins_dict

if __name__ == "__main__":
    PATH='example/lstm_25_tf/rnn_cell_0/'
    node_edge, node_info_dict = json_parser(PATH)
    ins_dict = process_edges_init(node_edge, node_info_dict, 'rnn_cell_0_input_1')#rnn_cell_0_input_1#EncoderModuleLayer0LayerLstmCellModule__input_1
    print_dict(ins_dict)
