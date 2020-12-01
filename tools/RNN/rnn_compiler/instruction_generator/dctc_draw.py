

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
import csv
import time
import copy

ins_list=['load', 'save', 'matmul', 'add', 'mul', 'actv']
bank_id_dict={'group':1, 'vector':2, 0:3, 1:4, 2:5, 3:6, 4:7, 5:8, 6:9, 7:10, 'ddr_addr':11}
t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
head=[
        "HEADER\n",
        "VTF File Version,1.0\n",
        "VTF File Type,1\n",
        "PID,1\n",
        "Generated on,%s\n" % t,
        "Resolution,ms\n",
        "Min Resolution,us\n",
        "Trace Version,1.0\n",
        "XRT  Version,2.6.0\n",
        "Tool Version,2020.1\n",
        "\n",
        "STRUCTURE\n",
        "Group_Start,LSTM\n",
        "Static_Row, 1, load\n",
        "Static_Row, 2, save\n",
        "Static_Row, 3, matmul\n",
        "Static_Row, 4, add\n",
        "Static_Row, 5, mul\n",
        "Static_Row, 6, actv\n",
        "\n",
        "Group_End,LSTM\n",
        "\n",
        "MAPPING\n",
        "1, Group\n",
        "2, vector\n",
        "3, Bank0\n",
        "4, Bank1\n",
        "5, Bank2\n",
        "6, Bank3\n",
        "7, Bank4\n",
        "8, Bank5\n",
        "9, Bank6\n",
        "10, Bank7\n",
        "11, ddr\n",
        "\n",
        "EVENTS\n",
        "\n",
]


def print_dict(op_dict):
    for i, j in op_dict.items():
        print(("%s : %s") % (i, j))

def load_time(dict_time):
    if(dict_time['type'] == 'droup'):
        time = 60 + dict_time['m']*dict_time['n']*16 //256//8
    else:
        time = 60 + dict_time['m']*dict_time['n']*16 //256
    return time

def save_time(dict_time):
    time = 60 + dict_time['m']*dict_time['n']*16 //256
    return time

def matmul_time(dict_time):
    time = 32 + dict_time['m']*dict_time['n'] //1024 +32
    return time

def mul_time(dict_time):
    time = 18 + dict_time['m']*dict_time['n']*16//512
    return time

def add_time(dict_time):
    time = 18 + dict_time['m']*dict_time['n']*16//512
    return time

def actv_time(dict_time):
    time = 18 + dict_time['m']*dict_time['n']*16//512
    return time
 
def judge_none(dp):
    for i in dp:
        if(i in ins_list):
            return False
        else:
            return True

def del_dp(ins_dict, current_ins, dpby):
    for ins in dpby:
        for index in range(len(ins_dict[ins])):
            dpon = ins_dict[ins][index][0]
            if(current_ins in dpon):
                ins_dict[ins][index][0].remove(current_ins)
                ins_dict[ins][index][0].append('none')
                break
    return  ins_dict

def if_dict_none(ins_dict):
    
    for i, j in ins_dict.items():
        if(j == []):
            pass
        else:
            return False
    return True


def ins_exe(ins_dict, single_ins_time):
    ins_time={'load':[[0,0]], 'matmul':[[0,0]], 'mul':[[0,0]], 'add':[[0,0]], 'actv':[[0,0]], 'save':[[0,0]]}
    ins_exe_count={'load':0, 'save':0, 'matmul': 0, 'add':0, 'mul':0, 'actv':0}
    end_ins=0
    all_time=0
    while(end_ins != -1):
        end_ins+=1
        for i, j in ins_dict.items():
            if(j == []):
                if(if_dict_none(ins_dict)):
                    all_time=(ins_time[i][-1][1])
                    print('all time->', all_time)
                    print(end_ins)
                    end_ins=-1

                continue
            dpon=ins_dict[i][0][0]
            dpby=ins_dict[i][0][1]
            
            print('loop ins ',end_ins, i, dpon, dpby)
            
            if(judge_none(dpon)):
                print('exe ->', i)
                #print(ins_time[i])
                ins_exe_count[i]+=1
                if(len(ins_time[i][-1]) == 1):
                    ins_time[i][-1].append(ins_time[i][-1][0]+single_ins_time[i][ins_exe_count[i]-1])
                else:
                     ins_time[i].append([ins_time[i][-1][1]+1, ins_time[i][-1][1]+1+single_ins_time[i][ins_exe_count[i]-1]])

                if(dpby[0] == 'end'):
                    print('end')
                    all_time=(ins_time[i][-1][1])
                    print(all_time)
                    print(end_ins)
                    end_ins=-1
                elif(judge_none(dpby)):
                    #print(i,' del before',ins_dict[i])
                    del(ins_dict[i][0])
                    #print(i, ' del after ',ins_dict[i])
                else:
                    for add_time in dpby:
                        if(len(ins_time[add_time][-1]) == 1):
                            if(ins_time[add_time][-1][0] < ins_time[i][-1][1])+1:
                                ins_time[add_time][-1][0]=ins_time[i][-1][1]+1
                        else:
                            if(ins_time[add_time][-1][1]< ins_time[i][-1][1]+1):
                                ins_time[add_time].append([ins_time[i][-1][1]+1])
                            else:
                                ins_time[add_time].append([ins_time[add_time][-1][1]+1])

                    del(ins_dict[i][0])
                    ins_dict = del_dp(ins_dict, i, dpby)

    for ins in ins_time:
        del ins_time[ins][0]

    return ins_time, all_time

def ins_fetch(temp_dict):
    ins_dict = copy.deepcopy(temp_dict)
    load_fetch=[]
    save_fetch=[]
    matmul_fetch=[]
    add_fetch=[]
    mul_fetch=[]
    actv_fetch=[]

    single_ins_time={'load':[], 'save':[], 'matmul': [], 'add':[], 'mul':[], 'actv':[]}
    ins_squence={'load':[], 'save':[], 'matmul': [], 'add':[], 'mul':[], 'actv':[]}

    for ins_i in range(len(ins_dict)):
        for i in range(len(ins_dict[ins_i])):
            temp_dict = ins_dict[ins_i][i+1]
            if(temp_dict['name']== 'load'):
                load_fetch.append([temp_dict['dpon'], temp_dict['dpby']])
                single_ins_time['load'].append(load_time(temp_dict))
                ins_squence['load'].append(temp_dict['op_name'])

            elif(temp_dict['name']== 'save'):
                save_fetch.append([temp_dict['dpon'], temp_dict['dpby']])
                single_ins_time['save'].append(save_time(temp_dict))
                ins_squence['save'].append(temp_dict['op_name'])

            elif(temp_dict['name']== 'matmul'):
                matmul_fetch.append([temp_dict['dpon'], temp_dict['dpby']])
                single_ins_time['matmul'].append(matmul_time(temp_dict))
                ins_squence['matmul'].append(temp_dict['op_name'])

            elif(temp_dict['name']== 'add'):
                add_fetch.append([temp_dict['dpon'], temp_dict['dpby']])
                single_ins_time['add'].append(add_time(temp_dict))
                ins_squence['add'].append(temp_dict['op_name'])

            elif(temp_dict['name']== 'mul'):
                mul_fetch.append([temp_dict['dpon'], temp_dict['dpby']])
                single_ins_time['mul'].append(mul_time(temp_dict))
                ins_squence['mul'].append(temp_dict['op_name'])

            elif(temp_dict['name']== 'actv'):
                actv_fetch.append([temp_dict['dpon'], temp_dict['dpby']])
                single_ins_time['actv'].append(actv_time(temp_dict))
                ins_squence['actv'].append(temp_dict['op_name'])

    exe_ins={'load':load_fetch, 'save':save_fetch, 'matmul': matmul_fetch, 'add':add_fetch, 'mul':mul_fetch, 'actv':actv_fetch}

    time_ins, all_time = ins_exe(exe_ins, single_ins_time)
    

    # print_dict(time_ins)
    # print_dict(ins_squence)
    for single_ins in time_ins:
        for i in range(len(time_ins[single_ins])):
            time_ins[single_ins][i].append(ins_squence[single_ins][i])
    #write_csv(time_ins)
    print_dict(time_ins)
    
    return time_ins ,all_time
    
     
def write_csv(time_ins):
    def takefirst(elem):
        return elem[0] 

    ins_series=[]
    for ins in time_ins:
        for i in range(len(time_ins[ins])):
            ins_series.append([time_ins[ins][i][0], ins_list.index(ins)+1])
            ins_series.append([time_ins[ins][i][1], ins_list.index(ins)+1])
    
    ins_series.sort(key=takefirst)
    count_ins=[-1,-1,-1,-1,-1,-1]
    end_index=[0, 0, 0, 0, 0, 0]
    for i in range(len(ins_series)):
        ins_series[i].insert(0, i+1)

    for i in range(len(ins_series)):
        ins_series[i].append('KERNEL')
        count_ins[ins_series[i][2]-1]+=1
        if(count_ins[ins_series[i][2]-1] %2 ==0):
            end_index[ins_series[i][2]-1]= i+1
            ins_series[i].insert(1, 0)
        else:
            ins_series[i].insert(1, end_index[ins_series[i][2]-1])

    print(ins_series)
    f = open('time.csv','w',encoding='utf-8')
    #csv_writer = csv.writer(f)
    f.writelines(head)
    for i in range(len(ins_series)):
        #csv_writer.writerow(ins_series[i])
        f.writelines("%d,%d,%d,%d,%s\n"%(ins_series[i][0], ins_series[i][1],ins_series[i][2],ins_series[i][3],ins_series[i][4]))
    f.close()




