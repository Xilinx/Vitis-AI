

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

class Bank:
    def __init__(self, size):
        self.data=size*'0'
        self.size=size

    def memory_alloc(self, size):
        mem_str=self.data
        mem_zero=size*'0'
        mem_flag=size*'1'
        addr=mem_str.find(mem_zero)
        if(addr == -1):
            print('not have enougth memory')
            exit()
        list_mem=list(mem_str)
        list_mem[addr : addr+size]=mem_flag
        self.data=''.join(list_mem)

        print(('size: %d addr: %d')%(size, addr))
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

class new_Bank:
    def __init__(self, size, start=0):
        self.data=[start, start+size-1]
        self.unuse_addr=[[start, start+size-1]]
    def correct_mem(self):
        for mem_addr in self.unuse_addr:
            pass

    def memory_alloc(self, size):
        addr=''
        for segment_addr in self.unuse_addr:
            if(segment_addr[1]-segment_addr[0] < (size-1)):
                break
            else:
                addr = segment_addr[0]
                print('Alloc size: ', size,'Alloc addr start: ',addr)
                segment_addr[0] = segment_addr[0]+size
        if(addr == ''):
            print('Bank Do not have enougth memory')
            exit()
        print(self.unuse_addr)
        return addr
    def memory_free(self, addr, size):
        pass

    def memory_zero(self):
        self.unuse_addr=[self.data]

if __name__ == "__main__":
    mem = Bank(512*32)
    a= mem.memory_alloc(32)
    a= mem.memory_alloc(512)