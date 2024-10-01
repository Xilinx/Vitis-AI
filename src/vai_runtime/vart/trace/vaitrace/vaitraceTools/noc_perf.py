#!/usr/bin/python3
# -*- coding:utf-8 -*-

# Copyright 2022-2023 Advanced Micro Devices Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
import os
import logging
import time
import math
import copy
import threading
from ctypes import *
import copy
import logging
import json

idcode_ddrmc_num_dict = {'0x4c9b093':3, '0x4c93093':2, '0x4c98093':3, '0x4ca9093':4, '0x4ca8093':4, '0x4cd2093':3, '0x4cd0093':3,
                 '0x4c9a093':3, '0x4cc1093':1, '0x4cc0093':1, '0x4cc9093':1, '0x4cc8093':1, '0x4cd2093':3, '0x4cd3093':3}

def merge(a: dict, b: dict):
    if hasattr(a, "keys") and hasattr(b, "keys"):
        for kb in b.keys():
            if kb in a.keys():
                merge(a[kb], b[kb])
            else:
                a.update(b)

class NOCRecord(Structure):
    _fields_ = [('time', c_double), ('data', c_ulonglong * 10)]

    def output(self):
        return "NOC %.7f %d %d %d %d %d %d %d %d %d %d\n" % (self.time, *self.data)


class NOC:
    def __init__(self, mem_type="noc", path='/usr/lib/libmemperf.so', enable=True):
        self.nocLib = cdll.LoadLibrary(path)
        self.data = []
        self.base_addr = []
        self.ddrmc_name = []
        self.interval = 0
        self.act_period = 0
        self.record_data_len = 0
        self.type = mem_type
        self.enabled = enable
        self.idcode = self.getChipId()
        self.get_noc_base_addr(self.idcode)
        self.noc_base_addr = (c_int * len(self.base_addr))(*self.base_addr)
        self.nocLib.create_noc_instance(
            self.type.encode(), self.noc_base_addr, int(len(self.base_addr)))

    def start(self, interval=0.01):
        if self.enabled == False:
            return
        self.interval = interval
        self.nocLib.start.restype = c_int
        return self.nocLib.start(c_double(interval))

    def stop(self):
        if self.enabled == False:
            return

        self.nocLib.stop()
        data = NOCRecord()
        pd = pointer(data)

        while (self.nocLib.pop_data(pd) == 0):
            self.data.append(copy.deepcopy(data))

        self.nocLib.get_act_period.restype = c_double
        self.act_period = self.nocLib.get_act_period()

        self.nocLib.get_record_data_len.restype = c_int
        self.record_data_len = self.nocLib.get_record_data_len()

    def transTimebase(self):
        if self.enabled == False:
            return
        for i in range(0, len(self.data)):
            for j in range(0, self.record_data_len):
                self.data[i].data[j] = int(
                    self.data[i].data[j] / self.interval)

    def printData(self, data):
        if self.enabled == False:
            return
        info = str()
        info += "TimeStamp: %.7f\n" % data.time

        read = [data.data[i]/self.interval/1000 /
                1000 for i in range(0, 10) if i % 2 == 0]
        info += "Read Ports:  "
        for d in read:
            info += "%8.1f" % d

        info += " MB/s\n"

        write = [data.data[i]/self.interval/1000 /
                 1000 for i in range(0, 10) if i % 2 == 1]
        info += "Write Ports: "
        for d in write:
            info += "%8.1f" % d

        info += " MB/s\n  "

        return info
    
    
    def get_noc_base_addr(self, devid):
        ddrmc_number = idcode_ddrmc_num_dict.get(devid, 0)
        if ddrmc_number == 0:
            logging.info("Can not get ddrmc/noc number!")
            return
        pre_base_addr = []
        for (dirpath, dirnames, filenames) in os.walk("/proc/device-tree/axi"):
            for file in dirnames:
                if file.startswith("memory-controller"):
                    compatible = open(os.path.join(os.path.abspath(os.path.join(
                        dirpath, file)), "compatible"), "rt").read().strip().split(',')
                    driver_name = compatible[1].split('-')
                    if driver_name[1] != 'ddrmc':
                        continue
                    reg = os.path.join(os.path.abspath(
                        os.path.join(dirpath, file)), "reg")
                    f = open(reg, "rb")
                    numl = list(f.read())
                    addr_off = ((numl[20] << 24) + (numl[21] <<
                                16) + (numl[22] << 8) + (numl[23] << 0))
                    pre_base_addr.append(int(addr_off))
        pre_base_addr.sort()
        self.base_addr =  pre_base_addr[0:ddrmc_number]
    
    def getChipId(self):
        try:
            assert (os.access("/sys/kernel/debug/zynqmp-firmware/pm", os.W_OK) == True)
            pm = open("/sys/kernel/debug/zynqmp-firmware/pm", "wt")
            assert (pm.write("PM_GET_CHIPID") == len("PM_GET_CHIPID"))
            pm.close()
            pm = open("/sys/kernel/debug/zynqmp-firmware/pm", "rt")
            chip_id_raw = pm.read().strip()  # 'Idcode: 0x4724093, Version:0x20000513'
            pm.close()
            idcode = (chip_id_raw.split(',', 1)[0]).split(':', 1)[1].strip()
            return idcode
        except:
            logging.error(
                    "Cannot get chip id, kernel config CONFIG_ZYNQMP_FIRMWARE_DEBUG=y is required.")
            exit(1)
    
    def getData(self):
        return [d.output() for d in self.data]

    def toMB_s(self,r):
        # 100 M 
        return float(r) / 1000 /10

    def convert_to_json_format(self,raw_data,time_offset=0.0): 
        len2 = len(raw_data[1].split()[1:])
        list_data = []
        cur_list_data = []
        cunt = 0
        for cunt in range(0,len2):
            for rr in raw_data:
                if not rr.startswith("NOC"):
                    continue
                r = rr.split()[1:]
                if cunt == 0 :
                    timestamp = round((float(r[0]) - time_offset) * 1000, )
                    if timestamp < 0:
                        continue
                    cur_list_data.append(timestamp)
                else:
                    cur_list_data.append(round(self.toMB_s(r[cunt]), 2))
            list_data.append(cur_list_data)
            cur_list_data = []
        return list_data

    def create_json_file(self,json_data):
       
        ddrmc_name = []
        ddrmc_data = {}
        data = {
                "ddrmc_nsu_timestamp": json_data[0]
        }
        for k in range (len(self.base_addr)):
            ddrmc_name.append("ddrmc" + "@" + str(hex(self.base_addr[k])) + "_read")
            ddrmc_name.append("ddrmc" + "@" + str(hex(self.base_addr[k])) + "_write")

        for i in range (len(self.base_addr) * 2):# every addr map read and write
            ddrmc_data[ddrmc_name[i]] =  json_data[i+1]
        merge(data, ddrmc_data) 
        json_data = json.dumps(data, indent=4, separators=(",", ":"))
        with open('ddrmc_noc_nsu_data.json','w') as json_file:
            json_file.write(json_data)
        

if __name__ == '__main__':

    noc = NOC()
    noc.start(0.001)
    time.sleep(1)
    noc.stop()
    noc.transTimebase()
    raw_data = noc.getData()
    data_cur = noc.convert_to_json_format(raw_data)
    noc.create_json_file(data_cur)

