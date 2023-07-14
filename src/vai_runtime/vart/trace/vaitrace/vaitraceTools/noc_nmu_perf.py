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
import argparse
import json

def merge(a: dict, b: dict):
    if hasattr(a, "keys") and hasattr(b, "keys"):
        for kb in b.keys():
            if kb in a.keys():
                merge(a[kb], b[kb])
            else:
                a.update(b)
def record_data_factory(size=10):
    class APMRecord(Structure):
        _fields_ = [('time', c_double), ('data', c_longlong * size)]

        def output(self):
            data_cur = []
            list_data = []
            data_cur = [*self.data]
            list_data.append("NOC")
            list_data.append(self.time)
            for i in range(0, size):
                list_data.append(data_cur[i])
            return list_data
    data =  APMRecord()
    return data

class NOC:
    def __init__(self, mem_type="noc_nmu", path='/usr/lib/libnmuperf.so', enable=True):
        self.nmuLib = cdll.LoadLibrary(path)
        self.data = []
        self.base_addr = []
        self.location_name = []
        self.interval = 0
        self.npi_freq = 0
        self.act_period = 0
        self.record_data_len = 0
        self.started = False
        self.type = mem_type
        self.enabled = enable
        option = self.get_noc_nmu_info()
        nodes = self.parser_noc_nmu_info(option)
        self.base_addr = self.get_noc_nmu_addr(nodes)
        self.npi_base_addr = self.get_noc_npi_addr(nodes)
        self.noc_nmu_addr = (c_int * len(self.base_addr))(*self.base_addr)
        self.nmuLib.create_noc_nmu_instance(self.type.encode(),
                int(self.npi_freq),self.noc_nmu_addr,
                int(self.npi_base_addr),int(len(self.base_addr)))
        self.aie_run_thread = threading.Thread(target=self.aie_run_loop)

   
    def start(self, interval=0.01):
        if self.enabled == False:
            return
        self.started = True
        self.interval = interval
        self.nmuLib.start.restype = c_int
        self.aie_run_thread.start()
        return self.nmuLib.start(c_double(interval))

    def stop(self):

        struct_data_len = 10
        if self.enabled == False:
            return

        self.started = False
        self.aie_run_thread.join()
        self.nmuLib.stop()
        data = record_data_factory(len(self.base_addr) * 2)
        pd = pointer(data)

        while (self.nmuLib.pop_data(pd) == 0):
            self.data.append(copy.deepcopy(data))

        self.nmuLib.get_act_period.restype = c_double
        self.act_period = self.nmuLib.get_act_period()


        self.nmuLib.get_record_data_len.restype = c_int
        self.record_data_len = self.nmuLib.get_record_data_len()
    


    def transTimebase(self):
        if self.enabled == False:
            return
        for i in range(0, len(self.data)):
            for j in range(0, self.record_data_len):
                self.data[i].data[j] = int(
                    self.data[i].data[j] / self.act_period)

    def getData(self):
        return [d.output() for d in self.data]

    def printData(self, data):
        if self.enabled == False:
            return
        info = str()
        info += "TimeStamp: %.7f\n" % data.time

        read = [data.data[i]/self.interval/1000/1000 
                for i in range(0, 10) if i % 2 == 0]
        info += "Read Ports:  "
        for d in read:
            info += "%8.1f" % d

        info += " MB/s\n"

        write = [data.data[i]/self.interval/1000 /1000
                for i in range(0, 10) if i % 2 == 1]
        info += "Write Ports: "
        for d in write:
            info += "%8.1f" % d

        info += " MB/s\n  "

        return info

    def get_noc_nmu_info(self):
        default_conf_json = ""
        cmd_parser = argparse.ArgumentParser(prog="test")
        cmd_parser.add_argument("-c", dest="config", nargs='?',
                 default=default_conf_json, help="Specify the config file")
        args = cmd_parser.parse_args()
        cfg_path = args.config
        if cfg_path != "":
            try:
                cfg_file = open(cfg_path, 'rt')
                cfg = json.load(cfg_file)
                overlayOption = cfg.get('options', {})
                return overlayOption
            except:
                logging.error(f"Invalid config file: {cfg_path}")
                exit(-1)
                logging.info(f"Applying Config File {cfg_path}")
            else:
                cfg = options
                overlayOption = {}
        return option

    def parser_noc_nmu_info(self, option):
        nmuOption = option.get('tracer', {}).get('noc_perf', {})
        self.interval = nmuOption.get("interval",{})
        self.npi_freq = nmuOption.get("npi_freq",{})
        nodes = nmuOption.get("nodes", {})
        
        return nodes

    def get_noc_npi_addr(self, noc_nodes):
        for k in range(len(noc_nodes)):
            if("npi" == noc_nodes[k].get("type", {})):
                return int(noc_nodes[k].get("phy_address", {}), 16)

    def get_noc_nmu_addr(self, noc_nodes):
        noc_nmu_addr = []
        for k in range(len(noc_nodes)):
            if("nmu" == noc_nodes[k].get("type", {})):
                addr = noc_nodes[k].get("phy_address", {})
                noc_nmu_addr.append(int(addr, 16))
                self.location_name.append(noc_nodes[k].get("location_name", {}))
        if (len(noc_nmu_addr)) > 8:
            logging.error("The maximum of sample channals is 8!")
            exit(1)
        return noc_nmu_addr

    def aie_run_loop(self):
        while self.started:
            os.system("xdputil run  /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel  /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel")

    def toMB_s(self,r):
        return float(r) / 1000 / 1000

    def convert_to_json_format(self,raw_data,time_offset=0.0):
        len2 = len(raw_data[1]) - 1
        list_data = []
        cur_list_data = []
        cunt = 0
        for cunt in range(0,len2):
            for rr in raw_data:
                if rr[0] != "NOC":
                    continue
                r = rr[1:]
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
        json_name = []
        D = {}
        data = {
            "noc_nmu_timestamp": json_data[0]
        }

        for k in range(0, len(self.location_name)):
            json_name.append(self.location_name[k] + "_READ")
            json_name.append(self.location_name[k] + "_WRITE")

        for i in range(0,len(json_name)):
            D[json_name[i]] = json_data[i+1]
        
        merge(data, D) 
        json_data = json.dumps(data, indent=4, separators=(",", ":"))
        with open('noc_nmu_data.json','w') as json_file:
            json_file.write(json_data)


if __name__ == '__main__':
    noc = NOC()
    noc.start(noc.interval)
    time.sleep(1)
    noc.stop()
    noc.transTimebase()
    raw_data = noc.getData()
    data_cur = noc.convert_to_json_format(raw_data)
    noc.create_json_file(data_cur)
    
