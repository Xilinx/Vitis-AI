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


from ctypes import *
import logging

import tracer.tracerBase
import copy
import os

def record_data_factory(size=10):
    class APMRecord(Structure):
        _fields_ = [('time', c_double), ('data', c_longlong * size)]

        def output(self):
            data_cur = []
            list_data = []
            data_cur = [*self.data]
            list_data.append("NOC_NMU")
            list_data.append(self.time)
            for i in range(0, size):
                list_data.append(data_cur[i])
            return list_data
    data =  APMRecord()
    return data

class NMU:
    def __init__(self, options, path='/usr/lib/libnmuperf.so', enable=True):
        self.nmuLib = cdll.LoadLibrary(path)
        self.data = []
        self.base_addr = []
        self.location_name = []
        self.interval = 0
        self.npi_freq = 0
        self.act_period = 0
        self.record_data_len = 0
        self.started = False
        self.type = "noc_nmu"
        self.enabled = enable
        self.npi_freq = options.get("npi_freq",{})
        nodes = options.get("nodes", {})
        self.base_addr = self.get_noc_nmu_addr(nodes)
        self.npi_base_addr = self.get_noc_npi_addr(nodes)
        self.noc_nmu_addr = (c_int * len(self.base_addr))(*self.base_addr)
        self.nmuLib.create_noc_nmu_instance(self.type.encode(),int(self.npi_freq),self.noc_nmu_addr,self.npi_base_addr,int(len(self.base_addr)))

        

    def start(self, interval=1.0):
        if self.enabled == False:
            return
        self.interval = interval
        self.nmuLib.start.restype = c_int
        return self.nmuLib.start(c_double(interval))

    def pushData(self, data):
        if self.enabled == False:
            return
        for i in range(0, len(data.data)):
            data.data[i] = data.data[i]

        self.data.append(copy.deepcopy(data))


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
        if len(noc_nmu_addr) > 8:
            logging.error("The maximum of sample channals is 8!")
            exit(1)
        return noc_nmu_addr

    def stop(self):
        if self.enabled == False:
            return
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
        for i in range(0, len(self.data)):
            for j in range(0, self.record_data_len):
                self.data[i].data[j] = int(
                    self.data[i].data[j] / self.act_period)


class nmuTracer(tracer.tracerBase.Tracer):
    def __init__(self):
        super().__init__('nmu', source=[],
                         compatible={'machine': ["aarch64"]})
        self.nmu_info = {}
        self.nmu = None
        self.nmu_enable = False
        self.nodes = []
        self.location_name = []

    def prepare(self, option: dict, debug: bool):
       nmuOption = option.get('tracer', {}).get('noc_perf', {})
       if nmuOption.get('noc_nmu_json_enable',{}) == "enable":
           self.interval = nmuOption.get("interval", 0)
           self.npi_freq = nmuOption.get("npi_freq",{})
           self.nodes = nmuOption.get("nodes", {})
           self.nmu = NMU(nmuOption)
           self.location_name = self.nmu.location_name
           self.nmu_info.update({"interval": self.interval})
           self.nmu_info.update({"npi_freq" : self.npi_freq})
           self.nmu_info.update({"nodes": self.nodes})
           self.nmu_info.update({"location_name": self.location_name})
           self.nmu_enable = True
       option['noc_nmu'] = self.nmu_enable
       return option

    def start(self):
        super().start()
        if self.nmu_enable:
            self.nmu.start(self.interval)

    def stop(self):
        super().stop()
        if self.nmu_enable:
            self.nmu.stop()

    def process(self, data, t_range=[]):
        if self.nmu_enable:
            self.nmu.transTimebase()
        pass

    def compatible(self, platform: {}):
        if super().compatible(platform) == False:
            return False
        if platform.get('model').startswith('xlnx,zocl-versal'):
            return True
        else:
            return False
    
    def getData(self):
        if self.nmu_enable:
            return self.nmu_info,[d.output() for d in self.nmu.data]
        else:
            print("noc_nmu feature disabled")

tracer.tracerBase.register(nmuTracer())
