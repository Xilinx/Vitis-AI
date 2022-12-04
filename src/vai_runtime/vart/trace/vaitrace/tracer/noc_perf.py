#!/usr/bin/python3
# -*- coding:utf-8 -*-

# Copyright 2019 Xilinx Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys, os, logging, time, math, copy, threading
from ctypes import *
import copy
import logging
import tracer.tracerBase

class NOCRecord(Structure):
    _fields_ = [('time', c_double), ('data', c_uint * 8)]

class NOC:
    def __init__(self, path='/usr/lib/libnoc.so', enable=True):
        self.nocLib = cdll.LoadLibrary(path)
        self.data = []
        self.interval = 0
        self.act_period = 0
        self.enabled = enable

    def start(self, interval=0.01):
        if self.enabled == False:
            return
        self.interval = interval
        self.nocLib.noc_start.restype = c_int
        return self.nocLib.noc_start(c_double(interval))

    def stop(self):
        if self.enabled == False:
            return

        self.nocLib.noc_stop()
        data = NOCRecord()
        pd = pointer(data)

        while (self.nocLib.noc_pop_data(pd) == 0):
            self.data.append(copy.deepcopy(data))

        self.nocLib.noc_act_period.restype = c_double;
        self.act_period = self.nocLib.noc_act_period()

    def transTimebase(self):
        if self.enabled == False:
            return
        for i in range(0, len(self.data)):
            for j in range(0, len(self.data[i].data)):
                self.data[i].data[j] = int(
                        self.data[i].data[j])

    def printData(self):
        if self.enabled == False:
            return
        bw_sum = 0
        bw_list = []

        for i in range(0, len(self.data)):
            for j in range(len(self.data[i].data)):
                bw_sum += (self.data[i].data[j] / 1000 / 1000 / self.act_period )
            bw_list.append(bw_sum)

        report_fmt_str = f"DDR Bandwidth Report: Peak [{max(bw_list):.3f} MB/s], Average: [{sum(bw_list) / len(bw_list):.3f} MB/s]"
        report_separator = "=" * len(report_fmt_str)

        print("{}\n{}\n{}\n".format(report_separator, report_fmt_str, report_separator))



class nocPerfTracer(tracer.tracerBase.Tracer):
    def __init__(self):
        super().__init__('noc_perf', source=[],
                         compatible={'machine': ["aarch64"]})
        self.noc = None
    
    def prepare(self, option: dict, debug: bool):
        noc_perf_tracer_option = option.get('tracer', {}).get('noc_perf', {})
        self.sampling_period = noc_perf_tracer_option.get("sample_interval", 0.001)

        self.noc = NOC()
        "Handle Output Options"
        return option

    def start(self):
        super().start()

        print("## NoC Perf Tracer Start")
        self.noc.start(self.sampling_period)

    def stop(self):
        super().stop()
        print("## NoC Perf waiting sample thread stop")
        self.noc.stop()

    def process(self, data, t_range=[]):
        print(f"## NoC Perf Process, n: {len(self.noc.data)}")
        self.noc.transTimebase()
        self.noc.printData()


    def compatible(self, platform: {}):
        if super().compatible(platform) == False:
            return False

        if not platform.get('model').startswith('xlnx,zocl-versal'):
            return False

        return True

    def getData(self):
        return []

tracer.tracerBase.register(nocPerfTracer())
