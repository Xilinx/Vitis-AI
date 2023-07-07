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


class NOCRecord(Structure):
    _fields_ = [('time', c_double), ('data', c_uint * 10)]

    def output(self):
        return "NOC %.7f %d %d %d %d %d %d %d %d %d %d\n" % (self.time, *self.data)


class NOC:
    def __init__(self, mem_type="noc", path='/usr/lib/libmemperf.so', enable=True):
        self.nocLib = cdll.LoadLibrary(path)
        self.data = []
        self.base_addr = []
        self.interval = 0
        self.act_period = 0
        self.record_data_len = 0
        self.type = mem_type
        self.enabled = enable
        self.get_noc_base_addr()
        self.noc_base_addr = (c_int * len(self.base_addr))(*self.base_addr)
        self.nocLib.create_instance(
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

    def get_noc_base_addr(self):
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
                    self.base_addr.append(int(addr_off))
                    self.base_addr.sort()


if __name__ == '__main__':

    noc = NOC()
    noc.start(0.001)
    time.sleep(1)
    noc.stop()

    for a in noc.data:
        print(noc.printData(a))
