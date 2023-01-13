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


class NOC:
    def __init__(self, path='./libnoc.so', enable=True):
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

        self.nocLib.noc_act_period.restype = c_double
        self.act_period = self.nocLib.noc_act_period()

    def transTimebase(self):
        if self.enabled == False:
            return
        for i in range(0, len(self.data)):
            for j in range(0, len(self.data[i].data)):
                self.data[i].data[j] = int(
                    self.data[i].data[j])

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


if __name__ == '__main__':

    noc = NOC()
    noc.start(0.001)
    time.sleep(1)
    noc.stop()

    for a in noc.data:
        print(noc.printData(a))
