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

import copy
import os
import time


class APMRecord(Structure):
    _fields_ = [('time', c_double), ('data', c_uint * 10)]

    def output(self):
        return "APM %.7f %d %d %d %d %d %d %d %d %d %d\n" % (self.time, *self.data)


class APM:
    def __init__(self, path='./libxapm.so', enable=True):
        self.apmLib = cdll.LoadLibrary(path)
        self.data = []
        self.interval = 0
        self.enabled = enable

    def start(self, interval=1.0):
        if self.enabled == False:
            return
        self.interval = interval
        self.apmLib.apm_start.restype = c_double
        return self.apmLib.apm_start(c_double(interval))

    def pushData(self, data):
        if self.enabled == False:
            return
        for i in range(0, len(data.data)):
            data.data[i] = data.data[i]

        self.data.append(copy.deepcopy(data))

    def printData(self, data):
        if self.enabled == False:
            return
        info = str()
        info += "TimeStamp: %.7f\n" % data.time

        read = [data.data[i]/self.interval/1024 /
                1024 for i in range(0, 10) if i % 2 == 0]
        info += "Read Ports:  "
        for d in read:
            info += "%8.1f" % d

        info += " MB/s\n"

        write = [data.data[i]/self.interval/1024 /
                 1024 for i in range(0, 10) if i % 2 == 1]
        info += "Write Ports: "
        for d in write:
            info += "%8.1f" % d

        info += " MB/s\n  "

        return info

    def stop(self):
        if self.enabled == False:
            return
        self.apmLib.apm_stop()

        data = APMRecord()
        pd = pointer(data)

        while (self.apmLib.apm_pop_data(pd) == 0):
            self.data.append(copy.deepcopy(data))

    def transTimebase(self):
        for i in range(0, len(self.data)):
            for j in range(0, len(self.data[i].data)):
                self.data[i].data[j] = int(
                    self.data[i].data[j] / self.interval)


if __name__ == '__main__':
    apm = APM()
    apm.start(0.001)
    time.sleep(1)
    apm.stop()

    print("======================")
    print(len(apm.data))
    print("======================")
    apm.transTimebase()
    # print(apm.data)
    for a in apm.data:
        print(apm.printData(a))
