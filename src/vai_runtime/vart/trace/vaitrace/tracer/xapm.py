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

idcode_ddrmc_num_dict = {'0x4c9b093':3, '0x4c93093':2, '0x4c98093':3, '0x4ca9093':4, '0x4ca8093':4, '0x4cd2093':3, '0x4cd0093':3,
         '0x4c9a093':3, '0x4cc1093':1, '0x4cc0093':1, '0x4cc9093':1, '0x4cc8093':1, '0x4cd2093':3, '0x4cd3093':3}
class APMRecord(Structure):
    _fields_ = [('time', c_double), ('data', c_ulonglong * 10)]

    def output(self):
        return "APM %.7f %d %d %d %d %d %d %d %d %d %d\n" % (self.time, *self.data)


class APM:
    def __init__(self, mem_type, devid, path='/usr/lib/libmemperf.so', enable=True):
        self.apmLib = cdll.LoadLibrary(path)
        self.data = []
        self.base_addr = []
        self.interval = 0
        self.act_period = 0
        self.record_data_len = 0
        self.type = mem_type
        self.enabled = enable
        if mem_type == "noc":
            self.base_addr = self.get_noc_base_addr(self.type, devid)
            if self.enabled == False:
                return
            self.noc_base_addr = (c_int * len(self.base_addr))(*self.base_addr)
            self.apmLib.create_noc_instance(
                    self.type.encode(), self.noc_base_addr, int(len(self.base_addr)))
        elif mem_type == "apm":
            self.apmLib.create_apm_instance(self.type.encode())

    def start(self, interval=1.0):
        if self.enabled == False:
            return
        self.interval = interval
        self.apmLib.start.restype = c_double
        return self.apmLib.start(c_double(interval))

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

        read = [data.data[i]/self.act_period/1024 /
                1024 for i in range(0, 10) if i % 2 == 0]
        info += "Read Ports:  "
        for d in read:
            info += "%8.1f" % d

        info += " MB/s\n"

        write = [data.data[i]/self.act_period/1024 /
                 1024 for i in range(0, 10) if i % 2 == 1]
        info += "Write Ports: "
        for d in write:
            info += "%8.1f" % d

        info += " MB/s\n  "

        return info

    def get_noc_base_addr(self, mem_type, devid):
        if mem_type != "noc":
            return
        ddrmc_number = idcode_ddrmc_num_dict.get(devid, 0)
        if ddrmc_number == 0:
            logging.info("Can not get ddrmc/noc number!")
            self.enabled = False
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
        return pre_base_addr[0:ddrmc_number]

    def stop(self):
        if self.enabled == False:
            return
        self.apmLib.stop()

        data = APMRecord()
        pd = pointer(data)

        while (self.apmLib.pop_data(pd) == 0):
            self.data.append(copy.deepcopy(data))

        self.apmLib.get_act_period.restype = c_double
        self.act_period = self.apmLib.get_act_period()

        self.apmLib.get_record_data_len.restype = c_int
        self.record_data_len = self.apmLib.get_record_data_len()

    def transTimebase(self):
        if self.enabled == False:
            return
        for i in range(0, len(self.data)):
            for j in range(0, self.record_data_len):
                self.data[i].data[j] = int(
                    self.data[i].data[j] / self.act_period)


def checkAPM():
    if os.path.exists("/dev/uio1") == False:
        return False

    """/sys/class/uio/uio1/device/of_node/name"""
    if os.path.exists("/sys/class/uio/uio1/device/of_node/name") == False:
        return False

    with open("/sys/class/uio/uio1/device/of_node/name", "rt") as f:
        try:
            name = f.read().strip()
            if name.startswith('perf-monitor') == False:
                return False
        except:
            return False

    return True


class xapmTracer(tracer.tracerBase.Tracer):
    def __init__(self):
        super().__init__('xapm', source=[],
                         compatible={'machine': ["aarch64"]})
        self.apm = None
        self.apm_type = None
        self.devid = None
        self.apm_json_enable = None
        self.sample_base_addr = []

    def prepare(self, option: dict, debug: bool):
        "Handle Input Options"
        xapmOption = option.get('tracer', {}).get('xapm', {})
        self.interval = xapmOption.get("APM_interval", 0.002)
        self.apm = APM(self.apm_type, self.devid)
        if self.apm_type == "noc":
            self.sample_base_addr = self.apm.base_addr
            option.update({"ddrmc_base_addr": self.sample_base_addr})
        "Handle Output Options"
        self.apm_json_enable = option.get('tracer', {}).get('noc_perf', {}).get('ddrmc_json_enable', {})
        option['ddrmc_json_enable'] = self.apm_json_enable
        return option

    def start(self):
        super().start()
        # self.timesync = self.apm.start(self.interval)
        self.apm.start(self.interval)

    def stop(self):
        super().stop()
        self.apm.stop()

    def process(self, data, t_range=[]):
        self.apm.transTimebase()

    def compatible(self, platform: {}):
        if super().compatible(platform) == False:
            return False

        if platform.get('model').startswith('xlnx,zocl-versal'):
            self.apm_type = "noc"
            self.devid = '0x' + platform.get('idcode')[-7:]
            return True
        elif platform.get('model').startswith('xlnx,zocl'):
            self.apm_type = "apm"
            return checkAPM()
        else:
            return False

    def getData(self):
        if self.apm.enabled == False:
            return
        return [d.output() for d in self.apm.data]


if __name__ == '__main__':
    apm = APM()
    apm.start(0.001)
    time.sleep(1)
    apm.stop()

    for a in apm.data:
        print(apm.printData(a))

else:
    tracer.tracerBase.register(xapmTracer())
