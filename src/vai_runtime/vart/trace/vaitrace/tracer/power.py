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
import sys
import os
import time
import threading
import logging
import tracer.tracerBase
import copy

supported_boards = ["zcu102", "zcu104", "vck190", "kv260"]
sys_hw_mon = "/sys/class/hwmon/"


class hwmon_ina226:
    def __init__(self, path, libpath='/usr/lib/libpowermon.so'):
        self.valid = False
        self.board = ""
        self.sample_phases = 1
        self.powerLib = cdll.LoadLibrary(libpath)

        if not os.path.exists(path):
            logging.error("Invalid path [%s]" % path)
            return

        try:
            self.path = os.path.abspath(path)
            self.name = open(os.path.join(path, "name"), "rt").read().strip()
        except:
            logging.error("Invalid path [%s]" % path)
            return

        if self.name.find("ina2") < 0:
            logging.info("Invalid hwmon [%s], name: [%s]" % (path, self.name))
            return

        """
        Fix:
        INA226 for VCCINT monitors output of only one phase of IR35215 regulator. 
        Since VCCINT power rail is supplied by 6 phases of that regulator,
        you would need to multiple INA226 readings by 6
        (that is what sc_app utility do)
        """
        if self.board == "vck190" and self.path.find("hwmon0") >= 0:
            self.sample_phases = 6

        self.v = []
        self.c = []
        self.p = []

        self.volt = 0

        for r, d, files in os.walk(path):
            if os.path.abspath(r) != self.path:
                continue
            for f in files:
                if f.startswith("curr"):
                    self.c.append(os.path.abspath(os.path.join(r, f)))
                if f.startswith("power"):
                    self.p.append(os.path.abspath(os.path.join(r, f)))
                if f.startswith("in"):
                    self.v.append(os.path.abspath(os.path.join(r, f)))

        self.powerLib.get_device_volt.restype = c_int
        self.volt = max([int(self.powerLib.get_device_volt(v.encode()))
                        for v in self.v]) / 1000.0

        if self.volt == 0:
            logging.error("Invalid path [%s]" % path)
            return

        self.valid = True

    def sample_curr(self):
        if self.valid == False:
            return 0
        self.powerLib.get_device_curr.restype = c_int
        curr = max([int(self.powerLib.get_device_curr(c.encode()))
                   for c in self.c]) / 1000.0

        return self.sample_phases * curr

    def sample_power(self):
        if self.valid == False:
            return 0
        self.powerLib.get_device_power.restype = c_int
        power = max([int(self.powerLib.get_device_power(p.encode()))
                    for p in self.p]) / 1000.0 / 1000.0
        return self.sample_phases * power


class PowerTracer(tracer.tracerBase.Tracer):
    def __init__(self):
        super().__init__('power', source=[],
                         compatible={'machine': ["aarch64"]})
        self.mons = []
        self.board = None
        self.power_records = []
        self.power_info = []
        self.started = False
        self.sample_interval = None
        self.sample_thread = threading.Thread(target=self.sample_loop)

    def process(self, data, t_range=[]):
        pass

    def start(self):
        super().start()
        logging.debug("Power Tracer Start")
        self.started = True
        self.sample_thread.start()

    def stop(self):
        super().stop()
        logging.debug("Power Tracer Stop")
        self.started = False
        logging.debug("Power Tracer waiting sample thread stop")
        self.sample_thread.join()

    def checkBoard(self):
        host_name = os.uname().nodename
        # 'xilinx-zcu102-20221'

        for b in supported_boards:
            if host_name.find(b) >= 0:
                self.board = b
        if self.board is None:
            logging.error("Invalid board name: [%s]" % host_name)
            return False

        return True

    def compatible(self, platform: {}):
        if super().compatible(platform) == False:
            return False

        return self.checkBoard()

    def prepare(self, options: dict, debug: bool):
        power_tracer_option = options.get('tracer', {}).get('power', {})
        self.sample_interval = power_tracer_option.get("sample_interval", 0.1)

        logging.info(f"Power tracer sample interval: {self.sample_interval}s")

        for i in range(0, 50):
            mon_path = os.path.join(sys_hw_mon, "hwmon%d" % i)
            if os.path.exists(mon_path):
                mon = hwmon_ina226(mon_path)
                if mon.valid:
                    self.mons.append(mon)

    def getData(self):
        return self.power_records

    def sample_loop(self):
        while self.started:
            power = 0
            for m in self.mons:
                power += m.sample_power()
            self.power_records.append(power)
            time.sleep(self.sample_interval)


tracer.tracerBase.register(PowerTracer())
