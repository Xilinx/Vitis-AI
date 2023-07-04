#!/usr/bin/python3
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
import atexit
import copy

available_boards = ["zcu102", "zcu104", "vck190", "kv260"]


class hwmon_ina226:
    def __init__(self, path, libpath='/usr/lib/libpowermon.so'):
        self.valid = False
        self.board = ""
        self.sample_phases = 1
        self.powerLib = cdll.LoadLibrary(libpath)

        if not os.path.exists(path):
            print("Err: Invalid path [%s]" % path)
            return

        try:
            self.path = os.path.abspath(path)
            self.name = open(os.path.join(path, "name"), "rt").read().strip()
        except:
            print("Err: Invalid path [%s]" % path)
            return

        if self.name.find("ina2") < 0:
            print("Err: Invalid hwmon [%s], name: [%s]" % (path, self.name))
            return

        # check_board
        host_name = os.uname().nodename
        # 'xilinx-zcu102-20221'

        for b in available_boards:
            if host_name.find(b) >= 0:
                self.board = b
        if self.board == "":
            print("Err: Invalid board name: [%s]" % host_name)
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
            print("Err: Invalid path [%s]" % path)
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


mons = []
power_records = []
sys_hw_mon = "/sys/class/hwmon/"


def print_statistics():
    idle_power = min(power_records)
    max_power = max(power_records)
    ave_power = sum(power_records) / len(power_records)

    report_fmt = "Power Statistics: Idle {:.3f} w, Peak {:.3f} w, Average {:.3f} w"
    report_str = report_fmt.format(idle_power, max_power, ave_power)
    report_separator = "=" * len(report_str)

    print("\n\n{}\n{}\n{}".format(report_separator, report_str, report_separator))


for i in range(0, 100):
    mon_path = os.path.join(sys_hw_mon, "hwmon%d" % i)
    if os.path.exists(mon_path):
        mon = hwmon_ina226(mon_path)
        if mon.valid:
            mons.append(mon)

if __name__ == "__main__":
    try:
        while True:
            power = 0

            for m in mons:
                #print("%5.4f" % (m.sample_power()), end = " " )
                power += m.sample_power()

            print("%.3f" % power)
            print("-" * 10)
            power_records.append(power)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print_statistics()
