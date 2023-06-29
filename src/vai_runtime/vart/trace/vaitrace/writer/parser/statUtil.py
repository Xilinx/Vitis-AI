
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

import statistics


class statItem:
    def __init__(self, name, timeunit="ms"):
        self.name = name
        self.runs = 0
        self.total_t = 0.0
        self.timeunit = timeunit
        self.raw_t = []

    def add(self, time):
        self.runs += 1
        self.total_t += time
        self.raw_t.append(time)

    @property
    def min_t(self):
        if len(self.raw_t) == 0:
            return 0
        return min(self.raw_t)

    @property
    def max_t(self):
        if len(self.raw_t) == 0:
            return 0
        return max(self.raw_t)

    @property
    def ave_t(self):
        if len(self.raw_t) == 0:
            return 0
        return statistics.median(self.raw_t)

    def setTimeUnit(self, _timeunit):
        if _timeunit == "s" or _timeunit == "ms" or _timeunit == "us":
            self.timeunit = _timeunit

    def __str__(self):
        if self.timeunit == "ms":
            tu = 1000
        elif self.timeunit == "us":
            tu = 1000 * 1000
        else:
            tu = 1

        min_t = self.min_t * tu
        max_t = self.max_t * tu
        ave_t = self.ave_t * tu
        total_t = self.total_t * tu

        return "%s,%d,CPU,%.3f,%.3f,%.3f,\n" % (self.name, self.runs, min_t, ave_t, max_t)


class statTable:
    def __init__(self, name):
        self.items = {}
        self.name = name

    def add(self, name, time):
        if name not in self.keys():
            self.items.update({name: statItem(name)})
        self.items.get(name).add(time)

    def keys(self):
        return self.items.keys()

    def output(self, fmt="csv", timeunit="ms"):
        csv = []
        #csv.append(self.name + " " + "Summary\n")
        #csv.append("Function Name,Number Of Runs,Minimum Time (ms),Maximum Time (ms),Average Time (ms),\n")
        for k in self.items.keys():
            self.items[k].setTimeUnit(timeunit)
            csv.append(str(self.items[k]))
        return csv


"""
DPU Summary
Kernel Name,Number Of Runs,CU Full Name,Minimum Time (ms),Maximum Time (ms),Average Time (ms),
"""
