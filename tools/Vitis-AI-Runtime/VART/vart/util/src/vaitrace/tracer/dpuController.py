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


import os
import sys
import re
import json
import pickle
import tracer.tracerBase


DPU_CONTROLLER_FILTER = """EVENT_"""
DPU_CONTROLLER_FILTER_TAGS = ["DPUC", "DPUR"]


class DPUControllerTracer(tracer.tracerBase.Tracer):
    def __init__(self):
        super().__init__('DPUController', source=[
            'tracepoint'], compatible={'machine': ["x86_64", "aarch64"]})
        self.traceLog = []
        self.timesync = 0

    def process(self, data, t_range=[]):
        for s in self.source:
            d = data.get(s, None)
            if d is None:
                continue
            for l in d:
                """#XRT_TIMELINE_SYNC=1600004209.089879036"""
                if l.startswith('EVENT_TIME_SYNC'):
                    self.timesync = float(
                        l.strip().split()[6]) - float(l.strip().split()[5])

                # if l.find(DPU_CONTROLLER_FILTER) < 0:
                #    continue
                try:
                    tag = l.split()[3]
                    if not tag in DPU_CONTROLLER_FILTER_TAGS:
                        continue
                except:
                    continue
                self.traceLog.append(l)

    def start(self):
        super().start()

    def stop(self):
        super().stop()

    def prepare(self, options: dict, debug: bool):
        self.saveTo = None
        if debug:
            self.saveTo = "./%s.trace" % self.name

    def getData(self):
        if self.saveTo != None:
            with open(self.saveTo, "w+t") as save:
                save.writelines(self.traceLog)
        return self.traceLog


tracer.tracerBase.register(DPUControllerTracer())
