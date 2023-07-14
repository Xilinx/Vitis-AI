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


import os
import sys
import re
import json
import pickle
import tracer.tracerBase


class vartTracer(tracer.tracerBase.Tracer):
    def __init__(self):
        super().__init__('vart', source=[
            'tracepoint'], compatible={'machine': ["x86_64", "aarch64"]})
        self.traceData = []
        self.timesync = 0

    def process(self, data, t_range=[]):
        for s in self.source:
            d = data.get(s, None)
            if d is None:
                continue
            self.traceData = d

            self.timesync = 0

            for trace in d:
                if trace.get('classname', "") == "trace_timesync":
                    self.timesync = float(
                        trace.get('steady_clock', 0)) - float(trace.get('xrt_ts', 0))
                    break

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
                for t in self.traceData:
                    save.write(str(t))
                    save.write('\n')
        return self.traceData


tracer.tracerBase.register(vartTracer())
