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

import ctypes
import json
import tracer.tracerBase


class pyfuncTracer(tracer.tracerBase.Tracer):
    def __init__(self):
        super().__init__('pyfunc', source=['tracepoint'], compatible={
            'machine': ["x86_64", "aarch64"]})
        self.traceLog = []
        self.timesync = 0

    def process(self, data, t_range=[]):
        for s in self.source:
            d = data.get(s, None)
            if d is None:
                continue
            for l in d:
                try:
                    """EVENT_TIME_SYNC 0 -1 XRT -1 0.000000610 6285.087307830"""
                    # if l.startswith('EVENT_TIME_SYNC'):
                    #    xrt_t = float(l.strip().split()[5])
                    #    steady_t = float(l.strip().split()[6])
                    #    self.timesync = steady_t - xrt_t
                    # if l.find(PYFUNC_FILTER_TAGS) < 0:
                    if l.get("classname", None) != "py":
                        continue
                except:
                    continue

                self.traceLog.append(l)

    def start(self):
        super().start()

    def stop(self):
        libvart_trace = ctypes.CDLL('libvart-trace.so')
        f_trace_stop = libvart_trace.stop
        f_trace_stop.argtypes = None
        f_trace_stop.restype = None

        f_trace_stop()
        super().stop()

    def prepare(self, options: dict, debug: bool):
        self.saveTo = None
        if debug:
            self.saveTo = "./%s.trace" % self.name

    def getData(self):
        if self.saveTo != None:
            with open(self.saveTo, "w+t") as save:
                save.writelines(json.dumps(self.traceLog))
        return self.traceLog


tracer.tracerBase.register(pyfuncTracer())
