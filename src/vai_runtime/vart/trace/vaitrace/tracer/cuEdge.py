
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

import tracer.tracerBase
import sys
import os
import json
import ctypes


class cuEdgeTracer(tracer.tracerBase.Tracer):
    def __init__(self):
        super().__init__('cuEdge', source=[
            "ftrace"], compatible={'machine': ["aarch64"]})

    def prepare(self, conf: dict, debug: bool):
        "Handle Input Options"

        "Handle Output Options"
        optForFtrace = {
            "collector": {
                "ftrace": {
                    "cuEdge": {
                        "name": "cuEdge",
                        "type": "kprobe",
                        "saveTo": './cuEdge.trace',
                        "traceList": [
                            ["cu_start", "zocl", "zocl_hls_start",
                                ["cu_idx=+0(%x0):u32"]],
                            ["cu_done",  "zocl", "zocl_hls_check+0x70",
                                ["cu_idx=+0(%x20):u32"]]
                        ]
                    }
                }
            }
        }

        return optForFtrace

    def process(self, data, t_range):
        tmp = [l for l in data.get('ftrace', {}).get(
            self.name) if not l.startswith('#')]
        self.data = [l for l in tmp if l.find('cu_idx=0') < 0]

    def getData(self):
        return self.data


tracer.tracerBase.register(cuEdgeTracer())
