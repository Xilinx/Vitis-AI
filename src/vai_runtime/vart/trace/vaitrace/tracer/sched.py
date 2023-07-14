
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

import vaitraceSetting
import tracer.tracerBase
import sys
import os
import json
import ctypes


class schedTracer(tracer.tracerBase.Tracer):
    def __init__(self):
        super().__init__('sched', source=["ftrace"], compatible={
            'machine': ["aarch64"]})

    def prepare(self, option: dict, debug: bool):
        #
        # dpu.enableEvent("sched", "sched_switch",\
        #     """(prev_comm == "%s") || (next_comm == "%s")""" % (comm, comm))
        #
        "Handle Input Options"
        #comm = args.cmd[0].split('/')[-1][:15]
        comm = option['control']['cmd'][0].split('/')[-1][:15]
        python_mode = option.get("cmdline_args", {}).get('python', False)
        if python_mode:
            comm = "vaitrace"

        "Handle Output Options"
        saveTo = None
        if debug:
            saveTo = './sched.trace'

        optForFtrace = {
            "collector": {
                "ftrace": {
                    "sched": {
                        "name": "sched",
                        "type": "event",
                        "saveTo": saveTo,
                        "traceList": [
                            ["sched", "sched_process_exec"],
                            ["sched", "sched_process_fork"],
                            ["sched", "sched_process_exit"],
                            ["sched", "sched_switch",
                             """(prev_comm == "%s") || (next_comm == "%s")""" % (comm, comm)]
                        ]
                    }
                }
            }
        }

        return optForFtrace

    def compatible(self, platform: {}):
        if super().compatible(platform) == False:
            return False

        """Do some tests"""
        return vaitraceSetting.checkFtrace()

    def process(self, data, t_range=[]):
        self.data = [l for l in data.get('ftrace', {}).get(
            self.name) if not l.startswith('#')]

    def getData(self):
        return self.data


tracer.tracerBase.register(schedTracer())
