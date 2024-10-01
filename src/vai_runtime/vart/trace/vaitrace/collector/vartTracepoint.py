
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

import sys
import os
import collector
import logging
import json
from tempfile import TemporaryDirectory


def trans_col(i):
    result = {}
    for rec in i.split(';'):
        if len(rec.split(':')) < 2:
            continue
        k, v = rec.split(':', 1)
        result.update({k: v})

    return result


"""
Trace data is stored in dict & sorted by classname
{
    "null" : [...] (classname not specified in raw trace)
    "class_a": [...],
    "class_b": [...]
}
"""


def trans_row(trace_data, xmodel_info):
    current_section = "null"
    result = []

    for d in trace_data:
        if d.startswith('#'):
            if d.startswith("#SECTION"):
                current_section = trans_col(d).get('#SECTION', "null")
            continue

        item = trans_col(d)
        item.update({"section": current_section})
        cname = item.get("classname", "null")

        result.append(item)

    for d in xmodel_info:
        item = json.loads(d)
        item.pop("mc_code_sstr.str()")
        item.update({"section": "XMODEL"})

        result.append(item)

    return result


class tracepointCollector(collector.collectorBase.Collector):
    def __init__(self, stdout="", stderr=""):
        super().__init__(name='tracepoint')
        self.logdir = TemporaryDirectory()
        self.logdirName = self.logdir.name + '/'

    def __del__(self):
        self.logdir.cleanup()

    def prepare(self, conf: dict) -> dict:

        # "VAI_TRACE_ENABLE"=1
        # "VAI_TRACE_DIR"="/tmp/vai/"
        # "VAI_TRACE_TS" = "boot" or "x86_tsc"

        os.environ.setdefault("VAI_TRACE_ENABLE", "true")
        os.environ.setdefault("VAI_TRACE_DIR", self.logdirName)

        # preferClocks = ["boot", "x86-tsc", "global"]
        ts_mode = conf.get("control").get("traceClock")
        logging.debug(ts_mode)
        """
        +--------------+-------------------+------------+
        | clock_source | x86               | arm        |
        +--------------+-------------------+------------+
        | VA           | global + XRT      | boot + XRT |
        | XAT          | x86-tsc + x86-tsc | boot + XRT |
        +--------------+-------------------+------------+
        """
        os.environ.setdefault("VAI_TRACE_TS", ts_mode)

        va_enabled = conf['cmdline_args']['va']
        if va_enabled:
            os.environ["VAI_TRACE_TS"] = "XRT"
            if ts_mode == "x86-tsc":
                """VTF do not accept x86-tsc"""
                conf["control"]["traceClock"] = "global"
                print(conf["control"]["traceClock"])

        return conf

    def start(self):
        super().start()

    def stop(self):
        """Clear env"""
        os.environ.pop("VAI_TRACE_ENABLE", None)
        super().stop()

    def getData(self):
        traceEvents = []
        xmodelInfo = []

        for root, dirs, files in os.walk(self.logdirName):
            for name in files:
                f = os.path.join(root, name)
                if name.find("subgraph_info") > 0:
                    xmodelInfo += open(f).readlines()
                else:
                    traceEvents += open(f).readlines()

        return trans_row(traceEvents, xmodelInfo)


collector.collectorBase.register(tracepointCollector())
