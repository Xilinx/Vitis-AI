
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

import parser.parserBase
from parser.tracepointUtil import *
from parser.statUtil import *

def pyFuncPaired(start: vaiTimelineEvent, end: vaiTimelineEvent):
    if not (start.type == 'start' and end.type == 'done'):
        return None
    if end.ts < start.ts:
        return None
    if start.pid != end.pid:
        return None
    if start.code != end.code:
        return None

    pairedEvent = vaiTimelineEvent(
        start.startTime, start.pid, start.type, start.coreType, start.coreId)

    pairedEvent.type = 'period'
    pairedEvent.startTime = start.ts
    pairedEvent.endTime = end.ts
    pairedEvent.duration = end.ts - start.ts
    pairedEvent.code = start.code

    return pairedEvent


def convert_pyfunc(pyfunc_data, options):
    threads = {}
    pairedEvents = []

    transTs = lambda x: round(float(x), 6)
    if options.get('traceClock', "") == 'x86-tsc':
        tsc_hz = int(options.get('x86_tsc_khz') * 1000)
        transTs = lambda x: round((float(x) / tsc_hz), 6)

    for l in pyfunc_data:
        event = tracepointEvent(l).toTimelineEvent()
        if event.coreType != "PY":
            continue
        pid = event.pid
        event.ts = transTs(event.ts)
        threads.setdefault(pid, []).append(event)

    stack = []
    for pid in threads.keys():
        for e in threads[pid]:
            stack.append(e)
            for s in stack[::-1]:
                r = pyFuncPaired(s, e)
                if r != None:
                    try:
                        stack.remove(e)
                        stack.remove(s)
                    except BaseException:
                        pass

                    pairedEvents.append(r)

    py_func_calls = statTable("Python Function")
    for event in pairedEvents:
        if event.type != 'period':
            continue
        py_func_calls.add(event.code, event.duration)

    return py_func_calls


class pyfuncParser(parser.parserBase.Parser):
    def __init__(self):
        super().__init__('pyfunc')

    def parse(self, data, options):
        py_func = convert_pyfunc(data, options)
        return {"STAT-PYFUNC": py_func.output(fmt="list")}

parser.parserBase.register(pyfuncParser())
