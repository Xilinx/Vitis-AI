
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

from writer.parser.timelineUtil import *
from writer.parser.tracepointUtil import *
from writer.parser.statUtil import *


def get_funcname(data: dict) -> str:
    return


def pyFuncPaired(start: vaiTimelineEvent, end: vaiTimelineEvent):
    if not (start.type == 'start' and end.type == 'done'):
        return None
    if end.ts < start.ts:
        return None
    if start.pid != end.pid:
        return None
    if start.get_info('py_func_name') != end.get_info('py_func_name'):
        return None

    pairedEvent = vaiTimelineEvent(
        start.startTime, start.pid, start.type, start.coreType, start.coreId, start.info)

    pairedEvent.type = 'period'
    pairedEvent.startTime = start.ts
    pairedEvent.endTime = end.ts
    pairedEvent.duration = end.ts - start.ts

    return pairedEvent


def convert_pyfunc(data):
    data = select_trace_classes(data, ["py"])
    threads = {}
    pairedEvents = []
    for l in data:
        event = tracepointEvent(l).toTimelineEvent()
        if event.coreType != "py":
            continue
        pid = event.pid
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
        py_func_calls.add(event.get_info('py_func_name'), event.duration)

    return py_func_calls.output()
