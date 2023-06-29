
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
from writer.parser.ftraceUtil import *


def cppFuncPaired(start: vaiTimelineEvent, end: vaiTimelineEvent):
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


def convert_cppfunc(cppfunc_data):
    threads = {}
    pairedEvents = []
    for l in cppfunc_data:
        event = ftraceParse(l, {}).toTimelineEvent()
        pid = event.pid
        threads.setdefault(pid, []).append(event)

    stack = []
    for pid in threads.keys():
        for e in threads[pid]:
            stack.append(e)
            if len(stack) < 2:
                continue
            for s in stack[::-1]:
                r = cppFuncPaired(s, e)
                if r != None:
                    try:
                        stack.remove(e)
                        stack.remove(s)
                    except BaseException:
                        pass

                    pairedEvents.append(r)
                    break

    cpp_func_calls = statTable("C/C++ Function")
    for event in pairedEvents:
        if event.type != 'period':
            continue
        cpp_func_calls.add(event.code, event.duration)
    return cpp_func_calls.output()
