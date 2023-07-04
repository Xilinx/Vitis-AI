
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
from functools import reduce


def get_funcname(data: dict) -> str:
    return


def cpuTaskPaired(start: vaiTimelineEvent, end: vaiTimelineEvent):
    if not (start.type == 'start' and end.type == 'done'):
        return None
    if end.ts < start.ts:
        return None
    if start.pid != end.pid:
        return None
    if start.get_info('subgraph') != end.get_info('subgraph'):
        return None

    pairedEvent = vaiTimelineEvent(
        start.startTime, start.pid, start.type, start.coreType, start.coreId, start.info)

    pairedEvent.type = 'period'
    pairedEvent.startTime = start.ts
    pairedEvent.endTime = end.ts
    pairedEvent.duration = end.ts - start.ts

    return pairedEvent


def convert_cpu_task(data):
    xmodel_info = select_trace(data, 'section', "XMODEL")
    cpu_subgraph_info = [subg for subg in xmodel_info if subg.get(
        "device", "").lower() == 'cpu']

    trace_data = select_trace_classes(data, ["cpu-task"])
    threads = {}
    pairedEvents = []
    for l in trace_data:
        event = tracepointEvent(l).toTimelineEvent()
        if event.coreType != "cpu-task":
            continue
        pid = event.pid
        threads.setdefault(pid, []).append(event)

    stack = []
    for pid in threads.keys():
        for e in threads[pid]:
            stack.append(e)
            for s in stack[::-1]:
                r = cpuTaskPaired(s, e)
                if r != None:
                    try:
                        stack.remove(e)
                        stack.remove(s)
                    except BaseException:
                        pass

                    pairedEvents.append(r)

    cpu_task_calls = statTable("CPU Tasks")
    for event in pairedEvents:
        if event.type != 'period':
            continue
        cpu_task_calls.add(event.get_info('subgraph'), event.duration)

    ret = []

    for i in cpu_task_calls.output():
        subgraph_name = i.split(',')[0]
        subg_op_desc = ""
        for s in cpu_subgraph_info:
            if s.get("subgraph_name", "") == subgraph_name:
                op_list = s.get("op_list", "").split('|')
                op_type = [op.split('@')[1]
                           for op in op_list if len(op.split('@')) == 2]

                subg_op_desc = reduce(lambda x, y: "%s|%s" % (x, y), op_type)

        subg_op_desc = reduce(lambda x, y: "%s|%s" %
                              (x, y), (set(subg_op_desc.split('|'))))
        ret.append("%s,%s,\n" % (i[0:-2], subg_op_desc))

    return ret
