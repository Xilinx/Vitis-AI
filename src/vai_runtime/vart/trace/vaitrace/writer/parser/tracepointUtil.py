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
import re

TraceEventType = [
    "VAI_EVENT_HOST_START",
    "VAI_EVENT_HOST_END",
    "VAI_EVENT_INFO",

    "VAI_EVENT_DEVICE_START",
    "VAI_EVENT_DEVICE_END",
    "VAI_EVENT_MARKER",
    "VAI_EVENT_COUNTER"
]


"""trace_select(data, "classname", "dpu-runner") -> [...]"""


def select_trace(trace_data, sel_by, val) -> []:
    if trace_data is None:
        return []
    return [d for d in trace_data if d.get(sel_by, None) == val]


"""trace_select(data, "classname", "dpu-runner") -> [...]"""


def select_trace_classes(trace_data, classes) -> []:
    if type(classes) != list:
        raise TypeError
    if trace_data is None:
        return []
    return [d for d in trace_data if d.get("classname", None) in classes]


class tracepointEvent:
    def __init__(self, l: dict):
        """event_state: 1: start, 0: end, -1: not defined"""
        self.event_state = int(l.pop("event_state", -1))
        self.pid = int(l.pop("pid", 0))
        self.cpu_id = int(l.pop("cpu_id", -1))
        self.t_class = l.pop("classname", "null")
        self.time_stamp = float(l.pop("ts", 0.0))
        self.payload = l

    def toTimelineEvent(self):
        if self.event_state == 1:
            et = "start"
        elif self.event_state == 0:
            et = "done"
        else:
            et = "marker"

        ct = self.t_class
        cid = self.payload.get("device_core_idx", -1)

        return vaiTimelineEvent(self.time_stamp, self.pid, et, ct, cid, self.payload)


def parse(l, options) -> tracepointEvent:
    """Selet trace clock"""
    tc = options.get('traceClock', None)

    return tracepointEvent(l.split(" ", 6))
