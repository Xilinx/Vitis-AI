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

from writer.vtf.timelineUtil import *
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


class tracepointEvent:
    def __init__(self, l):
        _eventType, _pid, _cpuId, _tag, _devId, _timeStamp, * \
            _info = l.split(" ", 6)
        self.eventType = _eventType
        self.pid = int(_pid)
        self.cpuId = int(_cpuId)
        self.tag = _tag
        self.devId = int(_devId)
        if _timeStamp.find('.') > 0:
            # boot time
            self.timeStamp = float(_timeStamp)
        else:
            # tsc time
            self.timeStamp = int(_timeStamp)
        if len(_info) == 0:
            self.info = ""
        else:
            self.info = _info[0]

    def toTimelineEvent(self):
        if self.eventType.find("START") >= 0:
            et = "start"
        elif self.eventType.find("END") >= 0:
            et = "done"
        else:
            et = "marker"

        if self.eventType.find("HOST") >= 0:
            ct = "CPU"
            cid = self.cpuId
        else:
            ct = self.tag
            cid = self.devId

        return vaiTimelineEvent(self.timeStamp, self.pid, et, ct, cid, self.info.strip())


def parse(l, options) -> tracepointEvent:
    """Selet trace clock"""
    tc = options.get('traceClock', None)

    return tracepointEvent(l.split(" ", 6))
