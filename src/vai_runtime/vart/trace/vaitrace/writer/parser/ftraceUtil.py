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

import re
from writer.parser.timelineUtil import *

hwCUPatt = re.compile(
    r"""
        .+
        cu_idx=(?P<cu_idx>\d+)
        """,
    re.X | re.M
)
"""for arm, trace_clock = boot"""
pattFtrace = re.compile(
    r"""
        (?P<taskComm>.+)\-
        (?P<taskPid>\d{1,})\s+\[
        (?P<cpuID>\d{3})\].{4,}\s
        (?P<timeStamp>\d+\.\d{6})\:\s+
        (?P<func>[\w:]+)\:\s+
        (?P<info>.*)
        """,
    re.X | re.M
)


class ftraceEvent:
    def __init__(self, taskComm, taskPid, cpuId, timeStamp, func, info):
        self.taskComm = taskComm.strip()
        self.taskPid = int(taskPid)
        self.cpuId = int(cpuId)
        self.timeStamp = float(timeStamp)
        self.func = func.strip()
        self.info = info.strip()
        self.infoDetail = dict()
        self.isTarget = False
        self.dir = None

    def __str__(self):
        return "%.6f:%17s-%4d@[%02d]:%18s: %s" %\
               (self.timeStamp, self.taskComm, self.taskPid,
                self.cpuId, self.func, self.info)

    def toTimelineEvent(self):
        if self.func.endswith("_entry"):
            et = "start"
        elif self.func.endswith("_exit"):
            et = "done"
        else:
            et = "marker"

        ct = "CPU"
        cid = self.cpuId

        func = self.func.replace("_entry", "").replace("_exit", "")
        return vaiTimelineEvent(self.timeStamp, self.taskPid, et, ct, cid, func)


def ftraceParse(l, options):
    patt = pattFtrace

    """Selet trace clock"""
    tc = options.get('traceClock', None)
    if tc == 'x86-tsc':
        patt = pattFtraceTSC
    tmp = re.match(patt, l.strip())

    """Not matched"""
    if tmp is None:
        return None

    tmp = tmp.groups()
    return (ftraceEvent(*tmp))
