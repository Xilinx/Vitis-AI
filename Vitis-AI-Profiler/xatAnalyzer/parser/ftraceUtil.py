
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

import re

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
        (?P<func>\w+)\:
        (?P<info>.*)
        """,
    re.X | re.M
)

"""for x86, trace_clock = x86-tsc"""
pattFtraceTSC = re.compile(
    r"""
        (?P<taskComm>.+)\-
        (?P<taskPid>\d{1,})\s+\[
        (?P<cpuID>\d{3})\].{4,}\s
        (?P<timeStamp>\d+)\:\s+
        (?P<func>\w+)\:
        (?P<info>.*)
        """,
    re.X | re.M
)

pattSchedSwitch = re.compile(
    r"""
        prev_comm\=(?P<prev_comm>.+)
        prev_pid\=(?P<prev_pid>.+)
        prev_prio\=(?P<prev_prio>.+)
        prev_state\=(?P<prev_state>.+)==>.+
        next_comm\=(?P<next_comm>.+)
        next_pid\=(?P<next_pid>.+)
        next_prio\=(?P<next_prio>.+)
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

        #self.type = None
        #self.dir = None

        if self.isSched():
            if not re.match(pattSchedSwitch, self.info):
                print("Match sched_switch failed", flush=True)
            else:
                infoDetail = re.match(pattSchedSwitch, self.info).groupdict()
                self.infoDetail['ss_prev_comm'] = infoDetail['prev_comm'].strip()
                self.infoDetail['ss_prev_pid'] = int(infoDetail['prev_pid'])
                self.infoDetail['ss_prev_prio'] = int(infoDetail['prev_prio'])
                self.infoDetail['ss_prev_state'] = infoDetail['prev_state'].strip()
                self.infoDetail['ss_next_comm'] = infoDetail['next_comm'].strip()
                self.infoDetail['ss_next_pid'] = int(infoDetail['next_pid'])
                self.infoDetail['ss_next_prio'] = int(infoDetail['next_prio'])

        elif self.isCu():
            if not re.match(hwCUPatt, self.info):
                print("Match hw dpu failed", flush=True)
            else:
                infoDetail = re.match(hwCUPatt, self.info).groupdict()
                self.infoDetail['cu_idx'] = int(infoDetail['cu_idx'].strip())

    def __str__(self):
        return "%.6f:%17s-%4d@[%02d]:%18s: %s" %\
               (self.timeStamp, self.taskComm, self.taskPid, self.cpuId, self.func, self.info)

    def isSched(self):
        return self.func.startswith('sched_switch')

    def isCu(self):
        return self.func.startswith("cu_")

    # def toTraceEvent(self):
    #    global T
    #    type = None
    #    if self.isSched():
    #        if self.infoDetail['ss_next_comm'] == T.targetComm:
    #            type = 'sched_switch_in'
    #        elif self.infoDetail['ss_prev_comm'] == T.targetComm:
    #            type = 'sched_switch_out'
    #        else:
    #            print("no used event", flush=True)
    #            return None
    #    else:
    #        type = self.func
    #
    #    if type is None:
    #        return None
    #    return traceEvent(type, self.timeStamp, self.infoDetail)


def parse(l, options):
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
    return(ftraceEvent(*tmp))
