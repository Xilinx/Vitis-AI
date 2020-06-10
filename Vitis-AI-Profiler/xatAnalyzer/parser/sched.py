
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

import parser.parserBase
import parser.ftraceUtil
import parser.timelineUtil

from parser.timelineUtil import *

IdleColor = "#fffbf0"
CPUs = []


class schedParser(parser.parserBase.Parser):
    def __init__(self):
        super().__init__('sched')
        self.schedProcess = []
        self.targetComm = None

    def formater(self, fevent: parser.ftraceUtil.ftraceEvent) -> vaiTimelineEvent:

        prevComm = fevent.infoDetail['ss_prev_comm']
        prevPid = fevent.infoDetail['ss_prev_pid']
        nextComm = fevent.infoDetail['ss_next_comm']
        nextPid = fevent.infoDetail['ss_next_pid']

        event_e = vaiTimelineEvent(fevent.timeStamp, prevPid, "done", "CPU", fevent.cpuId, prevComm)
        event_s = vaiTimelineEvent(fevent.timeStamp, nextPid, "start", "CPU", fevent.cpuId, nextComm)

        return [event_e, event_s]

    def findTarget(self):
        targetComm = ""
        targetPid = 0

        for _item in self.schedProcess:
            spTs = _item.timeStamp
            spType = _item.func.split('_process_')[1]
            if spType == 'exec':
                execInfoPatt = re.compile(
                    r"""
                    filename=(?P<filename>.+)\s
                    pid=(?P<pid>\d+)\s
                    old_pid=(?P<old_pid>\d+)
                    """,
                    re.X | re.M)
                spInfo = re.match(execInfoPatt, _item.info)
                if (spInfo is not None):
                    pid = int(spInfo['pid'])
                    targetComm = spInfo['filename'].split('/')[-1][:15]
                    targetPid = pid
                    # self.alivePidTable.append(pid)
                    print("Find Comm %s" % targetComm, flush=True)

        self.targetComm = targetComm
        return targetComm, targetPid

        # elif spType == 'fork':
        #    forkInfoPatt = re.compile(
        #        r"""
        #        comm=(?P<comm>.+)\s
        #        pid=(?P<pid>\d+)\s
        #        child_comm=(?P<child_comm>.+)\s
        #        child_pid=(?P<child_pid>\d+)
        #        """,
        #        re.X | re.M)
        #    spInfo = re.match(forkInfoPatt, _item.info)
        #    if (spInfo is not None):
        #        cpid = int(spInfo['child_pid'])
        #        if (cpid in self.alivePidTable):
        #            raise("PID ERROR")
        #        self.alivePidTable.append(cpid)
        #        #print(self.alivePidTable)
        #        #print("[%f]sched_process_%s,pid %s -> [forked] -> %s" % (spTs, spType, spInfo['pid'], spInfo['child_pid']))

        # elif spType == 'exit':
        #    exitInfoPatt = re.compile(
        #        r"""
        #        comm=(?P<comm>.+)\s
        #        pid=(?P<pid>\d+)\s
        #        prio=(?P<prio>\d+)
        #        """,
        #        re.X | re.M)
        #    spInfo = re.match(exitInfoPatt, _item.info)
        #    if (spInfo is not None):
        #        pid = int(spInfo['pid'])
        #        self.alivePidTable.remove(pid)
        #        #print("[%f]sched_process_%s, pid %s" % (spTs, spType, spInfo['pid']))

        # else:
        #    pass

    def parse(self, data, options):
        retData = {}

        """Support at most 16 cores"""
        global CPUs
        CPUs = createTimelines('CPU', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], options)

        for l in data:
            fevent = parser.ftraceUtil.parse(l, options)
            if fevent is None:
                continue
            if fevent.func != "sched_switch":
                self.schedProcess.append(fevent)
                continue
            if self.targetComm is None:
                comm, pid = self.findTarget()
                options.setdefault("targetComm", comm)
            else:
                timelineEvent = self.formater(fevent)
                for e in timelineEvent:
                    if timelineEvent is not None:
                        CPUs[fevent.cpuId].add(e)

        def threadColor(i: timelineEvent):
            if i.info != self.targetComm:
                return IdleColor
            else:
                return None

        def eventFilter(e: timelineEvent):
            if e.info != self.targetComm:
                return False
            else:
                return True

        for cpu in CPUs:
            if cpu.len() <= 0:
                continue

            cpuData = cpu.toJson(prefix=self.targetComm, _color=threadColor)

            coreId = CPUs.index(cpu)
            retData.update({"TIMELINE-CPU_%d" % coreId: cpuData})
            cpu.getUtil(_filter=eventFilter)

        return retData


parser.parserBase.register(schedParser())
