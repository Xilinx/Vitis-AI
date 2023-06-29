#!/usr/bin/python3
# -*- coding: UTF-8 -*-
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


import os
import re
import csv
import sys
import string
import time
import argparse
import json
import signal
import platform
import random
import gzip
from functools import reduce

AISDK_TRACE_VER = "v0.1-190708"
TRACE_NAME_MASK = "__cln2_"
RAW_RECORD_LIMIT = 1000000
fpsCountEvent = "xilinx::ai::DpuTaskImp::run".replace("::", TRACE_NAME_MASK)
cuThreadList = ['zocl-scheduler']

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

pattAPM = re.compile(
    r'#APM\s(\d+\.\d+)\s(\d+)\s(\d+)\s(\d+)\s(\d+)\s(\d+)\s(\d+)\s(\d+)\s(\d+)\s(\d+)\s(\d+)')

hwCUPatt = re.compile(
    r"""
        .+
        cu_idx=(?P<cu_idx>\d+)
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

#global T


class traceEvent:
    def __init__(self, _eventName, _timeStamp, _info=None):
        self.event = _eventName
        self.timeStamp = float(_timeStamp)
        self.name, self.postfix = _eventName.rsplit('_', 1)
        self.info = _info

    def __str__(self):
        return "%.6f: %s\n" % (self.timeStamp, self.event)

    def isSched(self):
        return self.event.startswith('sched_')

    def isDpuHw(self):
        return self.event.lower().startswith('cu_')

    def isHw(self):
        return self.isDpuHw()

    def isFtrace(self):
        return self.event.endswith('_entry') or self.event.endswith('_exit')

    def type(self):
        if self.isSched():
            return "sched"
        if self.isDpuHw():
            return "dpuhw"
        else:
            return "ftrace"

    def dir(self):
        if self.isFtrace():
            if self.postfix == 'entry':
                return "in"
            if self.postfix == 'exit':
                return "out"
        elif self.isSched():
            """In sched case, the dir is opposite with normal events"""
            if self.postfix == 'in':
                return "out"
            if self.postfix == 'out':
                return "in"
        elif self.isDpuHw():
            if self.postfix == 'start':
                return "in"
            if self.postfix == 'done':
                return "out"

        print("Event dir err %s" % self, flush=True)

        return "unknown"


class traceRawItem:
    def __init__(self, taskComm, taskPid, cpuId, timeStamp, func, info):
        self.taskComm = taskComm.strip()
        self.taskPid = int(taskPid)
        self.cpuId = int(cpuId)
        self.timeStamp = float(timeStamp)
        self.func = func.strip()
        self.info = info.strip()
        self.infoDetail = dict()
        self.isTarget = False

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

        elif self.isDpuHw():
            if not re.match(hwCUPatt, self.info):
                print("Match hw dpu failed", flush=True)
            else:
                infoDetail = re.match(hwCUPatt, self.info).groupdict()
                self.infoDetail['cu_idx'] = int(infoDetail['cu_idx'].strip())

    def __str__(self):
        return "%.6f:%17s-%4d@[%02d]:%18s: %s" %\
               (self.timeStamp, self.taskComm, self.taskPid,
                self.cpuId, self.func, self.info)

    def isSched(self):
        return self.func.startswith('sched_switch')

    def isCu(self):
        return self.taskComm in cuThreadList or self.func.startswith("cu_")

    def isDpuHw(self):
        return self.func.startswith('cu_')

    def toTraceEvent(self):
        global T
        type = None
        if self.isSched():
            # if self.infoDetail['ss_next_comm'] == 'test_performanc':
            if self.infoDetail['ss_next_comm'] == T.targetComm:
                type = 'sched_switch_in'
            # elif self.infoDetail['ss_prev_comm'] == 'test_performanc':
            elif self.infoDetail['ss_prev_comm'] == T.targetComm:
                type = 'sched_switch_out'
            else:
                print("no used event", flush=True)
                return None
        else:
            type = self.func

        if type is None:
            return None
        return traceEvent(type, self.timeStamp, self.infoDetail)


class traceProcess:
    def __init__(self, startEvent, endEvent):
        self.start = startEvent
        self.end = endEvent
        self.name = startEvent.name
        self.startTime = startEvent.timeStamp
        self.endTime = endEvent.timeStamp
        self.duration = int((self.endTime - self.startTime) * 1000 * 1000)
        self.durationSchedOut = 0
        self.subProc = []
        self.isHw = startEvent.isHw()

    def __str__(self):
        outTime = "" if self.durationSchedOut == 0 else "scheduled out [%sus]" % format(
            self.durationSchedOut, ',')
        hwflag = "[*DPUHW]" if self.isHw else ""

        s = "%s[%s]@[%f-%f]: takes [%sus] %s" % (
            hwflag, self.name, self.startTime, self.endTime, format(self.duration, ','), outTime)
        s = s.replace(TRACE_NAME_MASK, "::")

        return s

    def __lt__(self, other):
        if self.startTime == other.startTime:
            return self.duration > other.duration
        return self.startTime < other.startTime

    def __iter__(self):
        i = iter(self.getAllSubProcs())
        return i

    def getKey(self, _taskPid):
        key = str(_taskPid) + "%s" % str(round(self.startTime * 1000 * 1000))
        return key

    def getAllSubProcs(self):
        result = [self]

        for sub in self.subProc:
            result += sub.getAllSubProcs()

        return result

    def toJsonObj(self, taskPid=0, enableStat=False, taskDuration=1):
        obj = dict()

        #obj['key'] = str(taskPid) + "%s" % str(round(self.startTime * 1000 * 1000))
        obj['key'] = self.getKey(taskPid)
        obj['title'] = self.name.replace(TRACE_NAME_MASK, "::")
        obj['icon'] = False
        obj['durnation'] = self.duration
        obj['durnationSchedOut'] = self.durationSchedOut
        obj['isHw'] = self.isHw

        if not enableStat:
            obj['startTime'] = self.startTime
            obj['endTime'] = self.endTime
        else:
            obj['percentage'] = "%.1f" % ((self.duration / taskDuration) * 100)
            pass

        obj['children'] = [s.toJsonObj(
            taskPid, enableStat, taskDuration) for s in self.subProc]
        if len(obj['children']) > 0:
            obj['folder'] = True

        return obj

    def getRelation(self, subProc):
        rel = "unknown"

        if subProc.startTime < self.startTime:
            print("Wrong order %s-%s" % (self, subProc), flush=True)
            return None

        if subProc.startTime >= self.startTime and subProc.endTime <= self.endTime:
            rel = "child"
        elif subProc.startTime >= self.endTime:
            rel = "sibling"
        else:
            rel = "overlap"
        return rel

    def insertNode(self, subProc):
        for sub in self.subProc:
            if sub.getRelation(subProc) == "child":
                sub.insertNode(subProc)
                return
        self.subProc.append(subProc)

    def getFingerPrint(self):
        fp = hash(self.name)

        for sub in self.subProc:
            fp += sub.getFingerPrint()

        return fp

    # print to console
    def printTree(self, prefix="", file=sys.stdout):
        # print("%s%s"% (prefix, self), file=file)
        prefix = prefix + "----"
        for sub in self.subProc:
            sub.printTree(prefix, file)

    # dump readable result to to memory
    def dumpTree(self):
        pass


class eventParser:
    def __init__(self, parseSched=False, parseDpuHw=True):
        self.parseSched = parseSched

    """Thread paired rule: the same in [name] and opposite in [dir]"""

    def threadPaired(self, startEve, endEve):
        """Error Checking"""
        if startEve.type() != endEve.type():
            return False

        if startEve.timeStamp > endEve.timeStamp:
            return False

        if startEve.dir() != "in" or endEve.dir() != "out":
            return False

        if startEve.isFtrace() or startEve.isSched():
            return startEve.name == endEve.name

        if startEve.isDpuHw():
            return (startEve.name == endEve.name and
                    startEve.info['cu_idx'] == endEve.info['cu_idx'])

        return False

    def parseEvents(self, _events):
        stacks = []
        parseEventResults = []

        """Filter out all sched events if there is no [parseSched]"""
        if not self.parseSched:
            events = [tmp for tmp in _events if tmp.isFtrace()
                      or tmp.isDpuHw()]
        else:
            events = _events

        for e in events:
            stacks.append(e)
            for s in stacks[::-1]:
                if self.threadPaired(s, e):
                    try:
                        stacks.remove(e)
                        stacks.remove(s)
                    except:
                        #print("parse event error", flush=True)
                        pass

                    parseEventResults.append(traceProcess(s, e))

        return sorted(parseEventResults)

    def handleSched(self, rootNode):
        i = 0
        while i < len(rootNode.subProc):
            sub = rootNode.subProc[i]
            if sub.name == "sched_switch":
                if not rootNode.isHw:
                    rootNode.duration -= sub.duration
                    rootNode.durationSchedOut += sub.duration
                sub.duration = 0
                rootNode.subProc.remove(sub)
                i -= 1
            else:
                self.handleSched(sub)
            i += 1

    def parseRoot(self, _events, debug=False):
        proc = self.parseEvents(_events)
        root = proc[0]

        for p in proc[1:]:
            root.insertNode(p)

        if debug:
            root.printTree("|-")

        if self.parseSched:
            self.handleSched(root)

        return root

    def parseCUEvents(self, _events):
        stacks = []
        parseEventResults = []

        events = _events

        for e in events:
            stacks.append(e)
            for s in stacks[::-1]:
                if self.threadPaired(s, e):
                    try:
                        stacks.remove(e)
                        stacks.remove(s)
                    except:
                        #print("parse event error", flush=True)
                        pass

                    parseEventResults.append(traceProcess(s, e))

        return sorted(parseEventResults)

    def cpuEventFilter(self, event):
        if event.isDpuHw():
            return False
        if event.isSched():
            if (event.infoDetail['ss_next_comm'] != T.targetComm or
                    event.infoDetail['ss_prev_comm'] != T.targetComm):
                return False

        return True

    def parseCPUEvents(self, _CPUEvents):
        """Filter out all hardware events"""
        events = _CPUEvents
        schedEvents = [e for e in events if e.isSched()]

        events = [e for e in events if e.isSched()]

        cpuEventsList = []
        idx = 0
        activeTime = 0

        """
        There are three states on CPU:
        ready: sched switched in, but cannot find a named function on running
        run: sched switched in and a named function on running when cpu on this status the color bar will be highlighted
        out: sched switched out
        cpuUtil : cpuUtil = total time - sched switched out time
        """
        state = "out"
        while (idx < len(events) - 1):
            s = events[idx]
            e = events[idx+1]

            "Get a ramdom base color"
            #color = hash(str(s.taskPid)) % (256 * 256 * 256)
            color = (hash(str(s.infoDetail['ss_next_pid'])) >> 2) % (
                256 * 256 * 256)

            # if (s.infoDetail['ss_next_comm'] == T.targetComm):
            #    activeTime += (e.timeStamp - s.timeStamp)
            # else:
            #    color = color | 0xe0e0e0

            if s.isSched():
                if (s.infoDetail['ss_next_comm'] == T.targetComm):
                    state = "ready"
                else:
                    state = "out"
            else:
                if s.taskComm == T.targetComm:
                    state = "run"
                else:
                    state = "out"

            if state == "ready" or state == "run":
                activeTime += (e.timeStamp - s.timeStamp)
            else:
                # if state != "run":
                color = color | 0xe0e0e0

            cpuEventsList.append(
                [s.infoDetail['ss_next_comm'], s.infoDetail['ss_next_pid'], s.timeStamp, e.timeStamp, "#%06x" % color])

            idx = idx+1

        cpuUtil = (activeTime / (T.endTime - T.startTime))

        return cpuEventsList, cpuUtil


class traceThread:
    events = []
    traces = []
    parsed = False
    isCuThread = False

    def __init__(self, _taskComm, _taskPid, _opt, debug=False):
        self.taskComm = _taskComm
        self.taskPid = _taskPid
        self.traces = []
        self.events = []
        self.parsed = False
        self.dump = debug
        self.rootNode = None
        self.options = _opt

    def addItemToThread(self, item):
        self.traces.append(item)
        if item.isCu():
            self.isCuThread = True

    """Add a force end event to the end of records of this thread"""

    def addThreadFlags(self):
        last = self.traces[-1]
        first = self.traces[0]
        forceFirst = traceRawItem(
            "aisdk", "99999", 99, first.timeStamp - 0.00001, "Thread-%d_entry" % self.taskPid, " ")
        forceLast = traceRawItem(
            "aisdk", "99999", 99, last.timeStamp + 0.00001, "Thread-%d_exit" % self.taskPid, " ")
        self.traces.insert(0, forceFirst)
        self.traces.append(forceLast)

    def getTaskPid(self):
        return self.taskPid

    def parse(self):
        print('Analyzing thread: %s-%d' %
              (self.taskComm, self.taskPid), flush=True)

        """Step.1 parse every raw event for this thread """
        for item in self.traces:
            if item.isSched():
                if item.infoDetail['ss_prev_comm'] == self.taskComm and \
                   item.infoDetail['ss_prev_pid'] == self.taskPid:
                    self.events.append(traceEvent(
                        'sched_switch_out', item.timeStamp, item.infoDetail))
                elif item.infoDetail['ss_next_comm'] == self.taskComm and \
                        item.infoDetail['ss_next_pid'] == self.taskPid:
                    self.events.append(traceEvent(
                        'sched_switch_in', item.timeStamp, item.infoDetail))
                continue
            else:
                self.events.append(traceEvent(
                    item.func, item.timeStamp, item.infoDetail))

        """Step.2 divided events into each images"""
        # print(*self.events)
        print("---------------------------------", flush=True)
        p = eventParser(False)
        if self.isCuThread:
            self.rootNode = p.parseCUEvents(self.events)
        else:
            self.rootNode = p.parseRoot(self.events, debug=False)
        self.parsed = True

    def getId(self):
        return "%s-%d" % (self.taskComm, self.taskPid)

    def getTimeline(self):
        if not self.parsed:
            #print("[%s]:Must be parsed at first" % self.getId())
            return None

        title = []
        time = []
        ev = []

        title.append("%s-%d" % (self.taskComm, self.taskPid))

        for e in self.events:
            time.append(e.timeStamp)
            ev.append(e.event)

        return [title, time, ev]


class APMRecord:
    def __init__(self, rec):
        self.time = float(rec[0])
        self.data = []

        for d in rec[1:]:
            self.data.append(int(d)*1000/1024/1024.0)

        self.total = sum(self.data)

    def __str__(self):
        return "Time: %.7f  %.2f" % (self.time, self.total)


class traceTable:
    def __init__(self, _targetComm, _tracePatt, _opt, _title=""):
        self.title = _title
        self.targetComm = _targetComm
        self.targetPid = -1
        self.alivePidTable = []
        self.tracePatt = _tracePatt
        self.threads = []
        self.cpus = [[], [], [], [], [], [], [], [], [], []]
        self.options = _opt
        self.startTime = 1000000000.0
        self.endTime = 0.0
        self.firstRunTime = 9999999999.9
        self.lastRunTime = 0
        self.runCnt = 0
        self.cuUtil = []
        self.cpuUtil = []
        self.cuMap = {}

    def belongsToTarget(self, _pid, _ts):
        return True

    def addRawItems(self,  _items):
        items = _items[0: RAW_RECORD_LIMIT]

        for l in items:
            if not re.match(self.tracePatt, l.strip()):
                continue

            tmp = re.match(self.tracePatt, l.strip()).groups()
            item = traceRawItem(*tmp)

            self.findTarget(item)
            # print(f"Alive Threads' PID: {self.alivePidTable}")

            """Update Time"""
            if item.timeStamp < self.startTime:
                self.startTime = item.timeStamp
            if item.timeStamp > self.endTime:
                self.endTime = item.timeStamp

            #"""Get first and last event time"""
            # if items.index(l) == 0:
            #    self.startTimeStamp = item.timeStamp
            # if items.index(l) == len(items) - 1:
            #    self.endTimeStamp = item.timeStamp

            """Add item to threads"""
            for thread in self.getThread(item):
                if thread is not None:
                    thread.addItemToThread(item)

            """Add item to CPUs"""
            if not item.isCu():
                self.cpus[item.cpuId].append(item)

        for t in self.threads:
            # print(f"{t.taskComm}, {t.taskPid}")
            t.addThreadFlags()

    def updateTs(self, _item):
        pass

    def findTarget(self, _item):
        if not _item.func.startswith('sched_process_'):
            return

        if self.targetComm != None:
            return

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
                self.targetComm = spInfo['filename'].split('/')[-1][:15]
                self.targetPid = pid
                self.alivePidTable.append(pid)
                print("Find Comm %s" % self.targetComm, flush=True)

        elif spType == 'fork':
            forkInfoPatt = re.compile(
                r"""
                comm=(?P<comm>.+)\s
                pid=(?P<pid>\d+)\s
                child_comm=(?P<child_comm>.+)\s
                child_pid=(?P<child_pid>\d+)
                """,
                re.X | re.M)
            spInfo = re.match(forkInfoPatt, _item.info)
            if (spInfo is not None):
                c_pid = int(spInfo['child_pid'])
                if (c_pid in self.alivePidTable):
                    raise ("PID ERROR")
                comm = spInfo['comm']
                print(f"sched_fork: {comm}:{self.targetComm}")
                if self.targetComm == comm:
                    self.alivePidTable.append(c_pid)
                # print(self.alivePidTable)
                #print("[%f]sched_process_%s,pid %s -> [forked] -> %s" % (spTs, spType, spInfo['pid'], spInfo['child_pid']))

        elif spType == 'exit':
            exitInfoPatt = re.compile(
                r"""
                comm=(?P<comm>.+)\s
                pid=(?P<pid>\d+)\s
                prio=(?P<prio>\d+)
                """,
                re.X | re.M)
            spInfo = re.match(exitInfoPatt, _item.info)
            if (spInfo is not None):
                pid = int(spInfo['pid'])
                comm = spInfo['comm']
                print(f"sched_exit: {comm}:{self.targetComm}")
                if self.targetComm == comm:
                    self.alivePidTable.remove(pid)
                    #print("[%f]sched_process_%s, pid %s" % (spTs, spType, spInfo['pid']))
        else:
            pass

    def parseAll(self, opts=None):
        if self.targetComm is None or self.targetComm == "":
            raise RuntimeError("[targetComm] is None")

        for t in self.threads:
            # BUG: should use self.elongsToTarget here
            if t.taskComm.startswith(self.targetComm) or t.isCuThread:
                t.parse()

        self.threads = [t for t in self.threads if t.rootNode is not None]

    """addThread() rerurns the thread(s) this item belongs to, if not matched return none"""

    def getThread(self, item):
        switchOut = None
        switchIn = None

        """sched_switch records will be assigned for 2 times, one for switched in thread, one for switched out thread"""
        if item.isSched():
            for t in self.threads:
                if t.taskPid == item.infoDetail['ss_prev_pid']:
                    switchOut = t
                if t.taskPid == item.infoDetail['ss_next_pid']:
                    switchIn = t
            if not switchIn:
                switchIn = traceThread(
                    item.infoDetail['ss_next_comm'], item.infoDetail['ss_next_pid'], self.options)
                self.threads.append(switchIn)
            if not switchOut:
                switchOut = traceThread(
                    item.infoDetail['ss_prev_comm'], item.infoDetail['ss_prev_pid'], self.options)
                self.threads.append(switchOut)
        else:
            for t in self.threads:
                if t.taskPid == item.taskPid and t.taskComm == item.taskComm:
                    switchIn = t
            if not switchIn:
                self.threads.append(traceThread(
                    item.taskComm, item.taskPid, self.options))
                switchIn = self.threads[-1]

        return switchIn, switchOut

    def printImgView(self, file=sys.stdout):
        for t in self.threads:
            if not t.parsed:
                continue

            #print("[%s ID: %s]" % (prefix, img.getId()), file=file)
            t.rootNode.printTree(prefix='|--', file=file)
            t.rootNode.dumpTree()
            print("", file=file, flush=True)

    def saveImgViewTXT(self, file='image.txt'):
        with open(file, 'w') as f:
            print("AI SDK trace output file name: %s" % file, flush=True)
            self.printImgView(f)

    def getCPUJson(self):
        retData = {}
        p = eventParser(False)

        for i in range(0, len(self.cpus)):
            cpuData, cpuUtil = p.parseCPUEvents(self.cpus[i])
            if cpuUtil > 0:
                coreID = "%d" % i
                util = "%f" % cpuUtil
                T.cpuUtil.append({coreID: util})

            if (len(cpuData) > 0):
                retData['CPU-%d' % i] = cpuData

        return retData

    def getStatisticJson(self):
        stat = {}

        for t in self.threads:
            if t.isCuThread:
                continue
            if (len(t.rootNode.subProc) <= 1):
                continue

            """Filter out the last one"""
            for sub in t.rootNode.subProc[:-1]:
                if sub.name not in stat.keys():
                    stat[sub.name] = []
                stat[sub.name].append(sub)

        def mf(x, y): return dict(x, **y)
        utils = reduce(mf, self.cpuUtil + self.cuUtil)
        #utils["CPU"] = self.cpuUtil
        return {'cpu': 'utilization', 'data': utils}

    def getCUJson(self):
        cuProcs = []
        cuSet = set()
        retData = dict()

        for th in self.threads:
            if not th.isCuThread:
                continue
            for proc in th.rootNode:
                if proc.name == 'cu':
                    cuProcs.append(proc)

        for proc in cuProcs:
            idx = int(proc.start.info['cu_idx'])
            cuSet.add(idx)
            #coreID = "CU-%d" % (idx + list(cuSet).index(idx))
            coreID = "CU-%s" % T.cuMap[hex(idx)]
            startTs = proc.startTime
            endTs = proc.endTime

            if coreID not in retData.keys():
                retData[coreID] = []
            retData[coreID].append((startTs, endTs))

        for id in retData.keys():
            retData[id].sort(key=lambda x: x[0])
            totalDelta = retData[id][-1][1] - retData[id][0][0]
            runTime = 0
            for i in retData[id]:
                runTime += i[1] - i[0]
            util = "%.1f%%" % (runTime / totalDelta * 100)
            latency = runTime * 1000 / len(retData[id])
            util += "|Task Lat: %.1fms" % latency
            """Format CU-ID: [CU-2147745792-->CU@0x80040000] """
            if (id.startswith('CU-')):
                """Last 4-bit represent CU index"""
                addr = T.cuMap[id.split('-')[1]]
                #addr = hex((int(id.split('-')[1]) >> 4) << 4)
                #idx = int(id.split('-')[1]) & 0xf
                idx = T.cuMap[addr]
                id = "CU-%s@%s" % (idx, addr)

            self.cuUtil.append({id: util})

        return retData

    def getThreadJson(self):
        retData = dict()

        for t in self.threads:
            if (len(t.rootNode.subProc) == 0):
                continue
            retData['Thread-%d' % t.taskPid] = []
            for s in t.rootNode.subProc:
                #print (s.startTime, s.endTime)
                retData['Thread-%d' % t.taskPid].append(
                    (s.startTime, s.endTime, s.name.replace(TRACE_NAME_MASK, "::")))

        return retData


def sched_parser_main(src):

    if len(src) == 0:
        return {}

    options = {'thread_view': True, 'trace_sched_switch': False}
    data = {}

    global T
    T = traceTable(None, pattFtrace, options)
    T.addRawItems(src)
    T.parseAll()

    #targetFile = sys.argv[2]

    # with open(targetFile, "w+") as f:
    # data.update(T.getCUJson())

    """Get CPU data"""
    data.update(T.getCPUJson())

    """Get statstic info"""
    # data.update(T.getStatisticJson())
    stat = T.getStatisticJson()

    # print(stat)
    # utilization
    #f.write(json.dumps(stat, indent=1, sort_keys=True))
    ret = {"cpu": {"utilization": stat['data']}}

    return ret


if __name__ == '__main__':
    f = open(sys.argv[1], 'rt')

    source = f.readlines()
    sched_parser_main(source)
    f.close()
