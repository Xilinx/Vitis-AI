
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

TRACE_NAME_MASK = "__cln2_"


def isFpsCountEvent(eventName: str, type: str):
    fpsHLCountEvents = [
        "vitis::ai::ClassificationImp::run",
        "vitis::ai::DetectImp::run",
        "vitis::ai::FaceLandmarkImp::run",
        "vitis::ai::MultiTaskImp::run_8UC1",
        "vitis::ai::MultiTaskImp::run_8UC3",
        "vitis::ai::OpenPoseImp::run",
        "vitis::ai::PoseDetectImp::run",
        "vitis::ai::RefineDetImp::run",
        "vitis::ai::ReidImp::run",
        "vitis::ai::RoadLineImp::run",
        "vitis::ai::SSDImp::run",
        "vitis::ai::SegmentationImp::run_8UC1",
        "vitis::ai::SegmentationImp::run_8UC3",
        "vitis::ai::TFSSDImp::run",
        "vitis::ai::YOLOv2Imp::run",
        "vitis::ai::YOLOv3Imp::run",
        "vitis::ai::FaceFeatureImp::run",
        "vitis::ai::MedicalSegmentationImp::run",
        "vitis::ai::PlateDetectImp::run",
        "vitis::ai::PlateNumImp::run",
        "vitis::ai::PlateRecogImp::run"
    ]

    fpsLLCountEvents = [
        "XrtCu::run"
    ]

    "By default conut for high level events"
    eventList = fpsHLCountEvents
    if type == "lowlevel":
        eventList = fpsLLCountEvents

    for f in eventList:
        if eventName.find(f.replace("::", TRACE_NAME_MASK)) >= 0:
            return True

    return False


class traceEvent:
    def __init__(self, _eventName, _timeStamp, _info=None):
        self.event = _eventName
        self.timeStamp = float(_timeStamp)
        self.name, self.postfix = _eventName.rsplit('_', 1)
        self.info = _info

    def __str__(self):
        return "%.6f: %s\n" % (self.timeStamp, self.event)

    def isFtrace(self):
        return self.event.endswith('_entry') or self.event.endswith('_exit')

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

    def __str__(self):
        return "%.6f:%17s-%4d@[%02d]:%18s: %s" %\
               (self.timeStamp, self.taskComm, self.taskPid, self.cpuId, self.func, self.info)

    def toTraceEvent(self):
        global T
        type = self.func

        return traceEvent(type, self.timeStamp, self.infoDetail)


class traceProcess:
    def __init__(self, startEvent, endEvent):
        self.start = startEvent
        self.end = endEvent
        """Should handle c++ overloading case"""
        if startEvent.name.split('_')[-1].isdigit():
            self.name = startEvent.name.rsplit('_', 1)[0]
        else:
            self.name = startEvent.name
        self.startTime = startEvent.timeStamp
        self.endTime = endEvent.timeStamp
        self.duration = int((self.endTime - self.startTime) * 1000 * 1000)
        self.durationSchedOut = 0
        self.subProc = []
        self.isHw = False

    def __str__(self):
        outTime = "" if self.durationSchedOut == 0 else "scheduled out [%sus]" % format(self.durationSchedOut, ',')
        hwflag = "[*DPUHW]" if self.isHw else ""

        s = "%s[%s]@[%f-%f]: takes [%sus] %s" % (hwflag, self.name, self.startTime, self.endTime, format(self.duration, ','), outTime)
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

        obj['children'] = [s.toJsonObj(taskPid, enableStat, taskDuration) for s in self.subProc]
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


class eventParser:
    def __init__(self, parseSched=False, parseDpuHw=True):
        pass
    """Thread paired rule: the same in [name] and opposite in [dir]"""

    def threadPaired(self, startEve, endEve):
        """Error Checking"""
        if startEve.timeStamp > endEve.timeStamp:
            return False

        if startEve.dir() != "in" or endEve.dir() != "out":
            return False

        if startEve.isFtrace() or startEve.isSched():
            return startEve.name == endEve.name

        return False

    def parseEvents(self, events):
        stacks = []
        parseEventResults = []

        for e in events:
            stacks.append(e)
            for s in stacks[::-1]:
                if self.threadPaired(s, e):
                    try:
                        stacks.remove(e)
                        stacks.remove(s)
                    except BaseException:
                        #print("parse event error", flush=True)
                        pass

                    parseEventResults.append(traceProcess(s, e))

        return sorted(parseEventResults)

    def parseRoot(self, _events, debug=False):
        proc = self.parseEvents(_events)
        root = proc[0]

        for p in proc[1:]:
            root.insertNode(p)

        return root


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

    """Add a force end event to the end of records of this thread"""

    def addThreadFlags(self):
        last = self.traces[-1]
        first = self.traces[0]
        forceFirst = traceRawItem("aisdk", "99999", 99, first.timeStamp - 0.00001, "Thread-%d_entry" % self.taskPid, " ")
        forceLast = traceRawItem("aisdk", "99999", 99, last.timeStamp + 0.00001, "Thread-%d_exit" % self.taskPid, " ")
        self.traces.insert(0, forceFirst)
        self.traces.append(forceLast)

    def getTaskPid(self):
        return self.taskPid

    def parse(self):
        print('Analyzing thread: %s-%d' % (self.taskComm, self.taskPid), flush=True)

        """Step.1 parse every raw event for this thread """
        for item in self.traces:
            if False:
                if item.infoDetail['ss_prev_comm'] == self.taskComm and \
                   item.infoDetail['ss_prev_pid'] == self.taskPid:
                    self.events.append(traceEvent('sched_switch_out', item.timeStamp, item.infoDetail))
                elif item.infoDetail['ss_next_comm'] == self.taskComm and \
                        item.infoDetail['ss_next_pid'] == self.taskPid:
                    self.events.append(traceEvent('sched_switch_in', item.timeStamp, item.infoDetail))
                continue
            else:
                self.events.append(traceEvent(item.func, item.timeStamp, item.infoDetail))

        """Step.2 divided events into each images"""
        # print(*self.events)
        print("---------------------------------", flush=True)
        p = eventParser(False)
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


class traceTable:
    def __init__(self, _targetComm, _opt, _title=""):
        self.title = _title
        self.targetComm = _targetComm
        self.targetPid = -1
        self.alivePidTable = []
        self.threads = []
        self.options = _opt
        self.startTime = 1000000000.0
        self.endTime = 0.0
        self.firstRunTime = 9999999999.9
        self.lastRunTime = 0
        self.runCnt = 0
        self.cpuUtil = []

        self.transTs = lambda x: round(float(x), 6)
        if self.options.get('traceClock', "") == 'x86-tsc':
            tsc_hz = int(self.options.get('x86_tsc_khz') * 1000)
            self.transTs = lambda x: round((float(x) / tsc_hz), 6)

    def addRawItems(self, items):
        for l in items:
            fevent = parser.ftraceUtil.parse(l, self.options)

            if fevent is None:
                continue
            fevent.timeStamp = self.transTs(fevent.timeStamp)

            item = traceRawItem(fevent.taskComm, fevent.taskPid, fevent.cpuId, fevent.timeStamp, fevent.func, fevent.info)

            """Add item to threads"""
            for thread in self.getThread(item):
                if thread is not None:
                    thread.addItemToThread(item)

        for t in self.threads:
            t.addThreadFlags()

    def parseAll(self, opts=None):
        if self.targetComm is None or self.targetComm == "":
            raise RuntimeError("[targetComm] is None")

        for t in self.threads:
            # BUG: should use self.blongsToTarget here
            # if t.taskComm.startswith(self.targetComm) or t.isCuThread:
            t.parse()

        self.threads = [t for t in self.threads if t.rootNode is not None]

    """addThread() rerurns the thread(s) this item belongs to, if not matched return none"""

    def getThread(self, item):
        switchOut = None
        switchIn = None

        """sched_switch records will be assigned for 2 times, one for switched in thread, one for switched out thread"""
        if False:
            for t in self.threads:
                if t.taskPid == item.infoDetail['ss_prev_pid']:
                    switchOut = t
                if t.taskPid == item.infoDetail['ss_next_pid']:
                    switchIn = t
            if not switchIn:
                switchIn = traceThread(item.infoDetail['ss_next_comm'], item.infoDetail['ss_next_pid'], self.options)
                self.threads.append(switchIn)
            if not switchOut:
                switchOut = traceThread(item.infoDetail['ss_prev_comm'], item.infoDetail['ss_prev_pid'], self.options)
                self.threads.append(switchOut)
        else:
            for t in self.threads:
                if t.taskPid == item.taskPid and t.taskComm == item.taskComm:
                    switchIn = t
            if not switchIn:
                self.threads.append(traceThread(item.taskComm, item.taskPid, self.options))
                switchIn = self.threads[-1]

        return switchIn, switchOut

    def getCallGraphJson(self):
        retData = dict()
        retData['CG-0'] = []

        for t in self.threads:
            if (len(t.rootNode.subProc) is 0):
                continue
            retData['CG-0'].append(t.rootNode.toJsonObj(t.taskPid))

        return retData


class functionParser(parser.parserBase.Parser):
    def __init__(self):
        super().__init__('function')

    def parse(self, data, options):
        source = [l.strip() for l in data]

        targetComm = options.get("targetComm")
        T = traceTable(targetComm, options)
        T.addRawItems(source)
        T.parseAll()

        dataCG = T.getCallGraphJson()

        return dataCG


parser.parserBase.register(functionParser())
