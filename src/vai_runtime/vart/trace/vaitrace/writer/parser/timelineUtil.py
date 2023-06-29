
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

from enum import Enum, auto


class vaiTimelineEventState(Enum):
    START = auto()
    DONE = auto()
    PERIOD = auto()
    MARKER = auto()


class vaiTimelineEvent:
    def __str__(self):
        if self.type == 'period':
            t = "Duration:%10.8f" % self.duration
        else:
            t = "TimeStamp:%10.8f" % self.ts

        return "Core %10s-%02d: Type:%6s, pid:%06d ts:%f" % \
            (self.coreType, self.coreId, self.type, self.pid, self.ts) + " " + t

    """
    Four types: start, end (the same with done), marker, and period
    """

    def __init__(self, _ts, _pid, _eventType, _coreType, _coreId, _info=None):
        self.ts = float(_ts)
        self.pid = int(_pid)
        self.type = _eventType
        self.coreType = _coreType
        self.coreId = int(_coreId)
        self.startTime = 0
        self.endTime = 0
        self.duration = 0
        self.info = _info

        if self.type == 'start' or self.type == 'done':
            self.code = _info

    def paired(self, end):
        if not (self.type == 'start' and end.type == 'done'):
            return None
        if end.ts < self.ts:
            return None

        self.type = 'period'
        self.startTime = self.ts
        self.endTime = end.ts
        self.duration = self.endTime - self.startTime

    def get_info(self, key: str):
        return self.info.get(key, None)


def dpuEventPaired(start: vaiTimelineEvent, end: vaiTimelineEvent):
    if not (start.type == 'start' and end.type == 'done'):
        return None
    if end.ts < start.ts:
        return None
    if start.pid != end.pid:
        return None
    if start.coreId != end.coreId:
        return None

    start.type = 'period'
    start.startTime = start.ts
    start.endTime = end.ts
    start.duration = start.endTime - start.startTime
    start.hwcounter = int(end.info.get("hwconuter",0))

    return start


class vaiTimeline:
    def __init__(self, _coreType, _coreId: str, options: dict, _timeout: int = 0):
        self.coreType = _coreType
        self.coreId = _coreId
        self.dpuCore = None
        self.lastTraceItem = None
        self.timeline = []
        self.eventStack = []
        self.timeout = _timeout
        self.transTs = lambda x: round(float(x), 6)
        if options.get('traceClock', "") == 'x86-tsc':
            tsc_hz = int(options.get('x86_tsc_khz') * 1000)
            self.transTs = lambda x: round((float(x) / tsc_hz), 6)

    def tracsTs_x86_tsc(self):
        pass

    def get_util(self, _filter=None):
        """Drop all marker events"""
        events = [e for e in self.timeline if e.type == 'period']

        if _filter is not None:
            events = [e for e in events if _filter(e)]

        if len(events) == 0:
            return -1

        totalT = events[-1].endTime - events[0].startTime
        runT = 0

        for e in events:
            runT += e.duration

        util = runT * 100 / totalT
        # print("### Util of %s-%02d: %.2f" % (self.coreType, self.coreId, util))

        return util

    def get_core_name(self):
        try:
            n = self.dpuCore.name
        except:
            n = None

        return n

    def get_core_full_name(self):
        try:
            n = self.dpuCore.full_name
        except:
            n = None

        return n

    def len(self):
        return len(self.timeline)

    def add(self, new: vaiTimelineEvent):
        if new.coreId != self.coreId:
            return

        new.ts = self.transTs(new.ts)

        if new.type == 'start':
            self.eventStack.append(new)
        if new.type == 'done':
            """1. looking for start event in eventstack"""
            for s in self.eventStack[::-1]:
                r = dpuEventPaired(s, new)
                if r != None:
                    try:
                        self.eventStack.remove(s)
                    except BaseException:
                        pass
                    """2. if matched, add into self.timeline"""
                    self.timeline.append(r)

    def toJson(self, prefix="thread", _color=None):
        out = []
        for i in self.timeline:
            if i.type != 'period':
                continue

            if _color is None or _color(i) is None:
                color = "#%06x" % ((hash(str(i.pid)) >> 2) % (256 * 256 * 256))
            else:
                color = _color(i)

            if i.code != "":
                prefix = i.code
            out.append([prefix, i.pid, i.startTime, i.endTime, color])

        return out


def createTimelines(coreType, id, options: dict, timeout=0):
    timelines = []

    if isinstance(id, list):
        for i in id:
            timelines.append(vaiTimeline(coreType, i, options, timeout))
    elif isinstance(id, str):
        timelines.append(vaiTimeline(coreType, id, options, timeout))
    elif isinstance(id, int):
        for i in range(0, id):
            timelines.append(vaiTimeline(coreType, i, options, timeout))

    if len(timelines) == 0:
        assert ()

    return timelines
