
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

import parser.parserBase
import parser.tracepointUtil

from parser.timelineUtil import *

DPUs = []
"""
subGraphName: [time1, time2 ...]
"""
subGraphStat = {}
"""
dpu_id: {subGraphName: [time1, time2 ...]
"""
dpuLatencyStat = {}


def subGraphStatAdd(subGraphName: str, coreId: int, time: float):
    subGraphStat.setdefault(subGraphName, []).append(time)
    dpuLatencyStat.setdefault(coreId, {}).setdefault(subGraphName, []).append(time)


POINT_PER_SECOND = 100
WINDOW_LENGTH = 0.33


def DPULatencyProfiling():
    latency = []

    for key in dpuLatencyStat:
        kernelLatency = {}
        for sub in dpuLatencyStat[key]:
            t = sum(dpuLatencyStat[key][sub]) / len(dpuLatencyStat[key][sub])
            kernelLatency.setdefault(sub, "%.2f us" % (t * 1000 * 1000))

        latency.append({"type": "DPU_%d Latency" % (key), "info": kernelLatency})

    return {"INFO-LATE": latency}


def FPSProfiling():
    """Add FPS data"""
    dataFPS = []
    eventFPS = []

    totalRunCnt = 0
    global DPUs
    for dpu in DPUs:
        if dpu.len() == 0:
            continue
        for run in dpu.timeline:
            if run.type != 'period':
                continue

            try:
                batch = int(run.batch)
            except BaseException:
                continue

            totalRunCnt += batch
            eventFPS.append((run.startTime, batch))

    eventFPS.sort(key=lambda x: x[0])

    def eventFilter(events, start, end):
        ret = []
        for e in events:
            if e[0] > end:
                break
            elif e[0] < start:
                continue
            else:
                ret.append(e)
        return ret

    if len(eventFPS) < 20:
        return {'FPS-0': []}
    else:
        startTime = eventFPS[0][0]
        endTime = eventFPS[-1][0]
        overAllFPS = round(totalRunCnt / (endTime - startTime), 2)
        interval = 1 / POINT_PER_SECOND

        timePoint = startTime

        while (timePoint + WINDOW_LENGTH < endTime):
            eventsInWindow = eventFilter(eventFPS, timePoint, timePoint + WINDOW_LENGTH)
            if len(eventsInWindow) < 3:
                """Skip this point"""
                timePoint += interval
                continue
            frames = 0
            for e in eventsInWindow:
                frames += e[1]
            time = WINDOW_LENGTH
            fps = round(frames / time, 1)
            dataFPS.append([timePoint, fps])

            timePoint += interval

    print("Overall FPS %.2f" % overAllFPS)
    return {'FPS-0': dataFPS}


def fineGranularityProfiling():
    global DPUs
    for dpu in DPUs:
        if dpu.len() == 0:
            continue
        for run in dpu.timeline:
            if run.type != 'period':
                continue

            if run.info.find(':') < 1:
                continue

            title, info = run.info.split(':')
            if title == 'subgraph':
                subGraphName = info.strip()
                subGraphStatAdd(subGraphName, run.coreId, run.duration)


class DPUControllerParser(parser.parserBase.Parser):
    def __init__(self):
        super().__init__('DPUController')

        self.vartThreads = {
            # pid: {runtime info}
        }

    def getDpuRuntimeInfo(self, event, key):
        pid = event.pid
        subgraphName = self.vartThreads.get(pid, {}).get(key)
        return "%s" % subgraphName

    """DPUR runtime info format: 'key1: value1, key2: value2 ...' """

    def infoParse(self, info):
        retInfo = {}
        info = [i.strip() for i in info.split(',')]
        for i in info:
            if len(i.split(':')) != 2:
                continue
            k, v = [x.strip() for x in i.split(':')]
            retInfo[k] = v

        return retInfo

    def parseDpuRuntimeEvent(self, event: vaiTimelineEvent):
        runtimeEventInfo = event.info
        subgraphName = "subgraph_unknown"
        batchSize = 1

        pid = event.pid
        if pid not in self.vartThreads.keys():
            self.vartThreads.setdefault(pid, {"subgraph": None})
            self.vartThreads.setdefault(pid, {"batch": 1})

        runtimeEventList = self.infoParse(runtimeEventInfo)

        for key in runtimeEventList.keys():
            if key == 'subgraph':
                subgraphName = runtimeEventList.get(key, "subgraph")
            elif key == 'batch':
                batchSize = runtimeEventList.get(key, 1)
            elif key == 'hwconuter':
                pass
            else:
                pass

        self.vartThreads[pid]["subgraph"] = subgraphName
        self.vartThreads[pid]["batch"] = batchSize

    def parse(self, data, options):
        """Two types of event tracing data included: dpuRuntimeEvent & dpuControllerEvent"""
        global DPUs
        cuRetData = {}

        """Prepare at most 8 DPU timelines"""
        global DPUs
        DPUs = createTimelines('DPU', 8, options)

        for l in data:
            event = parser.tracepointUtil.tracepointEvent(l).toTimelineEvent()
            if event is None:
                continue

            if event.coreType == 'DPUR':
                self.parseDpuRuntimeEvent(event)
                continue

            event.info += "%s:%s" % ("subgraph", self.getDpuRuntimeInfo(event, "subgraph"))
            event.batch = self.getDpuRuntimeInfo(event, "batch")
            if event.coreType == 'DPUC':
                DPUs[event.coreId].add(event)

        for dpu in DPUs:
            count = dpu.len()
            if count > 0:
                cuRetData["TIMELINE-%s_%s" % (dpu.coreType, dpu.coreId)] = dpu.toJson()
                dpu.getUtil()

        fineGranularityProfiling()
        cuRetData.update(FPSProfiling())
        cuRetData.update(DPULatencyProfiling())

        return cuRetData


parser.parserBase.register(DPUControllerParser())
