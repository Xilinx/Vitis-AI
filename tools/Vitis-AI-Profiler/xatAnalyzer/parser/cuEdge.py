
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
import parser.ftraceUtil

CUs = []


class cuEdgeParser(parser.parserBase.Parser):
    def __init__(self):
        super().__init__('cuEdge')

    def parse(self, data, options):
        #cuMap = options['cuMap']
        cuEvents = {}
        cuRetData = {}

        for l in data:
            event = parser.ftraceUtil.parse(l, options)
            idx = event.infoDetail['cu_idx']
            coreID = "TIMELINE-CU_%s" % hex(idx)

            if coreID not in cuEvents.keys():
                cuEvents[coreID] = []

            cuEvents[coreID].append(event)

        """return {"CU-dpu_1": [[st, et], ...]}"""
        for core in cuEvents.keys():
            totalT = cuEvents[core][-1].timeStamp - cuEvents[core][0].timeStamp
            if totalT == 0:
                continue

            cuRetData[core] = []
            runT = 0
            start = 0

            for e in cuEvents[core]:
                eventType = e.func
                if eventType == 'cu_start':
                    start = e.timeStamp
                elif eventType == 'cu_done':
                    if start == 0:
                        continue
                    # prefix, i.pid, i.startTime, i.endTime, color
                    cuRetData[core].append(["thread", 88, start, e.timeStamp, "#ee0000"])
                    runT += (e.timeStamp - start)
                else:
                    continue

            print("##### Util of %s: %.2f" % (core, runT * 100 / totalT))

        return cuRetData


parser.parserBase.register(cuEdgeParser())
