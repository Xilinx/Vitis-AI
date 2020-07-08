
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

import re

pattAPM = re.compile(r'APM\s(\d+\.\d+)\s(\d+)\s(\d+)\s(\d+)\s(\d+)\s(\d+)\s(\d+)\s(\d+)\s(\d+)\s(\d+)\s(\d+)')


class APMRecord:
    def __init__(self, rec):
        self.time = float(rec[0])
        self.data = []

        for d in rec[1:]:
            self.data.append(int(d) / 1024 / 1024.0)

        self.total = sum(self.data)

    def __str__(self):
        return "Time: %.7f  %.2f" % (self.time, self.total)


class xapmParser(parser.parserBase.Parser):
    def __init__(self):
        super().__init__('xapm')

    def parse(self, data, options):
        apmData = []

        for l in data:
            apmRec = pattAPM.match(l.strip())
            if apmRec is not None:
                apm = APMRecord(apmRec.groups())
                if False:
                    apmData.append([apm.time, apm.total])
                else:
                    apmData.append([apm.time, apm.data[0] + apm.data[1],
                                    apm.data[2] + apm.data[3],
                                    apm.data[4] + apm.data[5],
                                    apm.data[6] + apm.data[7],
                                    apm.data[8] + apm.data[9]])

        if (len(apmData) < 10):
            print("Too little APM data")
            return {}

        return {'APM-0': apmData}


parser.parserBase.register(xapmParser())
