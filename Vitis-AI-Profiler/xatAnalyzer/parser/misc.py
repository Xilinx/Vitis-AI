
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
from parser.timelineUtil import utilization


class miscParser(parser.parserBase.Parser):
    def __init__(self):
        super().__init__('misc')

    def parse(self, options):
        util = []
        for key in utilization:
            util.append({"type": "util", "info": {key: utilization[key]}})

        return {"INFO-UTIL": [{"type": "Utilization", "info": utilization}]}


parser.parserBase.register(miscParser())
