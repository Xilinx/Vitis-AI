
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

import Options


class hwInfoParser(parser.parserBase.Parser):
    def __init__(self):
        super().__init__('hwInfo')

    def parse(self, data, options):
        hwinfo = data
        cuMap = {}

        for hw in hwinfo:
            if hw['type'] == 'CUs' or hw['type'] == 'CUMap':
                for cu in hw['info']:
                    cuMap.update({hw['info'][cu]: cu})
                    cuMap.update({cu: hw['info'][cu]})
                hwinfo.pop(hwinfo.index(hw))

        for hw in hwinfo:
            """Flatten DPU Multi Cores' Info"""
            if hw['type'] == 'DPU':
                DPUInfo = hwinfo.pop(hwinfo.index(hw))
                flattenDPUInfo = []
                for coreId in DPUInfo['info']:
                    dpuInfo = {'type': coreId, "info": DPUInfo['info'][coreId]}

                    hwinfo.append(dpuInfo)

        Options.merge(options, {'cuMap': cuMap})

        """return: {key, {v}}"""
        return {'INFO-HW': hwinfo}


parser.parserBase.register(hwInfoParser())
