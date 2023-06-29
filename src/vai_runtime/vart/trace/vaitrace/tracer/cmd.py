
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

import logging
import tracer.tracerBase


class cmdTracer(tracer.tracerBase.Tracer):
    def __init__(self):
        super().__init__('cmd', source=[], compatible={
            'machine': ["x86_64", "aarch64"]})
        self.cmd = []

    def prepare(self, option: dict, debug: bool):
        """
        { control: {
                cmd: [],
                xat: {}
            }
        }
        """
        "Handle Input Options"
        self.cmd = option.get('control', {})

        if len(self.cmd) == 0:
            assert ()

        "Handle Output Options"
        return {}

    def process(self, data, t_range=[]):
        pass

    def getData(self):
        return self.cmd


tracer.tracerBase.register(cmdTracer())
