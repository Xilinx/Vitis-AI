
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

import sys
import os
import collector


class fileCollector(collector.collectorBase.Collector):
    def __init__(self):
        super().__init__(name='file')
        self.fileList = []

    def prepare(self, conf: dict) -> dict:
        fileOption = conf.get('collector', {}).get('file')
        for f in fileOption.keys():
            self.fileList.append(f)
            fileInstOption = fileOption[f]

        return conf

    def start(self):
        super().start()

    def stop(self):
        super().stop()

    """
    Return Format:
    {
      'filename': content,
      'filename2': content,
      ...
    }
    """

    def getData(self):
        data = {}

        for f in self.fileList:
            data.update({f: open(f, 'rb').read()})
            print("@#@@@@ open %s" % f)

        return data


collector.collectorBase.register(fileCollector())
