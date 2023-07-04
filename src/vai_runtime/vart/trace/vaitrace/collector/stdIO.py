
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
from tempfile import TemporaryFile
import collector


class stdIOCollector(collector.collectorBase.Collector):
    def __init__(self, stdout="", stderr=""):
        super().__init__(name='stdIO')

    def __del__(self):
        if self.stdoutLogger is not None:
            self.stdoutLogger.close()
        if self.stderrLogger is not None:
            self.stderrLogger.close()

    def prepare(self, conf: dict) -> dict:
        self.stdoutLogger = TemporaryFile('w+t')
        self.stderrLogger = TemporaryFile('w+t')

        logger = {
            'stdout': self.stdoutLogger,
            'stderr': self.stderrLogger
        }

        return logger

    def start(self):
        super().start()

    def stop(self):
        super().stop()

    def getData(self):
        self.stdoutLogger.flush()
        self.stdoutLogger.seek(0)

        self.stderrLogger.flush()
        self.stderrLogger.seek(0)

        return self.stdoutLogger.readlines() + self.stderrLogger.readlines()


collector.collectorBase.register(stdIOCollector())
