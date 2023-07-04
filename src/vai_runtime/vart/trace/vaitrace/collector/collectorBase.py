
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

class Collector:
    def __init__(self, name):
        self.numClient = 0
        self.running = 0
        self.name = name

    def getData(self):
        pass

    def prepare(self):
        pass

    def start(self):
        self.running = True

    def stop(self):
        self.running = False


__collectors = []
__activeCollectors = []


def getCollectorInstances():
    return __collectors


def getCollectorInstances(name: str):
    for c in __collectors:
        if match(name == c.name):
            return c

    return None


def getAvailableCollectors():
    return [c.name for c in __collectors]


def start():
    for c in __activeCollectors:
        c.start()


def stop():
    for c in __activeCollectors:
        c.stop()


"""
option: {
    collectorOption: {
        ftrace: {
            sched: [],
            cu: [],
            ...
        },
    }
}

"""


def prepare(conf: dict, requirements: list) -> dict:
    def match(req: str, collector):
        if req.lower() == collector.name.lower():
            return True
        return False

    for name in list(set(requirements)):
        for c in __collectors:
            if match(name, c):
                __activeCollectors.append(c)
                conf.update(c.prepare(conf))

    return conf


def getData(source=None) -> dict:
    data = dict()

    """Make sure all avtive has been stopped"""
    for c in __activeCollectors:
        if c.running == True:
            assert ()

    for c in __activeCollectors:
        data.update({c.name: c.getData()})

    return data


def register(collectorInstance):
    __collectors.append(collectorInstance)
