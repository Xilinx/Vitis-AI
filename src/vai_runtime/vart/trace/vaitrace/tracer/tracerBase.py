
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
import time


class Tracer:
    def __init__(self, name: str, source, compatible):
        self.name = name
        self.timesync = 0
        self.compatibleList = compatible
        self.enabled = False

        if type(source) is str:
            self.source = [source]
        elif type(source) is list:
            self.source = source
        else:
            assert ()

    def compatible(self, platform: {}):
        machine = platform['machine']
        release = platform['release']

        if machine in self.compatibleList['machine']:
            logging.debug("Compatible [%s]: True" % self.name)
            return True

        logging.debug("Compatible [%s]: False" % self.name)
        return False

    def prepare(self, options, debug=False):
        return options

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def process(self, data, t_range=[]):
        pass

    def getData(self):
        pass


__tracers = []
__activeTracers = []


def getTracerInstances() -> list:
    return __tracers


def getAvailableTracers() -> list:
    return [t.name for t in __tracers]


def getSourceRequirement() -> list:
    req = []

    for t in __tracers:
        if type(t.source) is str:
            req.append(t.source)
        elif type(t.source) is list:
            req = req + t.source
        else:
            assert (True)

    return req


def start():
    global __start_time

    __start_time = time.monotonic()
    for t in __tracers:
        if t.enabled == False:
            continue
        t.start()


def stop():
    global __stop_time

    __stop_time = time.monotonic()
    for t in __tracers:
        if t.enabled == False:
            continue
        t.stop()


def process(data: dict):
    for t in __tracers:
        if t.enabled == False:
            continue
        logging.debug("Processing [%s]..." % t.name)
        t.process(data, [__start_time, __stop_time])


def prepare(options: dict) -> dict:
    debug = options['control'].get('debug', False)

    def merge(a: dict, b: dict):
        if hasattr(a, "keys") and hasattr(b, "keys"):
            for kb in b.keys():
                if kb in a.keys():
                    merge(a[kb], b[kb])
                else:
                    a.update(b)

    for t in __tracers:
        logging.debug("Preparing [%s]..." % t.name)
        platform = options['control']['platform']

        disable = options.get('tracer', {}).get(
            t.name, {}).get("disable", False)
        if disable:
            continue

        if t.compatible(platform):
            t.enable()
            merge(options, t.prepare(options, debug))

    return options


def getData():
    data = {}
    for t in __tracers:
        if t.enabled == False:
            continue
        logging.debug("Getting Data [%s]..." % t.name)
        data.update({t.name: t.getData()})
        data.setdefault("timesync", {}).update({t.name: t.timesync})
        data.setdefault("timesync", {}).update(
            {"time_range": [__start_time, __stop_time]})
        logging.debug("[time sync]: %s: %f" % (t.name, t.timesync))

    return data


def register(tracerInstance):
    __tracers.append(tracerInstance)
