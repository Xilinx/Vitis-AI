
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

class Parser:
    def __init__(self, name):
        self.name = name

    def parse(self, data: dict, options=None):
        return {}


__parsers = []


def getParsers():
    return __parsers


def getParser(name: str):
    for p in __parsers:
        if name == p.name:
            return p

    print("Parser for [%s] Could Not Found" % name)
    return False


def getParsersName():
    return [p.name for p in __parsers]


def parse(data: dict, options=None):
    outData = {}
    for p in getParsers():
        if p.name in data.keys():
            print("[%s] Processing ..." % p.name)
            print("###", p.name, "data length: ", len(data[p.name]))
            outData.update(p.parse(data[p.name], options))

        if p.name == "misc":
            print("[%s] Processing ..." % p.name)
            outData.update(p.parse(options))

    return outData


def register(parserInstance):
    __parsers.append(parserInstance)
