# Copyright 2019 Xilinx, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import json
import argparse


class Description:
    def __init__(self, desPath):
        self.desPath = desPath
        self.description = dict()
        self.description['name'] = ""
        self.description['description'] = ""
        self.description['flow'] = "hls"
        self.description['platform_whitelist'] = ["u280", "u250", "u200"]
        self.description['platform_blacklist'] = []
        self.description['part_whitelist'] = []
        self.description['part_blacklist'] = []
        self.description['project'] = ""
        self.description['solution'] = "sol"
        self.description['clock'] = "3.3333"
        self.description['topfunction'] = "uut_top"
        self.description['top'] = {
            "source": ["${XF_PROJ_ROOT}/L1/tests/hw/uut_top.cpp"],
            "cflags": "-std=c++11 -I${XF_PROJ_ROOT}/L1/include/hw"
        }
        self.description['testbench'] = {
            "source": ["${XF_PROJ_ROOT}/L1/tests/sw/src/test.cpp"],
            "cflags": "-std=c++11 -I${XF_PROJ_ROOT}/L1/tests/sw/include",
            "ldflags": "",
            "argv": {
                "hls_cosim": "",
                "hls_csim": ""
            },
            "stdmath": False}
        self.description['testinfo'] = {"disable": True,
                                        "jobs": [{"index": 0,
                                                  "dependency": [],
                                                  "env": "",
                                                  "cmd": "",
                                                  "max_memory_MB": 16384,
                                                  "max_time_min": 300
                                                  }],
                                        "targets": [
                                            "hls_csim",
                                            "hls_csynth",
                                            "hls_cosim",
                                            "hls_vivado_syn",
                                            "hls_vivado_impl"
                                        ],
                                        "category": "canary"
                                        }

    def setTest(self, b):
        self.description['testinfo']['disable'] = not b

    def addArgv(self, strs):
        self.description['testbench']['argv']['hls_csim'] = strs
        self.description['testbench']['argv']['hls_cosim'] = strs

    def setTestCFlags(self, cflags):
        self.description['testbench']['cflags'] += " " + cflags

    def setTopCFlags(self, cflags):
        self.description['top']['cflags'] += " " + cflags

    def setTestSource(self, name):
        self.description['testbench']['source'] = [name]

    def setTopSource(self, name):
        self.description['top']['source'] = [name]

    def load(self):
        with open(self.desPath, 'r') as fr:
            self.description = json.load(fr.read())

    def dump(self):
        with open(self.desPath, 'w') as fw:
            fw.write(json.dumps(self.description, indent=4, sort_keys=True))

    def setName(self, name):
        self.description['name'] = "Xilinx XF_BLAS.%s" % name

    def setProject(self, name):
        self.description['project'] = "%s_test" % name


def main(args):
    desc = Description(args.dirname)
    desc.setName(args.func + "." + args.testname)
    desc.setProject(args.func + "_" + args.testname)
    desc.addArgv(
        "${XF_PROJ_ROOT}/L1/tests/hw/" +
        args.func +
        "/tests/" +
        args.testname +
        "/data/")

    desc.setTestCFlags("-I${XF_PROJ_ROOT}/L1/include/hw/xf_blas/helpers/utils")
    desc.setTestCFlags("-I${XF_PROJ_ROOT}/L1/include/hw")
    desc.setTestCFlags("-I${XF_PROJ_ROOT}/L1/tests/hw/" + args.func)
    desc.setTestCFlags(
        "-I${XF_PROJ_ROOT}/L1/tests/hw/" +
        args.func +
        "/tests/" +
        args.testname)

    desc.setTopCFlags("-I${XF_PROJ_ROOT}/L1/include/hw/xf_blas/helpers/utils")
    desc.setTopCFlags("-I${XF_PROJ_ROOT}/L1/tests/hw/" + args.func)
    desc.setTopCFlags(
        "-I${XF_PROJ_ROOT}/L1/tests/hw/" +
        args.func +
        "/tests/" +
        args.testname)
    desc.setTestSource(
        "${XF_PROJ_ROOT}/L1/tests/hw/" +
        args.func +
        "/test.cpp")
    desc.setTopSource(
        "${XF_PROJ_ROOT}/L1/tests/hw/" +
        args.func +
        "/uut_top.cpp")
    desc.setTest(True)
    desc.dump()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate description json file')
    parser.add_argument('--dirname', type=str, default="./description.json")
    parser.add_argument('--testname', type=str)
    parser.add_argument('--func', type=str)
    args = parser.parse_args()
    main(args)
