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
        self.description['platform_whitelist'] = ["u280"]
        self.description['platform_blacklist'] = []
        self.description['part_whitelist'] = []
        self.description['part_blacklist'] = []
        self.description['project'] = ""
        self.description['solution'] = "sol"
        self.description['clock'] = "3.3333"
        self.description['topfunction'] = "top"
        self.description['top'] = {
            "source": [],
            "cflags": "-ICUR_DIR -I${XF_PROJ_ROOT}/L1/include/hw -I${XF_PROJ_ROOT}/../xf_blas/L1/include/hw"
        }
        self.description['testbench'] = {
            "source": ["CUR_DIR/../../main.cpp"],
            "cflags": "-std=c++14 -ICUR_DIR -I${XF_PROJ_ROOT}/L1/include/hw -I${XF_PROJ_ROOT}/../xf_blas/L1/include/hw -I${XF_PROJ_ROOT}/../xf_blas/L1/tests/sw/include",
            "ldflags": "",
            "argv": {
                "hls_cosim": "CUR_DIR/data/",
                "hls_csim": "CUR_DIR/data/"},
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

    def setTopSource(self, name):
        self.description['top']['source'] = [name]

    def load(self):
        with open(self.desPath, 'r') as fr:
            self.description = json.load(fr.read())

    def dump(self, relpath=None):
        text = json.dumps(self.description, indent=4, sort_keys=True)
        if relpath is not None:
            text = text.replace(r"CUR_DIR", relpath)
        with open(self.desPath, 'w') as fw:
            fw.write(text)

    def setName(self, name):
        self.description['name'] = "Xilinx XF_BLAS.%s" % name

    def setProject(self, name):
        self.description['project'] = "%s_test" % name


def main(args):
    relpath = r"${XF_PROJ_ROOT}/%s" % args.testDir[args.testDir.find("L1")]
    desc = Description(os.path.join(args.testDir, "description.json"))
    name = args.testDir.split('/')[-1]
    desc.setName(args.func + "." + name)
    desc.setProject(args.func + "_" + name)
    desc.setTopSource("CUR_DIR/../../%s.cpp" % args.func)
    desc.setTest(True)
    desc.dump(relpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate description json file')
    parser.add_argument('--testDir', type=str, required=True)
    parser.add_argument('--func', type=str, required=True)
    args = parser.parse_args()
    main(args)
