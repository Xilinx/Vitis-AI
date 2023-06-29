
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
import sys
import os
import json
import tracer.tracerBase


class xrtTracer(tracer.tracerBase.Tracer):
    def __init__(self):
        super().__init__('xrt', source=[], compatible={
            'machine': ["x86_64", "aarch64"]})
        self.xrtInfo = {}

    """
	$ xbutil scan
	INFO: Found total 1 card(s), 1 are usable
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	System Configuration
	OS name:        Linux
	Release:        3.10.0-957.el7.x86_64
	Version:        #1 SMP Thu Nov 8 23:39:32 UTC 2018
	Machine:        x86_64
	Model:          PowerEdge R740
	CPU cores:      12
	Memory:         46525 MB
	Glibc:          2.17
	Distribution:   CentOS Linux 7 (Core)
	Now:            Thu Sep 10 15:51:04 2020
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	XRT Information
	Version:        2.6.655
	Git Hash:       2d6bfe4ce91051d4e5b499d38fc493586dd4859a
	Git Branch:     2020.1
	Build Date:     2020-05-22 19:05:52
	XOCL:           2.6.655,2d6bfe4ce91051d4e5b499d38fc493586dd4859a
	XCLMGMT:        2.6.655,2d6bfe4ce91051d4e5b499d38fc493586dd4859a
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     [0] 0000:3b:00.1 xilinx_u50_gen3x4_xdma_base_2 user(inst=128)

    """

    def prepare(self, option: dict, debug: bool):
        return {}

    def process(self, data, t_range=[]):
        """
        Turn off all XRT profile and trace options
        So that, we can prevent 'xbutil dump' generates another xclbin.ex.run_summary that would
        over-write the prev one for vitis-ai process
        """
        os.environ["Debug.profile"] = "false"
        os.environ["Debug.xrt_profile"] = "false"
        os.environ["Debug.vitis_ai_profile"] = "false"
        os.environ["Debug.lop_trace"] = "false"
        os.environ["Debug.timeline_trace"] = "false"
        os.environ["Debug.data_transfer_trace"] = "off"

        xbutil_path_alt = ["/usr/bin/xbutil", "/opt/xilinx/xrt/bin/xbutil"]
        xbutil_cmd = "xbutil"
        for alt in xbutil_path_alt:
            if os.path.exists(alt):
                xbutil_cmd = alt

        try:
            d = os.popen('%s dump' % xbutil_cmd).read()

            if len(d) == 0:
                return

            self.xrtInfo = json.loads(d)
        except:
            return

    def getData(self):
        return self.xrtInfo


tracer.tracerBase.register(xrtTracer())
