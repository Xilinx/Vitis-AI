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

import json


class Model:
    def __init__(
            self,
            js=None):
        self.js = js

        if self.js is not None:
            self.load()

    def load(self):
        with open(self.js, 'r') as fr:
            self.data = json.load(fr)

        self.name = self.data['mname']
        self.time = self.data['nt']
        self.nxb = self.data['blen']
        self.nyb = self.data['blen']
        self.nzb = self.data['blen']
        self.x = self.data['nx'] + self.nxb * 2
        self.y = self.data['ny'] + self.nyb * 2
        self.z = self.data['nz'] + self.nzb * 2
        self.order = self.data['stencil_order']
        self.shot = self.data['source_total']
        self.fpeak = self.data['fpeak']
        self.taper_factor = self.data['taper_factor']
        self.dx = self.data['dx']
        self.dy = self.data['dy']
        self.dz = self.data['dz']
        self.dt = self.data['dt']
        self.vpefile = self.data['vpefile']

    def parse(self, args):
        self.time = args.time
        self.nxb = args.nxb
        self.nyb = args.nyb
        self.nzb = args.nzb
        self.x = args.x
        self.y = args.y
        self.z = args.z
        self.order = args.order
        self.shot = args.shot
        self.dx = 15
        self.dy = 15
        self.dz = 15
        self.dt = 0.0002
