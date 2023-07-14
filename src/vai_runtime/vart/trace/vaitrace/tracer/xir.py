#!/usr/bin/python3
# -*- coding:utf-8 -*-

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


import os
import sys
import re
import json
import logging
import tracer.tracerBase
from ctypes import *
from subprocess import Popen, PIPE

import vaitraceDefaults


class XIRTracer(tracer.tracerBase.Tracer):
    def __init__(self):
        super().__init__('XIR', source=['stdIO'], compatible={
            'machine': ["x86_64", "aarch64"]})
        """
        for xmodel
        [
          {
            'model': 'abs path to model',
            'graph': {} // struct of the model
          }
        
        ]
        """
        self.models = []

    def getGraphStruct(self, modelPath: str, kernelName="") -> dict:

        try:
            a = Popen(['./dump_graph', modelPath], stdout=PIPE)
            a.wait()

            g = str(a.stdout.read(), encoding='utf-8')
        except:
            #logging.error("Acquiring xir graph failed")
            g = "{}"

        return json.loads(g)

    def process(self, data, t_range=[]):
        xirGraph = self.getGraphStruct(self.xmodel)
        if len(xirGraph) > 0:
            self.models.append({'model': self.xmodel,
                                'graph': xirGraph})

    def start(self):
        super().start()

    def stop(self):
        super().stop()

    def prepare(self, options: dict, debug: bool):
        self.xmodel = options.get('xmodel', "")

    def getData(self):
        return self.models


tracer.tracerBase.register(XIRTracer())
