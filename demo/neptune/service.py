# Copyright 2019 Xilinx Inc.
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

class Service(object):
    """
    The base class for all services. All services inherit from this class
    """

    def __init__(self, prefix, artifacts, graph):
        self._artifacts = artifacts
        self._prefix = prefix
        self._proc = None
        self._graph = graph


    def start(self, args):
        # os.environ["XDNN_VERBOSE"] = "1"
        # os.environ["XBLAS_EMIT_PROFILING_INFO"] = "1"
        self._graph.serve(args, background=True)


    def stop(self):
        self._graph.stop()
