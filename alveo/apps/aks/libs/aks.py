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
import numpy as np
import ctypes
import os
import sys
moddir = os.path.dirname(os.path.abspath(__file__))
# _lib = ctypes.cdll.LoadLibrary('%s/libAks.so' % moddir)
_lib = ctypes.CDLL('%s/libAks.so' % moddir)

# AKS::SysManager* createSysManager();
_lib.createSysManagerExt.restype  = ctypes.c_void_p
_lib.createSysManagerExt.argtypes = []

# void loadKernel(AKS::SysManager* sysMan, const char* kernelDir);
_lib.loadKernels.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

# void loadGraphs(AKS::SysManager* sysMan, const char* graphPath);
_lib.loadGraphs.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

# AKS::AIGraph* getGraph(AKS::SysManager* sysMan, const char* graphName);
_lib.getGraph.restype  = ctypes.c_void_p
_lib.getGraph.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

# void enqueueJob(AKS::SysManager* sysMan, AKS::AIGraph* graph,
#    const char* imagePath, AKS::UserParams* params);
_lib.enqueueJob.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

# void waitForAllResults(AKS::SysManager* sysMan);
_lib.waitForAllResults.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

# void report(AKS::SysManager* sysMan, AKS::AIGraph* graph);
_lib.report.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

# void deleteSysManager(AKS::SysManager* sysMan);
_lib.deleteSysManagerExt.argtypes = []


class SysManager:
  def __init__(self):
    self._csysmgr = _lib.createSysManagerExt()

  def loadKernels(self, kernelDir):
    _lib.loadKernels(self._csysmgr, kernelDir.encode('ascii'))

  def loadGraphs(self, graphDir):
    _lib.loadGraphs(self._csysmgr, graphDir.encode('ascii'))

  def getGraph(self, graphName):
    return _lib.getGraph(self._csysmgr, graphName.encode('ascii'))

  def enqueueJob(self, graph, imagePath):
    _lib.enqueueJob(self._csysmgr, graph, imagePath.encode('ascii'), ctypes.c_void_p())

  def waitForAllResults(self, graph = ctypes.c_void_p()):
    _lib.waitForAllResults(self._csysmgr, graph)

  def report(self, graph):
    _lib.report(self._csysmgr, graph)

  def __del__(self):
    _lib.deleteSysManagerExt()

