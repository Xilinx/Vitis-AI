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


# Legacy Python Extensions
class SysManagerLegacy:
  def __init__(self):
    moddir = os.path.dirname(os.path.abspath(__file__))
    # self._lib = ctypes.cdll.LoadLibrary('%s/libAks.so' % moddir)
    self._lib = ctypes.PyDLL('%s/libAks.so' % moddir)

    # AKS::SysManager* createSysManager();
    self._lib.createSysManagerExt.restype  = ctypes.c_void_p
    self._lib.createSysManagerExt.argtypes = []

    # void loadKernel(AKS::SysManager* sysMan, const char* kernelDir);
    self._lib.loadKernels.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    # void loadGraphs(AKS::SysManager* sysMan, const char* graphPath);
    self._lib.loadGraphs.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    # AKS::AIGraph* getGraph(AKS::SysManager* sysMan, const char* graphName);
    self._lib.getGraph.restype  = ctypes.c_void_p
    self._lib.getGraph.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    # void enqueueJob(AKS::SysManager* sysMan, AKS::AIGraph* graph,
    #    const char* imagePath, AKS::NodeParams* params);
    self._lib.enqueueJob.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

    # void waitForAllResults(AKS::SysManager* sysMan);
    self._lib.waitForAllResults.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    # void resetTimer(AKS::SysManager* sysMan);
    self._lib.resetTimer.argtypes = [ctypes.c_void_p]

    # void report(AKS::SysManager* sysMan, AKS::AIGraph* graph);
    self._lib.report.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    # void deleteSysManager(AKS::SysManager* sysMan);
    self._lib.deleteSysManagerExt.argtypes = []

    self._csysmgr = self._lib.createSysManagerExt()

  def loadKernels(self, kernelDir):
    self._lib.loadKernels(self._csysmgr, kernelDir.encode('ascii'))

  def loadGraphs(self, graphDir):
    self._lib.loadGraphs(self._csysmgr, graphDir.encode('ascii'))

  def getGraph(self, graphName):
    return self._lib.getGraph(self._csysmgr, graphName.encode('ascii'))

  def enqueueJob(self, graph, imagePath):
    self._lib.enqueueJob(self._csysmgr, graph, imagePath.encode('ascii'), ctypes.c_void_p())

  def waitForAllResults(self, graph = ctypes.c_void_p()):
    self._lib.waitForAllResults(self._csysmgr, graph)

  def resetTimer(self):
    self._lib.resetTimer(self._csysmgr)

  def report(self, graph):
    self._lib.report(self._csysmgr, graph)

  def clear(self):
    pass

  def __del__(self):
    self._lib.deleteSysManagerExt()


# New Python Extensions
default_flags = sys.getdlopenflags()
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)
import _aks
SysManager         = _aks.SysManager
NodeParams         = _aks.NodeParams
DynamicParamValues = _aks.DynamicParamValues
sys.setdlopenflags(default_flags)
