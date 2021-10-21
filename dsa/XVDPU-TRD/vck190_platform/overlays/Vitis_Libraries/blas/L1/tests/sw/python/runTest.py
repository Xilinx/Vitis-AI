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


import numpy as np
import ctypes as C
import argparse
import os
import sys
import json
import pdb
from blas_gen_bin import BLAS_GEN
from hls import HLS, Parameters
from makefile import Makefile
from operation import OP, BLAS_L1, BLAS_L2, OP_ERROR
from table import list2File
import threading
import time


class RunTest:
    def __init__(self, profile, args, opLocks=dict()):
        self.args = args
        self.opLocks = opLocks
        self.profilePath = profile
        self.profile = None
        self.parEntries = 1
        self.logParEntries = 0
        self.valueRange = None
        self.numToSim = 1
        self.numSim = 0
        self.makefile = args.makefile

        self.hls = None
        self.typeDict = {
            np.int8: 'int8_t',
            np.int16: 'int16_t',
            np.int32: 'int32_t',
            np.int64: 'int64_t',
            np.uint8: 'uint8_t',
            np.uint16: 'uint16_t',
            np.uint32: 'uint32_t',
            np.uint64: 'uint64_t',
            np.float32: 'float',
            np.float64: 'double'
        }

    def parseProfile(self):
        with open(self.profilePath, 'r') as fh:
            self.profile = json.loads(fh.read())

        self.opName = self.profile['op']
        self.opClass = OP.parse(self.opName)

        self.minValue = self.profile['valueRange'][0]
        self.maxValue = self.profile['valueRange'][1]

        self.op = eval(
            self.opClass).parse(
            self.opName,
            self.maxValue,
            self.minValue)

        if 'logParEntries' in self.profile:
            self.logParEntries = self.profile['logParEntries']
        else:
            self.logParEntries = -1

        if self.logParEntries == -1:
            self.parEntries = self.profile['parEntries']
        else:
            self.parEntries = 1 << self.logParEntries

        if 'dataTypes' in self.profile:
            self.dataTypes = [eval('np.%s' % dt)
                              for dt in self.profile['dataTypes']]

        if 'retTypes' in self.profile:
            self.retTypes = [eval('np.%s' % dt)
                             for dt in self.profile['retTypes']]

        self.numToSim = self.profile['numSimulation']

        if self.opName not in self.opLocks:
            self.opLocks[self.opName] = threading.Lock()
        self.opLock = self.opLocks[self.opName]
        self.testPath = os.path.join('out_test', self.op.name)
        self.reports = list()

        self.libPath = os.path.join(self.testPath, 'libs')
        with self.opLock:
            if not os.path.exists(self.libPath):
                os.makedirs(self.libPath)

        self.dataPath = os.path.join(self.testPath, 'data')
        with self.opLock:
            if not os.path.exists(self.dataPath):
                os.makedirs(self.dataPath)
        hlsTCL = os.path.join('.', 'build', r'run-hls.tcl')

        if self.args.override:
            if self.args.benchmark:
                self.args.cosim = True
            if self.args.cosim:
                self.args.csynth = True
            self.hls = HLS(
                hlsTCL,
                self.args.csim,
                self.args.csynth,
                self.args.cosim,
                self.args.benchmark)
        else:
            self.hls = HLS(
                hlsTCL,
                self.profile['b_csim'],
                self.profile['b_synth'],
                self.profile['b_cosim'],
                self.profile['b_cosim'])

        directivePath = os.path.join(
            self.testPath,
            r'directive_par%d.tcl' %
            (self.parEntries))
        self.hls.setParam(
            Parameters(
                self.op,
                self.logParEntries,
                self.parEntries,
                self.args.xpart))
        with self.opLock:
            self.hls.generateDirective(directivePath)

    def build(self):
        envD = dict()
        parEntriesList = ['spmv', 'tpmv']
        c_type = self.typeDict[self.op.dataType]
        self.hls.params.setDtype(c_type)
        envD["BLAS_dataType"] = c_type
        self.typeStr = 'd%s' % c_type
        if self.opClass == 'BLAS_L1':
            r_type = self.typeDict[self.op.rtype]
            self.typeStr = 'd%s_r%s' % (c_type, r_type)
            envD["BLAS_resDataType"] = r_type
            self.hls.params.setRtype(r_type)

        if self.op.name in parEntriesList:
            envD["BLAS_parEntries"] = "%d" % self.parEntries

        with self.makelock:
            make = Makefile(self.makefile, self.libPath)
            libPath = make.make(envD, self.testPath)
        self.lib = C.cdll.LoadLibrary(libPath)

    def runTest(self):
        paramTclPath = os.path.join(
            self.dataPath, r'parameters_%s_%s.tcl' %
            (self.op.sizeStr, self.typeStr))
        logfile = os.path.join(
            self.dataPath, r'logfile_%s_%s.log' %
            (self.op.sizeStr, self.typeStr))
        binFile = os.path.join(
            self.dataPath, 'TestBin_%s_%s.bin' %
            (self.op.sizeStr, self.typeStr))

        print("\n")
        print("=" * 64)
        dataList = [self.op.compute() for j in range(self.numToSim)]
        blas_gen = BLAS_GEN(self.lib)
        self.op.addInstr(blas_gen, dataList)
        blas_gen.write2BinFile(binFile)
        print(
            "\nOP %s: Data file %s has been generated sucessfully." %
            (self.op.name, binFile))
        del dataList
        self.hls.generateParam(paramTclPath)
        print("\nOP %s: Parameters in file %s." % (self.op.name, paramTclPath))
        print("\nOP %s: Log file %s" % (self.op.name, logfile))
        self.hls.execution(binFile, logfile, self.testPath)
        self.hls.checkLog(logfile)
        self.numSim += self.numToSim
        self.hls.benchmarking(logfile, self.op, self.reports)
        print("\nOP %s: Test of size %s passed." %
              (self.op.name, self.op.sizeStr))

    def run(self, makelock=threading.Lock()):
        self.makelock = makelock
        path = os.path.dirname(self.profilePath)
        self.hls.params.setPath(path)
        self.op.test(self)

    def writeReport(self, profile, flag='a+'):
        reportPath = os.path.join(self.testPath, 'report.rpt')
        if len(self.reports) == 0:
            raise OP_ERROR(
                "\nOP %s: Benchmark fails for op %s." %
                (self.op.name, self.op.name))
        list2File(
            self.reports,
            reportPath,
            addInfo="Profile path is %s.\n" %
            self.profilePath)
        return reportPath
