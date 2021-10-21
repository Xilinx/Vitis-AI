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

import shlex
import subprocess
import pdb
import os
import sys
import re
import shutil


class HLS_ERROR(Exception):
    def __init__(self, message, logFile):
        self.message = message
        self.logFile = logFile


class Parameters:
    def __init__(self, op, logParEntries, parEntries, xpart):
        self.op = op
        self.logParEntries = logParEntries
        self.parEntries = parEntries
        self.rtype = 'uint32_t'
        self.dtype = 'uint32_t'
        self.xpart = xpart

    def setRtype(self, rtype):
        self.rtype = rtype

    def setDtype(self, dtype):
        self.dtype = dtype

    def setPath(self, path):
        self.path = path


class HLS:
    def __init__(self, tclPath, b_csim, b_syn, b_cosim, b_benchmark):

        self.tcl = tclPath
        self.csim = b_csim
        self.syn = b_syn
        self.cosim = b_cosim
        self.benchmark = b_benchmark

        self.error = False
        self.passCsim = False
        self.passSyn = False
        self.passCosim = False

    def execution(self, binFile, logFile, workDir='.', b_print=False):
        testDir = os.getcwd()
        commandLine = 'vitis_hls -f %s %s %s %s %s' % (
            os.path.abspath(self.tcl),
            os.path.abspath(testDir),
            self.paramFile,
            self.directive,
            binFile)
        if not b_print:
            print(
                "\nOP %s: vivado_hls stdout print is hidden." %
                self.params.op.name)
        args = shlex.split(commandLine)
        hls = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=workDir)
        with open(logFile, 'w', buffering=1) as f:
            f.write(commandLine)
            f.write(
                "\nOP %s: Working directory is %s\n" %
                (self.params.op.name, workDir))
            while True:
                line = hls.stdout.readline()
                if not line:
                    break
                line = line.decode('utf-8')
                if b_print:
                    print(line, end='')
                else:
                    print('.', end='')
                    sys.stdout.flush()

                    # ERROR checking
                    if line.find("ERROR") >= 0:
                        self.error = True
                    if line.find(r"C/RTL co-simulation finished: PASS") >= 0:
                        self.passCosim = True
                    if line.find("Finished generating all RTL models") >= 0:
                        self.passSyn = True
                    if line.find("CSim done with 0 errors") >= 0:
                        self.passCsim = True

                    # process checking
                    if line.find("CSIM finish") >= 0:
                        print('\nOP %s: CSIM finished.' % self.params.op.name)
                    elif line.find(r'C/RTL co-simulation finished') >= 0:
                        print('\nOP %s: COSIM finished.' % self.params.op.name)
                    elif line.find(r'C/RTL SIMULATION') >= 0:
                        print(
                            "\nOP %s: SYNTHESIS finished." %
                            self.params.op.name)

                f.write(line)

        print('\nOP %s: vivado_hls finished execution.' % self.params.op.name)

    def checkLog(self, logFile):
        with open(logFile, 'r') as f:
            content = f.read()
        if self.error:
            raise HLS_ERROR("HLS execution met errors.", logFile)
        elif self.csim and not self.passCsim:
            raise HLS_ERROR("Csim FAILED.", logFile)
        elif self.syn and not self.passSyn:
            raise HLS_ERROR("SYNTHESIS FAILED.", logFile)
        elif self.cosim and not self.passCosim:
            raise HLS_ERROR("C/RTL co-simulation FAILED.", logFile)

    def benchmarking(self, logFile, op, reportList):
        if not self.benchmark:
            return

        features = op.features()
        features['DataType'] = self.params.dtype

        dirname = os.path.dirname(logFile)
        with open(logFile, 'r') as f:
            content = f.read()

        regex = r'RTL Simulation : \d / \d \[n/a\] @ "(\d+)"'
        matches = re.findall(regex, content)
        timeList = [int(mat) for mat in matches]
        time = (timeList[-1] - timeList[0]) / (len(timeList) - 1) / 1e3
        features['RTL T[ns]'] = '%.1f' % time

        regex = r"(\d+\.?\d*)ns"
        match = re.search(regex, content)
        clock = float(match.group(1))
        features[r'clock[ns]'] = clock
        t_time = op.time(self.params.parEntries, clock)
        features[r'P.Entries'] = self.params.parEntries
        features[r'Est. T[ns]'] = '%.1f' % (t_time)
        features[r'Eff.'] = '%.1f%%' % (t_time / time * 100)
        features[r'Perf. [GOPS]'] = '%.3f' % (float(features['No.OPs']) / time)

        regex0 = r"Opening and resetting solution '([\w/-]+)'"
        regex1 = r"Creating and opening solution '([\w/-]+)'"
        match0 = re.search(regex0, content)
        match1 = re.search(regex1, content)
        if match0:
            solDir = match0.group(1)
        elif match1:
            solDir = match1.group(1)
        elif os.path.exists(os.path.join('.', 'prj_hls_vu9p', 'sol')):
            solDir = os.path.join('.', 'prj_hls_vu9p', 'sol')
        else:
            raise HLS_ERROR("Can't find the solution directory.")

        rpt_syn = os.path.join(solDir, 'syn', 'report', 'uut_top_csynth.rpt')

        with open(rpt_syn, 'r') as f:
            rpt = f.read()

        regex = r'\|Total\s*' + r'\|\s*(\d+)' * 5
        match = re.search(regex, rpt)
        regex = r'\|Available SLR\s*' + r'\|\s*(\d+)' * 5
        match_slr = re.search(regex, rpt)
        features['BRAM_18K'] = '%s(%.2f%%)' % (match.group(
            1), 100 * int(match.group(1)) / int(match_slr.group(1)))
        features['DSP48E  '] = '%s(%.2f%%)' % (match.group(
            2), 100 * int(match.group(2)) / int(match_slr.group(2)))
        features['FF      '] = '%s(%.2f%%)' % (match.group(
            3), 100 * int(match.group(3)) / int(match_slr.group(3)))
        features['LUT     '] = '%s(%.2f%%)' % (match.group(
            4), 100 * int(match.group(4)) / int(match_slr.group(4)))
        features['URAM    '] = '%s(%.2f%%)' % (match.group(
            5), 100 * int(match.group(5)) / int(match_slr.group(5)))

        reportList.append(features)

    def setParam(self, m):
        self.params = m

    def generateParam(self, fileparams):
        m = self.params
        self.paramFile = fileparams
        with open(self.paramFile, 'w') as f:
            f.write('array set opt {\n ')
            f.write('   path %s\n ' % m.path)
            ###########  TEST PARAMETERS  ##############
            f.write('   part    %s\n ' % m.xpart)
            f.write('   dataType %s\n ' % m.dtype)
            f.write('   resDataType %s\n ' % m.rtype)
            f.write('   logParEntries %d\n ' % m.logParEntries)
            f.write('   parEntries %d\n ' % m.parEntries)
            ###########  OP PARAMETERS  ##############

            m.op.paramTCL(f)

            ###########  HLS PARAMETERS  ##############
            if self.csim:
                f.write('   runCsim     1\n ')
            else:
                f.write('   runCsim     0\n ')
            if self.syn:
                f.write('   runRTLsynth   1\n ')
            else:
                f.write('   runRTLsynth   0\n ')
            if self.cosim:
                f.write('   runRTLsim     1\n ')
            else:
                f.write('   runRTLsim     0\n ')
            ###########  FIXED PARAMETERS  ##############
            f.write('   pageSizeBytes 4096\n ')
            f.write('   memWidthBytes 64\n ')
            f.write('   instrSizeBytes 8\n ')
            f.write('   maxNumInstrs 16\n ')
            f.write('   instrPageIdx 0\n ')
            f.write('   paramPageIdx 1\n ')
            f.write('   statsPageIdx 2\n ')
            f.write(' }\n ')

    def generateDirective(self, directivePath):
        self.directive = directivePath
        with open(self.directive, 'w') as f:
            for inface in self.params.op.interfaceList:
                #f.write('set_directive_interface -mode m_axi -depth %d "uut_top" %s\n'%(vs, inface))
                f.write(
                    'set_directive_array_partition -type cyclic -factor %d -dim 1 "uut_top" %s\n' %
                    (self.params.parEntries, inface))
