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
import shlex
import subprocess
import numpy as np
import pdb


class Makefile:
    def __init__(self, makefile, libpath):
        self.makefile = makefile
        self.libpath = libpath

    def make(self, envDict, makePath='out_test', rebuild=False):

        envStr = ' '.join(["%s=%s" % (key, envDict[key]) for key in envDict])
        nameStr = '_'.join(["%s-%s" % (key, envDict[key]) for key in envDict])
        self.target = os.path.join(makePath, 'blas_gen_bin.so')
        self.libName = os.path.join(
            self.libpath,
            r'blas_gen_bin_%s.so' %
            nameStr)

        if os.path.exists(self.libName):
            if not rebuild:
                return self.libName
            else:
                os.remove(self.libName)
        if os.path.exists(self.libName):
            os.remove(self.target)

        commandLine = r'make -f %s %s %s MK_DIR=%s' % (
            self.makefile, self.target, envStr, makePath)
        args = shlex.split(commandLine)
        subprocess.call(args)
        if os.path.exists(self.target):
            os.rename(self.target, self.libName)
            return self.libName
        else:
            raise Exception("ERROR: make shared library failure.")
