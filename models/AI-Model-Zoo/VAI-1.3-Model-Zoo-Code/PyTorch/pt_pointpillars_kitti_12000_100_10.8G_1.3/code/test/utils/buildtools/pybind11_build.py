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


# This code is based on: https://github.com/nutonomy/second.pytorch.git
# 
# MIT License
# Copyright (c) 2018 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import shutil
import subprocess
import tempfile
from pathlib import Path

from utils.loader import import_file
from utils.find import find_cuda_device_arch
from .command import CUDALink, Gpp, Nvcc, compile_libraries, out


class Pybind11Link(Gpp):
    def __init__(self,
                 sources,
                 target,
                 std="c++11",
                 includes: list = None,
                 defines: dict = None,
                 cflags: str = None,
                 libraries: dict = None,
                 lflags: str = None,
                 extra_cflags: str = None,
                 extra_lflags: str = None,
                 build_directory: str = None):
        pb11_includes = subprocess.check_output(
            "python3 -m pybind11 --includes",
            shell=True).decode('utf8').strip("\n")
        cflags = cflags or '-fPIC -O3 '
        cflags += pb11_includes
        super().__init__(
            sources,
            target,
            std,
            includes,
            defines,
            cflags,
            link=True,
            libraries=libraries,
            lflags=lflags,
            extra_cflags=extra_cflags,
            extra_lflags=extra_lflags,
            build_directory=build_directory)


class Pybind11CUDALink(CUDALink):
    def __init__(self,
                 sources,
                 target,
                 std="c++11",
                 includes: list = None,
                 defines: dict = None,
                 cflags: str = None,
                 libraries: dict = None,
                 lflags: str = None,
                 extra_cflags: str = None,
                 extra_lflags: str = None,
                 build_directory: str = None):
        pb11_includes = subprocess.check_output(
            "python3 -m pybind11 --includes",
            shell=True).decode('utf8').strip("\n")
        cflags = cflags or '-fPIC -O3 '
        cflags += pb11_includes
        super().__init__(
            sources,
            target,
            std,
            includes,
            defines,
            cflags,
            libraries=libraries,
            lflags=lflags,
            extra_cflags=extra_cflags,
            extra_lflags=extra_lflags,
            build_directory=build_directory)


def load_pb11(sources,
              target,
              cwd='.',
              cuda=False,
              arch=None,
              num_workers=4,
              includes: list = None,
              build_directory=None,
              compiler="g++"):
    cmd_groups = []
    cmds = []
    outs = []
    main_sources = []
    if arch is None:
        arch = find_cuda_device_arch()

    for s in sources:
        s = str(s)
        if ".cu" in s or ".cu.cc" in s:
            assert cuda is True, "cuda must be true if contain cuda file"
            cmds.append(Nvcc(s, out(s), arch))
            outs.append(out(s))
        else:
            main_sources.append(s)

    if cuda is True and arch is None:
        raise ValueError("you must specify arch if sources contains"
                         " cuda files")
    cmd_groups.append(cmds)
    if cuda:
        cmd_groups.append(
            [Pybind11CUDALink(outs + main_sources, target, includes=includes)])
    else:
        cmd_groups.append(
            [Pybind11Link(outs + main_sources, target, includes=includes)])
    for cmds in cmd_groups:
        compile_libraries(
            cmds, cwd, num_workers=num_workers, compiler=compiler)

    return import_file(target, add_to_sys=False, disable_warning=True)
