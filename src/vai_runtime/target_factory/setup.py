#Copyright 2019 Xilinx Inc.
#
#Licensed under the Apache License, Version 2.0(the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http:  // www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import subprocess
import sys
import shutil

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext

def get_shell_result(shell_cmd):
    proc = subprocess.Popen(shell_cmd, stdout=subprocess.PIPE, shell=True)
    outs, _ = proc.communicate(timeout=2)
    res = outs.strip().decode("utf-8")
    return res

class CMakeExtension(Extension):

    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        _os = get_shell_result("lsb_release -a | grep 'Distributor ID'| sed 's/^.*:\s*//'")
        os_version = get_shell_result("lsb_release -a | grep 'Release' | sed 's/^.*:\s*//'")
        arch = get_shell_result("uname -p")
        target_info = f"{_os}.{os_version}.{arch}.Debug"
        home = os.environ.get("HOME")
        install_prefix_default = f"{home}/.local/{target_info}"
        conda_prefix=os.environ.get("CONDA_PREFIX")


        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}", 
            f"-DBUILD_PYTHON=ON",
            f"-DCMAKE_BUILD_TYPE=Debug",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            f"-DCMAKE_INSTALL_PREFIX={install_prefix_default}",
            f"-DCMAKE_PREFIX_PATH={conda_prefix}",
        ]
        cwd = os.getcwd()
        if not os.path.exists(extdir):
            os.makedirs(extdir)
        os.chdir(extdir)
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args)
        subprocess.check_call(["cmake", "--build", extdir])
        os.chdir(cwd)

def clean_install_info():
    try:
        for dir_name in ["dist", "target_factory.egg-info"]:
                if os.path.exists(dir_name):
                    shutil.rmtree(dir_name)
    except Exception:
        print("Faile to do the cleaning, please clean up manually")



setup(
    name="target_factory",
    version="0.0.1",
    author="Xilinx",
    description="Target factory using pybind11 and CMake",
    packages=find_packages(),
    ext_modules=[CMakeExtension("target_factory")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.6",
)

clean_install_info()
