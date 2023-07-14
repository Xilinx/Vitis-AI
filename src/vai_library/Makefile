#
# Copyright 2022-2023 Advanced Micro Devices Inc.
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

script_path := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))


all:
	bash -ex build_all.sh

## build latest dev branch(not include Vitis-AI-Library) 
dev:
	(cd ..; [ -d unilog ] && rm -rf unilog; [ -d xir ] && rm -rf xir; [ -d target_factory ] && rm -rf target_factory; [ -d vart ] && rm -rf vart; cd -; bash -ex build_all.sh)


clean_build:
	bash -ex build_all.sh --clean

unilog:
	(cd ..; [ -d unilog ] ||  git clone gits@xcdl190260:aisw/unilog; cd unilog; ${script_path}/cmake.sh --project unilog)

target:
	(cd ..; [ -d target_factory ] ||  git clone gits@xcdl190260:aisw/target_factory; cd target_factory; ${script_path}/cmake.sh --project target_factory)

xir:
	(cd ..; [ -d xir ] ||  git clone gits@xcdl190260:aisw/xir; cd xir; ${script_path}/cmake.sh --project xir --build-python)

vart:
	(cd ..; [ -d vart ] ||  git clone gits@xcdl190260:aisw/vart; cd vart; ${script_path}/cmake.sh --project vart --build-python --cmake-options='-DENABLE_DPU_RUNNER=ON' --cmake-options='-DENABLE_SIM_RUNNER=ON' --cmake-options='-DENABLE_CPU_RUNNER=ON')

vitis:
	(cd .; ${script_path}/cmake.sh --project Vitis-AI-Library --build-python --cmake-options="-DENABLE_OVERVIEW=ON")
