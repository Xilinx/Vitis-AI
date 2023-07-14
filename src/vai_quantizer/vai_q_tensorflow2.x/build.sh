#!/usr/bin/env bash
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
# ==============================================================================
set -e
set -x

# cmake args
declare -a args

# parse options
options=$(getopt -a -n 'parse-options' -o h \
		 -l help,clean,fast,abi1,build_with_cpu,build_without_plugin,conda,type:,cmake-options: \
		 -- "$0" "$@")
[ $? -eq 0 ] || {
    echo "Failed to parse arguments! try --help"
    exit 1
}
eval set -- "$options"
while true; do
    case "$1" in
	-h | --help) show_help=true; break;;
	--clean) clean=true;;
	--fast) fast=true;;
	--abi1) abi1=true;;
	--build_with_cpu) build_with_cpu=true;;
	--build_without_plugin) build_without_plugin=true;;
	--conda) conda=true;;
	--type)
	    shift
	    case "$1" in
		release) build_type=Release;;
		debug) build_type=Debug;;
		*) echo "Invalid build type \"$1\"! try --help"; exit 1;;
	    esac
	    ;;
	--cmake-options) shift; args+=($1);;
	--) shift; break;;
    esac
    shift
done
export abi1
export clean
export show_help
export fast
export build_with_cpu
export build_type
export conda
export args
export build_without_plugin
TF2_ROOT=$(pwd)
echo ${TF2_ROOT}
cd ${TF2_ROOT}/tensorflow_model_optimization/python/core/quantization/keras/vitis
TF1_ROOT=$(pwd)
echo ${TF1_ROOT}
if ${build_without_plugin:=false}; then
    echo "Build without vai_q_tensorflow plugin"
else
    bash ./build_fix_neron.sh
    echo "Build with vai_q_tensorflow plugin"
fi


if [ $? -eq 0 ]
then
    cd ${TF2_ROOT}
    bash ./pip_pkg.sh ./pkgs/ --release
fi
