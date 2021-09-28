#
# Copyright 2019-2020 Xilinx, Inc.
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
#


# ######################## Setting up Project Variables #################################
MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
XF_PROJ_ROOT ?= $(shell bash -c 'export MK_PATH=$(MK_PATH); echo $${MK_PATH%/L3/*}')
CUR_DIR := $(patsubst %/,%,$(dir $(MK_PATH)))
XFLIB_DIR = $(XF_PROJ_ROOT)

pyGenMat=${XFLIB_DIR}/L2/src/python/genGemv.py
SRCS=src/gemv_benchmark.cu
TARGET=gemv_benchmark.exe

dataDir=./data/
p_n = 256
p_m = 512
BLAS_dataType = double


HOST_ARGS = ${p_m} ${p_n} ${dataDir}

CFLAGS= -std=c++11 -g -O0 -lcublas -I include/ \
        -Xcompiler \
        -DBENCHMARK 

${TARGET}: ${SRCS}
	nvcc ${CFLAGS} $^ -o $@

data_gen:
	@mkdir -p ${dataDir} 
	python3 ${pyGenMat} --p_m $(p_m) --p_n $(p_n) --path ${dataDir} --datatype ${BLAS_dataType}

run: ${TARGET} data_gen 
	@mkdir -p ${dataDir} 
	./${TARGET} ${HOST_ARGS} 

clean:
	@rm -rf ${TARGET}

cleanall:
	@rm -rf ${TARGET}
	@rm -rf ${dataDir}

