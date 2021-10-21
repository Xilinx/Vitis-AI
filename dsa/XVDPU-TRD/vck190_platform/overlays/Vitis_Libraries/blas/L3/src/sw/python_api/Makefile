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


# ####################################### Help Section #####################################
.PHONY: help

help::
	@echo ""
	@echo "Makefile Usage:"
	@echo ""
	@echo "  make api "
	@echo "      Command to build shared library file."
	@echo ""
	@echo "  make clean "
	@echo "      Command to remove the generated non-hardware files."
	@echo ""
	@echo "  make cleanall"
	@echo "      Command to remove all the generated files."
	@echo ""

	
ifndef XILINX_XRT
  XILINX_XRT = /opt/xilinx/xrt
  export XILINX_XRT
endif

.PHONY: check_xrt
check_xrt:
ifeq (,$(wildcard $(XILINX_XRT)/lib/libxilinxopencl.so))
	@echo "Cannot locate XRT installation. Please set XILINX_XRT variable." && false
endif

export PATH := $(XILINX_XRT)/bin:$(PATH)

ifeq (,$(LD_LIBRARY_PATH))
LD_LIBRARY_PATH := $(XILINX_XRT)/lib
else
LD_LIBRARY_PATH := $(XILINX_XRT)/lib:$(LD_LIBRARY_PATH)
endif

CXX := g++

# ##################### Setting up default value of TARGET ##########################
TARGET ?= hw

# ######################## Setting up Project Variables #################################
MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
XF_PROJ_ROOT ?= $(shell bash -c 'export MK_PATH=$(MK_PATH); echo $${MK_PATH%/L3/*}')
CUR_DIR := $(patsubst %/,%,$(dir $(MK_PATH)))
XFLIB_DIR = $(XF_PROJ_ROOT)

EXE_NAME = 
EXE_FILE =
HOST_ARGS =

OBJ_DIR := obj
BUILD_DIR := lib
OBJ_FILES = $(OBJ_DIR)/api.o
API_OUT = xfblas.so

# ######################### Host compiler global settings ############################
HOST_SRCS = $(XFLIB_DIR)/L3/src/sw/api.cpp
CXXFLAGS += -g -I$(XILINX_XRT)/include -I $(XFLIB_DIR)/L3/include/sw/xf_blas
CXXFLAGS += -O3 -std=c++14 -fPIC -fopenmp -Wextra -Wall -Wno-unused-parameter -Wno-unused-variable -Wno-unused-result
CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=0
LDFLAGS += -pthread -L$(XILINX_XRT)/lib -lz -lstdc++ -lrt -pthread -lxrt_core -lxrt_coreutil -ldl -luuid -lgomp

# ######################### Setting Targets of Makefile ################################

.PHONY: all
all: host 

.PHONY: host
host: api

api: check_xrt $(API_OUT)

$(OBJ_FILES):  $(HOST_SRCS)
	mkdir -p $(OBJ_DIR)
	$(CXX) -shared -c -o $@ $< $(CXXFLAGS)
	
$(API_OUT): $(OBJ_FILES)
	mkdir -p $(BUILD_DIR)
	$(CXX) -shared -o $(BUILD_DIR)/$@ $(OBJ_FILES) $(LDFLAGS)
	chmod a+rx lib

run: api 
check: api
build: api

# ################################# Cleaning Rules ##################################

clean:
ifneq (,$(OBJ_DIR))
	rm -rf $(CUR_DIR)/$(OBJ_DIR)*
endif
	rm -rf $(BUILD_DIR)

cleanall: clean 



