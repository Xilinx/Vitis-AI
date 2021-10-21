#
# Copyright 2020 Xilinx, Inc.
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
############################### Setting up Project Variables ##############################
MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
XF_PROJ_ROOT ?= $(shell bash -c 'export MK_PATH=$(MK_PATH); echo $${MK_PATH%L3/lib/*}')
XFLIB_DIR = $(XF_PROJ_ROOT)

HOST_ARCH := x86
RMDIR = rm -rf

ifndef XILINX_XRT
  XILINX_XRT = /opt/xilinx/xrt
  export XILINX_XRT
endif

#Checks for g++
CXX := g++
ifeq ($(HOST_ARCH), x86)
ifneq ($(shell expr $(shell g++ -dumpversion) \>= 5), 1)
ifndef XILINX_VIVADO
$(error [ERROR]: g++ version older. Please use 5.0 or above)
else
CXX := $(XILINX_VIVADO)/tps/lnx64/gcc-6.2.0/bin/g++
$(warning [WARNING]: g++ version older. Using g++ provided by the tool : $(CXX))
endif
endif
else ifeq ($(HOST_ARCH), aarch64)
CXX := $(XILINX_VITIS)/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-g++
else ifeq ($(HOST_ARCH), aarch32)
CXX := $(XILINX_VITIS)/gnu/aarch32/lin/gcc-arm-linux-gnueabi/bin/arm-linux-gnueabihf-g++
endif

############################## Setting up Host Variables ##############################
CXXFLAGS += -I$(XFLIB_DIR)/L2/include
CXXFLAGS += -I$(XFLIB_DIR)/ext/xcl2
CXXFLAGS += -I$(XFLIB_DIR)/L3/include
CXXFLAGS += -I$(XFLIB_DIR)/../utils/L1/include

XILINX_XRM=/opt/xilinx/xrm

CXXFLAGS += -I$(XILINX_XRT)/include -I$(XILINX_VIVADO)/include -I$(XILINX_XRM)/include -std=c++11 -O3 -Wall -Wno-unknown-pragmas -Wno-unused-label
LDFLAGS += -L$(XILINX_XRT)/lib -lOpenCL -lpthread -lrt -Wno-unused-label -Wno-narrowing -DVERBOSE -L$(XILINX_XRM)/lib -lxrm
CXXFLAGS += -fmessage-length=0 -O3
CXXFLAGS +=-I$(CUR_DIR)/src/ 

ifneq ($(HOST_ARCH), x86)
	LDFLAGS += --sysroot=$(SYSROOT)
endif

CXXFLAGS += -D USE_HBM

############################## Setting Rules for so (Building Dynamic Library) ##############################
libGRAPHL3 = $(XFLIB_DIR)/L3/lib/libgraphL3.so
libgraphL3 : $(libGRAPHL3)
CPP_PATH += $(XFLIB_DIR)/L3/src/common.cpp
CPP_PATH += $(XFLIB_DIR)/L3/src/op_twohop.cpp
CPP_PATH += $(XFLIB_DIR)/L3/src/op_sp.cpp
CPP_PATH += $(XFLIB_DIR)/L3/src/op_trianglecount.cpp
CPP_PATH += $(XFLIB_DIR)/L3/src/op_labelpropagation.cpp
CPP_PATH += $(XFLIB_DIR)/L3/src/op_bfs.cpp
CPP_PATH += $(XFLIB_DIR)/L3/src/op_wcc.cpp
CPP_PATH += $(XFLIB_DIR)/L3/src/op_scc.cpp
CPP_PATH += $(XFLIB_DIR)/L3/src/op_convertcsrcsc.cpp
CPP_PATH += $(XFLIB_DIR)/L3/src/op_similaritysparse.cpp
CPP_PATH += $(XFLIB_DIR)/L3/src/op_similaritydense.cpp
CPP_PATH += $(XFLIB_DIR)/L3/src/op_pagerank.cpp
CPP_PATH += $(XFLIB_DIR)/L3/src/xf_graph_L3_handle.cpp
CPP_PATH += $(XFLIB_DIR)/L3/src/xf_graph_L3.cpp

$(libGRAPHL3): ${CPP_PATH}
	$(CXX) -fPIC $(CXXFLAGS) -rdynamic -shared -o $@ $^ $(LDFLAGS) -w -Wl,--export-dynamic

############################## Cleaning Rules ##############################
clean:
	-$(RMDIR) ${XFLIB_DIR}/L3/lib/*.so 
