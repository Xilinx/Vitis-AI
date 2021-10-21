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
XF_PROJ_ROOT ?= $(shell bash -c 'export MK_PATH=$(MK_PATH); echo $${MK_PATH%plugin/*}')
XFLIB_DIR = $(XF_PROJ_ROOT)

HOST_ARCH := x86
RMDIR = rm -rf

ifndef XILINX_XRT
  XILINX_XRT = /opt/xilinx/xrt
  export XILINX_XRT
endif

TigerGraphPath=/home/tigergraph/tigergraph/

############################## Setting up Host Variables ##############################
CXXFLAGS += -I$(XFLIB_DIR)/ext/xcl2
CXXFLAGS += -I$(XFLIB_DIR)/L3/include

XILINX_XRM=/opt/xilinx/xrm

CXXFLAGS += -I$(XILINX_XRT)/include -I$(XILINX_VIVADO)/include -I$(XILINX_XRM)/include -std=c++11 -O3 -Wall -Wno-unknown-pragmas -Wno-unused-label -w
LDFLAGS += -L$(XILINX_XRT)/lib -lstdc++ -lxrt_core -ldl -luuid -lOpenCL -lpthread -lrt -Wno-unused-label -Wno-narrowing -DVERBOSE -L$(XILINX_XRM)/lib -lxrm
CXXFLAGS += -fmessage-length=0 -O3

ifneq ($(HOST_ARCH), x86)
	LDFLAGS += --sysroot=$(SYSROOT)
endif

CXXFLAGS += -D USE_HBM
CXXFLAGS += -D TIGERGRAPH_PATH=\"$(TigerGraphPath)\"

############################## Setting Rules for so (Building Dynamic Library) ##############################
libGRAPHL3WRAPPER = $(XFLIB_DIR)/plugin/tigergraph/QueryUdf/libgraphL3wrapper.so
libgraphL3wrapper : $(libGRAPHL3WRAPPER)
CPP_PATH += $(XFLIB_DIR)/ext/xcl2/xcl2.cpp
CPP_PATH += $(XFLIB_DIR)/L3/src/common.cpp
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
CPP_PATH += $(XFLIB_DIR)/plugin/tigergraph/L3_wrapper.cpp

HPP_PATH = $(XFLIB_DIR)/L3/include/task.hpp \
		   $(XFLIB_DIR)/L3/include/xf_graph_L3_handle.hpp

$(libGRAPHL3WRAPPER): ${CPP_PATH} ${HPP_PATH}
	/usr/bin/g++ -fPIC $(CXXFLAGS) -rdynamic -shared -o $@ ${CPP_PATH} $(LDFLAGS) -Wl,--export-dynamic

############################## Cleaning Rules ##############################
clean:
	-$(RMDIR) ${XFLIB_DIR}/plugin/tigergraph/QueryUdf/*.so 
