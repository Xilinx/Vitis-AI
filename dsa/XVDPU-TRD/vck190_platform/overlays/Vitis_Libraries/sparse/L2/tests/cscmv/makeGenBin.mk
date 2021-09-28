#
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
#
# -----------------------------------------------------------------------------

SPARSE_maxParamDdrBlocks=1024
SPARSE_maxParamHbmBlocks=512
SPARSE_paramOffset=1024
SPARSE_maxColMemBlocks = 128 
SPARSE_maxColParBlocks = 512 
SPARSE_maxRowBlocks = 512 
SPARSE_dataType = float 
SPARSE_indexType = uint32_t
SPARSE_logParEntries = 2
SPARSE_parEntries = 4
SPARSE_logParGroups = 0 
SPARSE_parGroups = 1
SPARSE_dataBits = 32
SPARSE_indexBits = 32
SPARSE_hbmMemBits = 256
SPARSE_ddrMemBits = 512
SPARSE_hbmChannels = 16 
SPARSE_hbmChannelMegaBytes = 256
SPARSE_printWidth = 6
SPARSE_pageSize = 4096
DEBUG_dumpData=0
SEQ_KERNEL=0

COMMON_DEFS = -D SPARSE_maxParamDdrBlocks=$(SPARSE_maxParamDdrBlocks) \
				-D SPARSE_maxParamHbmBlocks=$(SPARSE_maxParamHbmBlocks) \
				-D SPARSE_paramOffset=$(SPARSE_paramOffset) \
				-D SPARSE_maxColMemBlocks=$(SPARSE_maxColMemBlocks) \
				-D SPARSE_maxColParBlocks=$(SPARSE_maxColParBlocks) \
				-D SPARSE_maxRowBlocks=$(SPARSE_maxRowBlocks) \
				-D SPARSE_dataType=$(SPARSE_dataType) \
				-D SPARSE_indexType=$(SPARSE_indexType) \
				-D SPARSE_logParEntries=$(SPARSE_logParEntries) \
				-D SPARSE_parEntries=$(SPARSE_parEntries) \
				-D SPARSE_logParGroups=$(SPARSE_logParGroups) \
				-D SPARSE_parGroups=$(SPARSE_parGroups) \
				-D SPARSE_dataBits=$(SPARSE_dataBits) \
				-D SPARSE_indexBits=$(SPARSE_indexBits) \
				-D SPARSE_hbmMemBits=$(SPARSE_hbmMemBits) \
				-D SPARSE_ddrMemBits=$(SPARSE_ddrMemBits) \
				-D SPARSE_printWidth=$(SPARSE_printWidth) \
				-D SPARSE_pageSize=$(SPARSE_pageSize) \
				-D SPARSE_hbmChannels=$(SPARSE_hbmChannels) \
				-D SPARSE_hbmChannelMegaBytes=$(SPARSE_hbmChannelMegaBytes) \
				-D DEBUG_dumpData=$(DEBUG_dumpData) \
				-D SEQ_KERNEL=$(SEQ_KERNEL)

GEN_BIN_DEFS = $(COMMON_DEFS) 

# ##################### Setting up default value of TARGET ##########################
TARGET ?= hw_emu

# ################### Setting up default value of DEVICE ##############################
DEVICE ?= xilinx_u280_xdma_201920_1

# ###################### Setting up default value of HOST_ARCH ####################### 
HOST_ARCH ?= x86

# ######################## Setting up Project Variables #################################
MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
XF_PROJ_ROOT ?= $(shell bash -c 'export MK_PATH=$(MK_PATH); echo $${MK_PATH%/L2/*}')
CUR_DIR := $(patsubst %/,%,$(dir $(MK_PATH)))
XFLIB_DIR = $(XF_PROJ_ROOT)

# ######################### Include environment variables in utils.mk ####################
include ./utils.mk
XDEVICE := $(call device2xsa, $(DEVICE))
TEMP_DIR := _x_temp.$(TARGET).$(XDEVICE)
TEMP_REPORT_DIR := $(CUR_DIR)/reports/_x.$(TARGET).$(XDEVICE)
BUILD_DIR := build_dir.$(TARGET).$(XDEVICE)
BUILD_REPORT_DIR := $(CUR_DIR)/reports/_build.$(TARGET).$(XDEVICE)
export XCL_BINDIR = $(XCLBIN_DIR)

# ######################### Setting up Host Variables #########################
#Include Required Host Source Files
GEN_BIN_SRCS += $(XFLIB_DIR)/L2/src/sw/fp32/gen_bin.cpp

BOOST_INCLUDE = $(XILINX_VIVADO)/tps/boost_1_64_0
BOOST_LIB = $(XILINX_VIVADO)/lib/lnx64.o

GEN_BIN_EXE_NAME = gen_bin.exe
#
GEN_BIN_CXXFLAGS += -I$(XFLIB_DIR)/L2/include/sw/fp32 -I$(XFLIB_DIR)/L1/include/sw -I$(BOOST_INCLUDE) -I$(XILINX_VIVADO)/include

ifeq (no,$(DEBUG))
	GEN_BIN_CXXFLAGS += -O3
else
	GEN_BIN_CXXFLAGS += -g -O0
endif

GEN_BIN_CXXFLAGS += -std=c++11
GEN_BIN_CXXFLAGS += $(GEN_BIN_DEFS)

GEN_BIN_LDFLAGS += -lpthread -L$(XILINX_XRT)/lib \
				 -L$(BOOST_LIB) \
				 -lboost_iostreams -lz \
				 -lstdc++ \
				 -lrt \
	   		     -Wl,--rpath=$(XILINX_XRT)/lib

GEN_BIN_EXE_FILE := $(BUILD_DIR)/$(GEN_BIN_EXE_NAME)

$(GEN_BIN_EXE_FILE): $(GEN_BIN_SRCS)
	@echo -e "----\nCompiling gen_bin ..."
	mkdir -p $(BUILD_DIR)
	$(CXX) -o $@ $^ $(GEN_BIN_CXXFLAGS) $(GEN_BIN_LDFLAGS)

.PHONY: gen_bin

gen_bin: $(GEN_BIN_EXE_FILE)

MTX_FILE = $(XFLIB_DIR)/L2/tests/mtxFiles/bcsstm01.mtx
gen_data: gen_bin
	mkdir -p $(CUR_DIR)/data
	$(GEN_BIN_EXE_FILE) -config-write $(CUR_DIR)/data/app.bin $(CUR_DIR)/data/inVec.bin $(CUR_DIR)/data/outVec.bin $(MTX_FILE)
