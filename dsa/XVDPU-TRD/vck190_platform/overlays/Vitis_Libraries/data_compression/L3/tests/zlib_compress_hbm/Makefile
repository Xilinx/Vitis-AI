#
# Copyright 2019-2021 Xilinx, Inc.
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
# makefile-generator v1.0.4
#

# ####################################### Help Section #####################################
.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<x86>"
	$(ECHO) "      Command to generate the design for specified Target and Shell."
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""
	$(ECHO) "  make run TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<x86>"
	$(ECHO) "      Command to run application in emulation or on board."
	$(ECHO) ""
	$(ECHO) "  make build TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<x86>"
	$(ECHO) "      Command to build xclbin application."
	$(ECHO) ""
	$(ECHO) "  make host HOST_ARCH=<x86>"
	$(ECHO) "      Command to build host application."
	$(ECHO) ""

# ##################### Setting up default value of TARGET ##########################
TARGET ?= sw_emu

# ################### Setting up default value of DEVICE ##############################
DEVICE ?= xilinx_u50_gen3x16_xdma_201920_3

# ###################### Setting up default value of HOST_ARCH ####################### 
HOST_ARCH ?= x86

# #################### Checking if DEVICE in blacklist #############################
ifeq ($(findstring u250, $(DEVICE)), u250)
$(error [ERROR]: This project is not supported for $(DEVICE).)
endif
ifeq ($(findstring vck, $(DEVICE)), vck)
$(error [ERROR]: This project is not supported for $(DEVICE).)
endif
ifeq ($(findstring u200, $(DEVICE)), u200)
$(error [ERROR]: This project is not supported for $(DEVICE).)
endif

# #################### Checking if DEVICE in whitelist ############################
ifneq ($(findstring u50, $(DEVICE)), u50)
ifneq ($(findstring u280, $(DEVICE)), u280)
$(warning [WARNING]: This project has not been tested for $(DEVICE). It may or may not work.)
endif
endif

# ######################## Setting up Project Variables #################################
MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
XF_PROJ_ROOT ?= $(shell bash -c 'export MK_PATH=$(MK_PATH); echo $${MK_PATH%/L3/*}')
CUR_DIR := $(patsubst %/,%,$(dir $(MK_PATH)))
XFLIB_DIR = $(XF_PROJ_ROOT)

# ######################### Include environment variables in utils.mk ####################
include ./utils.mk
XDEVICE := $(call device2xsa, $(DEVICE))
TEMP_DIR := _x_temp.$(TARGET).$(XDEVICE)
TEMP_REPORT_DIR := $(CUR_DIR)/reports/_x.$(TARGET).$(XDEVICE)
BUILD_DIR := build_dir.$(TARGET).$(XDEVICE)
BUILD_REPORT_DIR := $(CUR_DIR)/reports/_build.$(TARGET).$(XDEVICE)
EMCONFIG_DIR := $(BUILD_DIR)
XCLBIN_DIR := $(CUR_DIR)/$(BUILD_DIR)
export XCL_BINDIR = $(XCLBIN_DIR)

# ######################### Setting up Host Variables #########################
#Include Required Host Source Files
HOST_SRCS += $(CUR_DIR)/src/host.cpp
HOST_SRCS += $(XFLIB_DIR)/common/libs/compress/gzipOCLHost.cpp
HOST_SRCS += $(XFLIB_DIR)/common/libs/compress/gzipBase.cpp
HOST_SRCS += $(XFLIB_DIR)/common/libs/compress/gzipApp.cpp
HOST_SRCS += $(XFLIB_DIR)/common/libs/compress/compressApp.cpp
HOST_SRCS += $(XFLIB_DIR)/common/libs/xcl2/xcl2.cpp
HOST_SRCS += $(XFLIB_DIR)/common/libs/cmdparser/cmdlineparser.cpp
HOST_SRCS += $(XFLIB_DIR)/common/libs/logger/logger.cpp
HOST_SRCS += $(XFLIB_DIR)/common/thirdParty/zlib-1.2.7/crc32.c
HOST_SRCS += $(XFLIB_DIR)/common/thirdParty/zlib-1.2.7/adler32.c

CXXFLAGS += -DC_COMPUTE_UNIT=6
CXXFLAGS += -I$(XFLIB_DIR)/L3/include
CXXFLAGS += -I$(XFLIB_DIR)/L1/include/hw
CXXFLAGS += -I$(XFLIB_DIR)/common/thirdParty/zlib-1.2.7
CXXFLAGS += -I$(XFLIB_DIR)/common/libs/compress/
CXXFLAGS += -I$(XFLIB_DIR)/common/libs/xcl2
CXXFLAGS += -I$(XFLIB_DIR)/common/libs/cmdparser
CXXFLAGS += -I$(XFLIB_DIR)/common/libs/logger

ifeq ($(TARGET),sw_emu)
CXXFLAGS += -D SW_EMU_TEST
endif

ifeq ($(TARGET),hw_emu)
CXXFLAGS += -D HW_EMU_TEST
endif

# ######################### Host compiler global settings ############################
CXXFLAGS += -I$(XILINX_XRT)/include -I$(XILINX_HLS)/include -std=c++11 -O3 -Wall -Wno-unknown-pragmas -Wno-unused-label
LDFLAGS += -L$(XILINX_XRT)/lib -lOpenCL -lpthread -lrt -Wno-unused-label -Wno-narrowing -DVERBOSE
CXXFLAGS += -fmessage-length=0 -O3 
CXXFLAGS += -I$(CUR_DIR)/src/ 

# ################### Setting package and image directory #######################

EXE_NAME := xil_zlibc
EXE_FILE := $(BUILD_DIR)/$(EXE_NAME)
HOST_ARGS := -xbin $(BUILD_DIR)/compress.xclbin -c $(CUR_DIR)/sample.txt

# ##################### Kernel compiler global settings ##########################
VPP_FLAGS += -t $(TARGET) --platform $(XPLATFORM) --save-temps --optimize 2
VPP_FLAGS += --hls.jobs 8
VPP_LDFLAGS += --vivado.synth.jobs 8 --vivado.impl.jobs 8
VPP_FLAGS += -I$(XFLIB_DIR)/L1/include/hw
VPP_FLAGS += -I$(XFLIB_DIR)/L2/include
VPP_FLAGS += -I$(XFLIB_DIR)/L2/src

VPP_FLAGS += -I$(XFLIB_DIR)/L2/tests/src/
VPP_FLAGS += -I$(XFLIB_DIR)/../security/L1/include

VPP_FLAGS += --config $(CUR_DIR)/advanced.cfg

# Kernel linker flags
VPP_LDFLAGS_compress_temp := --kernel_frequency 250
VPP_LDFLAGS_compress += $(VPP_LDFLAGS_compress_temp)
# Adding config files to linker
VPP_LDFLAGS_compress+= --config auto_compress.cfg 

# Kernel args

# ############################ Declaring Binary Containers ##########################

BINARY_CONTAINERS += $(BUILD_DIR)/compress.xclbin
BINARY_CONTAINER_compress_OBJS += $(TEMP_DIR)/xilGzipCompBlock.xo

# ######################### Setting Targets of Makefile ################################

.PHONY: all clean cleanall docs emconfig
ifeq ($(HOST_ARCH), x86)
all:  check_version check_vpp check_platform check_xrt $(EXE_FILE) $(BINARY_CONTAINERS) emconfig
else
all:  check_version check_vpp check_platform check_sysroot $(EXE_FILE) $(BINARY_CONTAINERS) emconfig  sd_card
endif

.PHONY: host
ifeq ($(HOST_ARCH), x86)
host:   check_xrt $(EXE_FILE)
else
host:   check_sysroot $(EXE_FILE)
endif

.PHONY: xclbin
ifeq ($(HOST_ARCH), x86)
xclbin: check_vpp  check_xrt  $(BINARY_CONTAINERS)
else
xclbin: check_vpp  check_sysroot  $(BINARY_CONTAINERS)
endif

.PHONY: build
build: xclbin

# ################ Setting Rules for Binary Containers (Building Kernels) ################
$(TEMP_DIR)/xilGzipCompBlock.xo: $(XFLIB_DIR)/L2/src/gzip_compress_multicore_mm.cpp
	$(ECHO) "Compiling Kernel: xilGzipCompBlock"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(xilGzipCompBlock_VPP_FLAGS) $(VPP_FLAGS) -k xilGzipCompBlock -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' $^

$(BUILD_DIR)/compress.xclbin: $(BINARY_CONTAINER_compress_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) -l $(VPP_FLAGS) --temp_dir $(TEMP_DIR) --report_dir $(BUILD_REPORT_DIR)/compress $(VPP_LDFLAGS) $(VPP_LDFLAGS_compress) -o '$@' $(+)


# ################# Setting Rules for Host (Building Host Executable) ################
ifeq ($(HOST_ARCH), x86)
$(EXE_FILE):  $(HOST_SRCS) | check_xrt 
else
$(EXE_FILE):  $(HOST_SRCS) | check_sysroot 
endif

	mkdir -p $(BUILD_DIR)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(XPLATFORM) --od $(EMCONFIG_DIR)


# ###############Setting Essential Checks And Running Rules For Vitis Flow #############
run: all
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	$(CP) $(EMCONFIG_DIR)/emconfig.json .
	XCL_EMULATION_MODE=$(TARGET) $(EXE_FILE) $(HOST_ARGS)
else
	$(EXE_FILE) $(HOST_ARGS)
endif

# ################################# Cleaning Rules ##################################
cleanh:
	-$(RMDIR) $(EXE_FILE) vitis_* TempConfig system_estimate.xtxt *.rpt .run/
	-$(RMDIR) src/*.ll _xocc_* .Xil dltmp* xmltmp* *.log *.jou *.wcfg *.wdb sample_link.ini sample_compile.ini obj* bin* *.csv *.jpg *.jpeg *.png

cleank:
	-$(RMDIR) $(BUILD_DIR)/*.xclbin _vimage *xclbin.run_summary qemu-memory-_* emulation/ _vimage/ start_simulation.sh *.xclbin
	-$(RMDIR) _x_temp.*/_x.* _x_temp.*/.Xil _x_temp.*/profile_summary.* xo_* _x*
	-$(RMDIR) _x_temp.*/dltmp* _x_temp.*/kernel_info.dat _x_temp.*/*.log 
	-$(RMDIR) _x_temp.* 

cleanall: cleanh cleank
	-$(RMDIR) $(BUILD_DIR)  build_dir.* emconfig.json *.html $(TEMP_DIR) $(CUR_DIR)/reports *.csv *.run_summary $(CUR_DIR)/*.raw package_* run_script.sh .ipcache *.str
	-$(RMDIR) $(XFLIB_DIR)/common/data/*.xe2xd* $(XFLIB_DIR)/common/data/*.orig*
	-$(RMDIR) $(CUR_DIR)/sample.txt.* 
	-$(RMDIR) $(AIE_CONTAINERS) $(CUR_DIR)/Work $(CUR_DIR)/*.xpe $(CUR_DIR)/hw.o $(CUR_DIR)/*.xsa $(CUR_DIR)/xnwOut aiesimulator_output .AIE_SIM_CMD_LINE_OPTIONS

clean: cleanh
