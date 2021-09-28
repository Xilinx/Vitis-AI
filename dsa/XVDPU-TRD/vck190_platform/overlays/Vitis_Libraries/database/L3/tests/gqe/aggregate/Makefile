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
DEVICE ?= xilinx_u280_xdma_201920_3

# ###################### Setting up default value of HOST_ARCH ####################### 
HOST_ARCH ?= x86

# #################### Checking if DEVICE in blacklist #############################

# #################### Checking if DEVICE in whitelist ############################
ifneq ($(findstring xilinx_u280_xdma_201920_3, $(DEVICE)), xilinx_u280_xdma_201920_3)
$(error [ERROR]: This project is not supported for $(DEVICE).)
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
HOST_SRCS += $(CUR_DIR)/host/test_q1.cpp
HOST_SRCS += $(XFLIB_DIR)/L3/src/sw/gqe_table.cpp
HOST_SRCS += $(XFLIB_DIR)/L3/src/sw/gqe_aggr_config.cpp
HOST_SRCS += $(XFLIB_DIR)/L3/src/sw/gqe_ocl.cpp
HOST_SRCS += $(XFLIB_DIR)/L3/src/sw/gqe_aggr.cpp
CXXFLAGS += -I$(XFLIB_DIR)/L1/include/hw
CXXFLAGS += -I$(XFLIB_DIR)/L1/include/sw
CXXFLAGS += -I$(XFLIB_DIR)/L2/include/sw
CXXFLAGS += -I$(XFLIB_DIR)/L3/include/sw
CXXFLAGS += -I$(XFLIB_DIR)/L3/tests/gqe/common
CXXFLAGS += -I$(CUR_DIR)/host
CXXFLAGS += -I$(XFLIB_DIR)/../utils/L1/include
CXXFLAGS += -I$(XFLIB_DIR)/L1/include/hw
CXXFLAGS += -I$(XFLIB_DIR)/L3/include/sw
CXXFLAGS += -I$(XFLIB_DIR)/ext/xcl2

ifeq ($(TARGET),sw_emu)
CXXFLAGS += -D SW_EMU_TEST
endif

ifeq ($(TARGET),hw_emu)
CXXFLAGS += -D HW_EMU_TEST
endif

# ######################### Host compiler global settings ############################
CXXFLAGS += -I$(XILINX_XRT)/include -I$(XILINX_HLS)/include -std=c++11 -O3 -Wall -Wno-unknown-pragmas -Wno-unused-label
LDFLAGS += -L$(XILINX_XRT)/lib -lOpenCL -lpthread -lrt -Wno-unused-label -Wno-narrowing -DVERBOSE
CXXFLAGS += -fmessage-length=0 -O3 -D XDEVICE=$(XDEVICE)
CXXFLAGS += -I$(CUR_DIR)/src/ 

ifeq ($(HOST_ARCH), x86)
LDFLAGS += -L$(XILINX_HLS)/lnx64/tools/fpo_v7_0 -Wl,--as-needed -lgmp -lmpfr -lIp_floating_point_v7_0_bitacc_cmodel
endif

# ################### Setting package and image directory #######################

EXE_NAME := test_q1.exe
EXE_FILE := $(BUILD_DIR)/$(EXE_NAME)
ifeq ($(TARGET),$(filter $(TARGET),sw_emu))
HOST_ARGS :=  -xclbin $(BUILD_DIR)/gqe_aggr_part.xclbin -in $(CUR_DIR)/db_data/ -scale 10
endif
ifeq ($(TARGET),$(filter $(TARGET),hw_emu))
HOST_ARGS :=  -xclbin $(BUILD_DIR)/gqe_aggr_part.xclbin -in $(CUR_DIR)/db_data/ -scale 10000
endif
ifeq ($(TARGET),$(filter $(TARGET),hw))
HOST_ARGS :=  -xclbin $(BUILD_DIR)/gqe_aggr_part.xclbin -in $(CUR_DIR)/db_data/
endif

# ##################### Kernel compiler global settings ##########################
VPP_FLAGS += -t $(TARGET) --platform $(XPLATFORM) --save-temps --optimize 2
VPP_FLAGS += --hls.jobs 16
VPP_LDFLAGS += --vivado.synth.jobs 16 --vivado.impl.jobs 16
VPP_FLAGS += -I$(XFLIB_DIR)/L1/include/hw
VPP_FLAGS += -I$(XFLIB_DIR)/L2/include

VPP_FLAGS += -I$(XFLIB_DIR)/L1/include/hw
VPP_FLAGS += -I$(XFLIB_DIR)/L2/include/hw
VPP_FLAGS += -I$(XFLIB_DIR)/../utils/L1/include

gqeAggr_VPP_FLAGS += --hls.clock 250000000:gqeAggr
gqePart_VPP_FLAGS += --hls.clock 250000000:gqePart
VPP_LDFLAGS_gqe_aggr_part += --kernel_frequency 170


ifneq (,$(shell echo $(XPLATFORM) | awk '/u280/'))
VPP_LDFLAGS_gqe_aggr_part += --config $(CUR_DIR)/conn_u280.cfg --config $(CUR_DIR)/opts.cfg --vivado.prop run.impl_1.STEPS.OPT_DESIGN.TCL.POST=$(CUR_DIR)/post_opt.tcl
endif

# Kernel args

# ############################ Declaring Binary Containers ##########################

BINARY_CONTAINERS += $(BUILD_DIR)/gqe_aggr_part.xclbin
BINARY_CONTAINER_gqe_aggr_part_OBJS += $(TEMP_DIR)/gqeAggr.xo
BINARY_CONTAINER_gqe_aggr_part_OBJS += $(TEMP_DIR)/gqePart.xo

# ######################### Setting Targets of Makefile ################################

.PHONY: all clean cleanall docs emconfig
all: check_vpp check_platform check_sysroot $(EXE_FILE) $(BINARY_CONTAINERS) emconfig

.PHONY: host
host: check_xrt check_sysroot $(EXE_FILE)

.PHONY: xclbin
xclbin: check_vpp check_sysroot $(BINARY_CONTAINERS)

.PHONY: build
build: xclbin

# ################ Setting Rules for Binary Containers (Building Kernels) ################
$(TEMP_DIR)/gqeAggr.xo: $(XFLIB_DIR)/L2/src/hw/gqe_kernel_aggr_v2.cpp
	$(ECHO) "Compiling Kernel: gqeAggr"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(gqeAggr_VPP_FLAGS) $(VPP_FLAGS) -k gqeAggr -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' $^
$(TEMP_DIR)/gqePart.xo: $(XFLIB_DIR)/L2/src/hw/gqe_kernel_part_v2.cpp
	$(ECHO) "Compiling Kernel: gqePart"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(gqePart_VPP_FLAGS) $(VPP_FLAGS) -k gqePart -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' $^

ifneq (,$(shell echo $(XPLATFORM) | awk '/u280/'))

$(BUILD_DIR)/gqe_aggr_part.xclbin: $(BINARY_CONTAINER_gqe_aggr_part_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) -l $(VPP_FLAGS) --temp_dir $(TEMP_DIR) --report_dir $(BUILD_REPORT_DIR)/gqe_aggr_part $(VPP_LDFLAGS) $(VPP_LDFLAGS_gqe_aggr_part) -o '$@' $(+)
else 
$(BUILD_DIR)/gqe_aggr_part.xclbin: $(BINARY_CONTAINER_gqe_aggr_part_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) -l $(VPP_FLAGS) --temp_dir $(TEMP_DIR) --report_dir $(BUILD_REPORT_DIR)/gqe_aggr_part $(VPP_LDFLAGS) $(VPP_LDFLAGS_gqe_aggr_part) -o '$@' $(+)
endif


# ################# Setting Rules for Host (Building Host Executable) ################
$(EXE_FILE): $(HOST_SRCS) | check_xrt
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
	-$(RMDIR) $(BUILD_DIR)/*.xclbin _vimage *xclbin.run_summary qemu-memory-_* emulation/ _vimage/ pl* start_simulation.sh *.xclbin
	-$(RMDIR) _x_temp.*/_x.* _x_temp.*/.Xil _x_temp.*/profile_summary.* xo_* _x*
	-$(RMDIR) _x_temp.*/dltmp* _x_temp.*/kernel_info.dat _x_temp.*/*.log 
	-$(RMDIR) _x_temp.* 

cleanall: cleanh cleank
	-$(RMDIR) $(BUILD_DIR)  build_dir.* emconfig.json *.html $(TEMP_DIR) $(CUR_DIR)/reports *.csv *.run_summary $(CUR_DIR)/*.raw package_* run_script.sh .ipcache *.str
	-$(RMDIR) $(XFLIB_DIR)/common/data/*.xe2xd* $(XFLIB_DIR)/common/data/*.orig*

	-$(RMDIR) $(AIE_CONTAINERS) $(CUR_DIR)/Work $(CUR_DIR)/*.xpe $(CUR_DIR)/hw.o $(CUR_DIR)/*.xsa $(CUR_DIR)/xnwOut aiesimulator_output .AIE_SIM_CMD_LINE_OPTIONS

clean: cleanh
