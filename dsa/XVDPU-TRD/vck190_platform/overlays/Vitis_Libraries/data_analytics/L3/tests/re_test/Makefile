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
DEVICE ?= xilinx_u200_xdma_201830_2

# ###################### Setting up default value of HOST_ARCH ####################### 
HOST_ARCH ?= x86

# #################### Checking if DEVICE in blacklist #############################
ifeq ($(findstring zc, $(DEVICE)), zc)
$(error [ERROR]: This project is not supported for $(DEVICE).)
endif

# #################### Checking if DEVICE in whitelist ############################
ifneq ($(findstring u200, $(DEVICE)), u200)
ifneq ($(findstring u250, $(DEVICE)), u250)
ifneq ($(findstring u50, $(DEVICE)), u50)
ifneq ($(findstring aws-vu9p-f1, $(DEVICE)), aws-vu9p-f1)
$(warning [WARNING]: This project has not been tested for $(DEVICE). It may or may not work.)
endif
endif
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
HOST_SRCS += $(CUR_DIR)//host/main.cpp
HOST_SRCS += $(XFLIB_DIR)/L3/src/sw/text/regex_engine.cpp
HOST_SRCS += $(XFLIB_DIR)/L3/src/sw/text/reEngine_config.cpp
HOST_SRCS += $(XFLIB_DIR)/ext/xcl2/xcl2.cpp
CXXFLAGS += -I$(CUR_DIR)//kernel
CXXFLAGS += -I$(CUR_DIR)//re_compile/lib/include
CXXFLAGS += -I$(XFLIB_DIR)/L1/include/sw
CXXFLAGS += -I$(XFLIB_DIR)/L3/include/sw
CXXFLAGS += -I$(XFLIB_DIR)/ext/xcl2
CXXFLAGS += -I$(XFLIB_DIR)/../utils/L1/include
LDFLAGS += -L$(CUR_DIR)//re_compile/lib/lib
LDFLAGS += -lonig
LDFLAGS += -lxfcompile
LDFLAGS += 
CXXFLAGS += -I$(XFLIB_DIR)/L1/include/sw
CXXFLAGS += -I$(XFLIB_DIR)/L3/include/sw
CXXFLAGS += -I$(XFLIB_DIR)/ext/xcl2
CXXFLAGS += -I$(XFLIB_DIR)/../utils/L1/include

LDFLAGS +=


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

ifeq ($(HOST_ARCH), x86)
LDFLAGS += -L$(XILINX_HLS)/lnx64/tools/fpo_v7_0 -Wl,--as-needed -lgmp -lmpfr -lIp_floating_point_v7_0_bitacc_cmodel
endif

# ################### Setting package and image directory #######################

EXE_NAME := test.exe
EXE_FILE := $(BUILD_DIR)/$(EXE_NAME)
ifeq ($(TARGET),$(filter $(TARGET),sw_emu))
HOST_ARGS :=  -xclbin $(BUILD_DIR)/reEngineKernel.xclbin -in $(XFLIB_DIR)/L2/tests/text/reEngine/log_data/access_5k.log -out $(CUR_DIR)/check.log
LIBRARY_PATH =$(CUR_DIR)/re_compile/lib/lib:$(LD_LIBRARY_PATH)
endif
ifeq ($(TARGET),$(filter $(TARGET),hw_emu))
HOST_ARGS :=  -xclbin $(BUILD_DIR)/reEngineKernel.xclbin -in $(XFLIB_DIR)/L2/tests/text/reEngine/log_data/access_5k.log -lnm 10 -out $(CUR_DIR)/check.log
LIBRARY_PATH =$(CUR_DIR)/re_compile/lib/lib:$(LD_LIBRARY_PATH)
endif
ifeq ($(TARGET),$(filter $(TARGET),hw))
HOST_ARGS :=  -xclbin $(BUILD_DIR)/reEngineKernel.xclbin -in $(XFLIB_DIR)/L2/tests/text/reEngine/log_data/access_5k.log -out $(CUR_DIR)/check.log
LIBRARY_PATH =$(CUR_DIR)/re_compile/lib/lib:$(LD_LIBRARY_PATH)
endif

# ##################### Kernel compiler global settings ##########################
VPP_FLAGS += -t $(TARGET) --platform $(XPLATFORM) --save-temps --optimize 2
VPP_FLAGS += --hls.jobs 8
VPP_LDFLAGS += --vivado.synth.jobs 8 --vivado.impl.jobs 8
VPP_FLAGS += -I$(XFLIB_DIR)/L1/include/hw
VPP_FLAGS += -I$(XFLIB_DIR)/L2/include/hw

VPP_FLAGS += -I$(CUR_DIR)//kernel
VPP_FLAGS += -I$(XFLIB_DIR)/L2/include/hw
VPP_FLAGS += -I$(XFLIB_DIR)/L1/include/hw

reEngineKernel_VPP_FLAGS += --hls.clock 300000000:reEngineKernel
VPP_LDFLAGS_reEngineKernel += --kernel_frequency 300


ifneq (,$(shell echo $(XPLATFORM) | awk '/u200/'))
VPP_LDFLAGS_reEngineKernel += --config $(CUR_DIR)/conn_u200.cfg --config $(CUR_DIR)/opts.cfg
endif

ifneq (,$(shell echo $(XPLATFORM) | awk '/u250/'))
VPP_LDFLAGS_reEngineKernel += --config $(CUR_DIR)/conn_u250.cfg
endif

ifneq (,$(shell echo $(XPLATFORM) | awk '/u50/'))
VPP_LDFLAGS_reEngineKernel += --config $(CUR_DIR)/conn_u50.cfg
endif

# Kernel args

# ############################ Declaring Binary Containers ##########################

BINARY_CONTAINERS += $(BUILD_DIR)/reEngineKernel.xclbin
BINARY_CONTAINER_reEngineKernel_OBJS += $(TEMP_DIR)/reEngineKernel.xo

# ######################### Setting Targets of Makefile ################################
.PHONY: pre_build
pre_build:
	make -C $(CUR_DIR)/re_compile

.PHONY: all clean cleanall docs emconfig
ifeq ($(HOST_ARCH), x86)
all: pre_build check_version check_vpp check_platform check_xrt $(EXE_FILE) $(BINARY_CONTAINERS) emconfig
else
all: pre_build check_version check_vpp check_platform check_sysroot $(EXE_FILE) $(BINARY_CONTAINERS) emconfig  sd_card
endif

.PHONY: host
ifeq ($(HOST_ARCH), x86)
host: pre_build  check_xrt $(EXE_FILE)
else
host: pre_build  check_sysroot $(EXE_FILE)
endif

.PHONY: xclbin
ifeq ($(HOST_ARCH), x86)
xclbin: check_vpp pre_build check_xrt  $(BINARY_CONTAINERS)
else
xclbin: check_vpp pre_build check_sysroot  $(BINARY_CONTAINERS)
endif

.PHONY: build
build: pre_build xclbin

# ################ Setting Rules for Binary Containers (Building Kernels) ################
$(TEMP_DIR)/reEngineKernel.xo: $(CUR_DIR)//kernel/re_engine_kernel.cpp
	$(ECHO) "Compiling Kernel: reEngineKernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(reEngineKernel_VPP_FLAGS) $(VPP_FLAGS) -k reEngineKernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' $^

ifneq (,$(shell echo $(XPLATFORM) | awk '/u200/'))

$(BUILD_DIR)/reEngineKernel.xclbin: $(BINARY_CONTAINER_reEngineKernel_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) -l $(VPP_FLAGS) --temp_dir $(TEMP_DIR) --report_dir $(BUILD_REPORT_DIR)/reEngineKernel $(VPP_LDFLAGS) $(VPP_LDFLAGS_reEngineKernel) -o '$@' $(+)
else ifneq (,$(shell echo $(XPLATFORM) | awk '/u250/'))

$(BUILD_DIR)/reEngineKernel.xclbin: $(BINARY_CONTAINER_reEngineKernel_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) -l $(VPP_FLAGS) --temp_dir $(TEMP_DIR) --report_dir $(BUILD_REPORT_DIR)/reEngineKernel $(VPP_LDFLAGS) $(VPP_LDFLAGS_reEngineKernel) -o '$@' $(+)
else ifneq (,$(shell echo $(XPLATFORM) | awk '/u50/'))

$(BUILD_DIR)/reEngineKernel.xclbin: $(BINARY_CONTAINER_reEngineKernel_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) -l $(VPP_FLAGS) --temp_dir $(TEMP_DIR) --report_dir $(BUILD_REPORT_DIR)/reEngineKernel $(VPP_LDFLAGS) $(VPP_LDFLAGS_reEngineKernel) -o '$@' $(+)
else 
$(BUILD_DIR)/reEngineKernel.xclbin: $(BINARY_CONTAINER_reEngineKernel_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) -l $(VPP_FLAGS) --temp_dir $(TEMP_DIR) --report_dir $(BUILD_REPORT_DIR)/reEngineKernel $(VPP_LDFLAGS) $(VPP_LDFLAGS_reEngineKernel) -o '$@' $(+)
endif


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
	LD_LIBRARY_PATH=$(LIBRARY_PATH) XCL_EMULATION_MODE=$(TARGET) $(EXE_FILE) $(HOST_ARGS)
else
	LD_LIBRARY_PATH=$(LIBRARY_PATH) $(EXE_FILE) $(HOST_ARGS)
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

	-$(RMDIR) $(AIE_CONTAINERS) $(CUR_DIR)/Work $(CUR_DIR)/*.xpe $(CUR_DIR)/hw.o $(CUR_DIR)/*.xsa $(CUR_DIR)/xnwOut aiesimulator_output .AIE_SIM_CMD_LINE_OPTIONS

clean: cleanh
