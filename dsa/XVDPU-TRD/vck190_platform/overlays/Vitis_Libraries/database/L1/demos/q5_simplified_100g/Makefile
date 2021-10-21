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
# makefile-generator v1.0.0
#

# ####################################### Help Section #####################################
.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86>"
	$(ECHO) "      Command to generate the design for specified Target and Shell."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""
	$(ECHO) "  make TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86>"
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make run TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86>"
	$(ECHO) "      Command to run application in emulation."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make build TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86>"
	$(ECHO) "      Command to build xclbin application."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make host HOST_ARCH=<aarch32/aarch64/x86>"
	$(ECHO) "      Command to build host application."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  NOTE: For SoC shells, ENV variable SYSROOT needs to be set."
	$(ECHO) ""

# ##################### Setting up default value of TARGET ##########################
TARGET ?= sw_emu

# ################### Setting up default value of DEVICE ##############################
DEVICE ?= xilinx_u280_xdma_201920_3

# ###################### Setting up default value of HOST_ARCH ####################### 
HOST_ARCH ?= x86
# #################### Checking if DEVICE in blacklist #############################
ifeq ($(findstring zc, $(DEVICE)), zc)
$(error [ERROR]: This project is not supported for $(DEVICE).)
endif

# #################### Checking if DEVICE in whitelist ############################
ifneq ($(findstring xilinx_u280_xdma_201920_3, $(DEVICE)), xilinx_u280_xdma_201920_3)
ifneq ($(findstring u250, $(DEVICE)), u250)
ifneq ($(findstring u200, $(DEVICE)), u200)
$(warning [WARNING]: This project has not been tested for $(DEVICE). It may or may not work.)
endif
endif
endif

# ######################## Setting up Project Variables #################################
MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
XF_PROJ_ROOT ?= $(shell bash -c 'export MK_PATH=$(MK_PATH); echo $${MK_PATH%/L1/*}')
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
HOST_SRCS += $(CUR_DIR)/host/test_q5s.cpp
HOST_SRCS += $(CUR_DIR)/host/xclhost.cpp
CXXFLAGS += -I$(CUR_DIR)/host
CXXFLAGS += -I$(CUR_DIR)/kernel
CXXFLAGS += -I$(XFLIB_DIR)/L1/include/hw
CXXFLAGS += -I$(XFLIB_DIR)/L3/include/sw
CXXFLAGS += -I$(XFLIB_DIR)/ext/xcl2
CXXFLAGS += -I$(XFLIB_DIR)/../utils/L1/include

ifeq ($(TARGET),sw_emu)
CXXFLAGS += -D SW_EMU_TEST
endif

ifeq ($(TARGET),hw_emu)
CXXFLAGS += -D HW_EMU_TEST
endif

# ######################### Host compiler global settings ############################
CXXFLAGS += -I$(XILINX_XRT)/include -I$(XILINX_VIVADO)/include -std=c++11 -O3 -Wall -Wno-unknown-pragmas -Wno-unused-label
LDFLAGS += -L$(XILINX_XRT)/lib -lOpenCL -lpthread -lrt -Wno-unused-label -Wno-narrowing -DVERBOSE
CXXFLAGS += -fmessage-length=0 -O3 -std=c++11
CXXFLAGS += -I$(CUR_DIR)/src/ 

ifeq ($(HOST_ARCH), x86)
LDFLAGS += -L$(XILINX_VIVADO)/lnx64/tools/fpo_v7_0 -Wl,--as-needed -lgmp -lmpfr -lIp_floating_point_v7_0_bitacc_cmodel
endif
ifneq (,$(shell echo $(XPLATFORM) | awk '/u250/'))
CXXFLAGS += -D USE_DDR
else ifneq (,$(shell echo $(XPLATFORM) | awk '/u200/'))
CXXFLAGS += -D USE_DDR
else ifneq (,$(shell echo $(XPLATFORM) | awk '/u280/'))
CXXFLAGS += -D USE_HBM
endif

# ################### Setting package and image directory #######################
PKG_OUT_DIR := package_$(TARGET)
ifneq (,$(findstring zc706, $(XDEVICE)))
K_IMAGE := $(SYSROOT)/../../uImage
else
K_IMAGE := $(SYSROOT)/../../Image
endif

EXE_NAME := test_q5s.exe
EXE_FILE := $(BUILD_DIR)/$(EXE_NAME)
HOST_ARGS :=  -xclbin $(BUILD_DIR)/q5simplified.xclbin -work $(CUR_DIR)/db_data -sf 1
PKG_HOST_ARGS = $(foreach args,$(HOST_ARGS),$(subst $(dir $(patsubst %/,%,$(args))),,$(args)))
ifeq ($(TARGET),$(filter $(TARGET),hw_emu))
HOST_ARGS :=  -xclbin $(BUILD_DIR)/q5simplified.xclbin -work $(CUR_DIR)/db_data -sf 1
PKG_HOST_ARGS = $(foreach args,$(HOST_ARGS),$(subst $(dir $(patsubst %/,%,$(args))),,$(args)))
endif
ifeq ($(TARGET),$(filter $(TARGET),hw))
HOST_ARGS :=  -xclbin $(BUILD_DIR)/q5simplified.xclbin -work $(CUR_DIR)/db_data -sf 100
PKG_HOST_ARGS = $(foreach args,$(HOST_ARGS),$(subst $(dir $(patsubst %/,%,$(args))),,$(args)))
endif

# ##################### Kernel compiler global settings ##########################
VPP_FLAGS += -t $(TARGET) --platform $(XPLATFORM) --save-temps --optimize 2
VPP_FLAGS += --hls.jobs 8
VPP_LDFLAGS += --vivado.synth.jobs 8 --vivado.impl.jobs 8
ifneq (,$(shell echo $(XPLATFORM) | awk '/u250/'))
VPP_FLAGS += --config $(CUR_DIR)/conn_u250.cfg
endif

ifneq (,$(shell echo $(XPLATFORM) | awk '/u200/'))
VPP_FLAGS += --config $(CUR_DIR)/conn_u200.cfg
endif

ifneq (,$(shell echo $(XPLATFORM) | awk '/u280/'))
VPP_FLAGS += --config $(CUR_DIR)/conn_u280.cfg
endif

VPP_FLAGS += -I$(XFLIB_DIR)/L1/include/hw
VPP_FLAGS += -I$(XFLIB_DIR)/L2/include

VPP_FLAGS += -I$(XFLIB_DIR)/L1/include/hw
VPP_FLAGS += -I$(CUR_DIR)/kernel
VPP_FLAGS += -I$(CUR_DIR)/host

q5simplified_VPP_FLAGS += --hls.clock 300000000:q5simplified
VPP_LDFLAGS_q5simplified += --kernel_frequency 300


# ############################ Declaring Binary Containers ##########################

ifeq ($(HOST_ARCH), x86)
BINARY_CONTAINERS += $(BUILD_DIR)/q5simplified.xclbin
BINARY_CONTAINERS_PKG += $(BUILD_DIR)/q5simplified.xclbin
else
BINARY_CONTAINERS += $(BUILD_DIR)/q5simplified_xo.xclbin
BINARY_CONTAINERS_PKG += $(BUILD_DIR)/q5simplified.xclbin
endif
BINARY_CONTAINER_q5simplified_OBJS += $(TEMP_DIR)/q5simplified.xo

# ######################### Setting Targets of Makefile ################################

.PHONY: all clean cleanall docs emconfig
all: check_vpp check_platform  $(EXE_FILE) $(BINARY_CONTAINERS) emconfig

.PHONY: host
host: check_xrt check_sysroot $(EXE_FILE)

.PHONY: xclbin
xclbin: check_vpp check_sysroot $(BINARY_CONTAINERS)

.PHONY: build
build: xclbin

# ################ Setting Rules for Binary Containers (Building Kernels) ################
$(TEMP_DIR)/q5simplified.xo: $(CUR_DIR)/kernel/q5simplified.cpp
	$(ECHO) "Compiling Kernel: q5simplified"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(q5simplified_VPP_FLAGS) $(VPP_FLAGS) -k q5simplified -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'


$(BINARY_CONTAINERS): $(BINARY_CONTAINER_q5simplified_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) -l $(VPP_FLAGS) --temp_dir $(BUILD_DIR) --report_dir $(BUILD_REPORT_DIR)/q5simplified $(VPP_LDFLAGS) $(VPP_LDFLAGS_q5simplified) -o '$@' $(+)


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
