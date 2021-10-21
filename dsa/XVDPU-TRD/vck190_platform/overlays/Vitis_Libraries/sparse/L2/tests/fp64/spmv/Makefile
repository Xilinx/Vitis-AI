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
# makefile-generator v1.0.3
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
ifeq ($(findstring vck190, $(DEVICE)), vck190)
$(error [ERROR]: This project is not supported for $(DEVICE).)
endif

# #################### Checking if DEVICE in whitelist ############################
ifneq ($(findstring u280, $(DEVICE)), u280)
$(error [ERROR]: This project is not supported for $(DEVICE).)
endif

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
EMCONFIG_DIR := $(BUILD_DIR)
XCLBIN_DIR := $(CUR_DIR)/$(BUILD_DIR)
export XCL_BINDIR = $(XCLBIN_DIR)

# ######################### Setting up Host Variables #########################
#Include Required Host Source Files
HOST_SRCS += $(XFLIB_DIR)/L2/tests/fp64/spmv/main.cpp
HOST_SRCS += $(XFLIB_DIR)/../blas/L2/src/xcl2/xcl2.cpp

CXXFLAGS += -DSPARSE_dataType=double -DSPARSE_indexType=uint16_t -DSPARSE_parEntries=4 -DSPARSE_hbmChannels=16 -DSPARSE_hbmMemBits=256
CXXFLAGS += -I$(XFLIB_DIR)/..
CXXFLAGS += -I$(XFLIB_DIR)/../blas/L2/include/xcl2
CXXFLAGS += -I$(XFLIB_DIR)/L2/include/sw/fp64
CXXFLAGS += -I$(XFLIB_DIR)/L1/include/hw
CXXFLAGS += -I$(XFLIB_DIR)/L1/include/sw
CXXFLAGS += -I$(XFLIB_DIR)/L2/include/hw
CXXFLAGS += -I$(XFLIB_DIR)/L2/include/hw

ifeq ($(TARGET),sw_emu)
CXXFLAGS += -D SW_EMU_TEST
endif

ifeq ($(TARGET),hw_emu)
CXXFLAGS += -D HW_EMU_TEST
endif

# ######################### Host compiler global settings ############################
CXXFLAGS += -I$(XILINX_XRT)/include -I$(XILINX_VIVADO)/include -std=c++11 -O3 -Wall -Wno-unknown-pragmas -Wno-unused-label
LDFLAGS += -L$(XILINX_XRT)/lib -lOpenCL -lpthread -lrt -Wno-unused-label -Wno-narrowing -DVERBOSE
CXXFLAGS += -fmessage-length=0 -g -O0
CXXFLAGS += -I$(CUR_DIR)/src/ 

ifeq ($(HOST_ARCH), x86)
LDFLAGS += -L$(XILINX_VIVADO)/lnx64/tools/fpo_v7_0 -Wl,--as-needed -lgmp -lmpfr -lIp_floating_point_v7_0_bitacc_cmodel
endif

# ################### Setting package and image directory #######################

EXE_NAME := host.exe
EXE_FILE := $(BUILD_DIR)/$(EXE_NAME)
HOST_ARGS := $(BUILD_DIR)/spmv.xclbin $(CUR_DIR)/sig_dat $(CUR_DIR)/vec_dat  nasa2910

# ##################### Kernel compiler global settings ##########################
VPP_FLAGS += -t $(TARGET) --platform $(XPLATFORM) --save-temps --optimize 2
VPP_FLAGS += --jobs 8
VPP_FLAGS += -I$(XFLIB_DIR)/L1/include/hw
VPP_FLAGS += -I$(XFLIB_DIR)/L1/include/sw
VPP_FLAGS += -I$(XFLIB_DIR)/L2/include/hw
VPP_FLAGS += -I$(XFLIB_DIR)/L2/include/hw

VPP_FLAGS += -I$(XFLIB_DIR)/L1/include/hw/
VPP_FLAGS += -I$(XFLIB_DIR)/L2/include/hw/fp64
VPP_FLAGS += -I$(XFLIB_DIR)/..
VPP_FLAGS += -I$(XFLIB_DIR)/../blas/L1/include/hw

loadNnzKernel_VPP_FLAGS += --hls.clock 333000000:loadNnzKernel
loadParXkernel_VPP_FLAGS += --hls.clock 333000000:loadParXkernel
fwdParParamKernel_VPP_FLAGS += --hls.clock 333000000:fwdParParamKernel
moveXkernel_VPP_FLAGS += --hls.clock 333000000:moveXkernel
selMultXkernel_VPP_FLAGS += --hls.clock 333000000:selMultXkernel
loadRbParamKernel_VPP_FLAGS += --hls.clock 333000000:loadRbParamKernel
rowAccKernel_VPP_FLAGS += --hls.clock 333000000:rowAccKernel
assembleYkernel_VPP_FLAGS += --hls.clock 333000000:assembleYkernel
storeYkernel_VPP_FLAGS += --hls.clock 333000000:storeYkernel
VPP_LDFLAGS_spmv += --kernel_frequency 333

VPP_FLAGS += -DSPARSE_dataType=double -DSPARSE_indexType=uint16_t -DSPARSE_parEntries=4 -DSPARSE_hbmChannels=16 -DSPARSE_hbmMemBits=256 -DSPARSE_maxRows=4096 -DSPARSE_maxCols=4096 -DSPARSE_accLatency=8 -DSPARSE_dataBits=64 -DSPARSE_indexBits=16 

# Kernel linker flags
VPP_LDFLAGS_spmv_temp := --config $(CUR_DIR)/opts.cfg

ifneq (,$(shell echo $(XPLATFORM) | awk '/u280/'))
VPP_LDFLAGS_spmv_temp += --config $(CUR_DIR)/conn_u280.cfg
endif
VPP_LDFLAGS_spmv += $(VPP_LDFLAGS_spmv_temp)

# ############################ Declaring Binary Containers ##########################

BINARY_CONTAINERS += $(BUILD_DIR)/spmv.xclbin
BINARY_CONTAINER_spmv_OBJS += $(TEMP_DIR)/loadNnzKernel.xo
BINARY_CONTAINER_spmv_OBJS += $(TEMP_DIR)/loadParXkernel.xo
BINARY_CONTAINER_spmv_OBJS += $(TEMP_DIR)/fwdParParamKernel.xo
BINARY_CONTAINER_spmv_OBJS += $(TEMP_DIR)/moveXkernel.xo
BINARY_CONTAINER_spmv_OBJS += $(TEMP_DIR)/selMultXkernel.xo
BINARY_CONTAINER_spmv_OBJS += $(TEMP_DIR)/loadRbParamKernel.xo
BINARY_CONTAINER_spmv_OBJS += $(TEMP_DIR)/rowAccKernel.xo
BINARY_CONTAINER_spmv_OBJS += $(TEMP_DIR)/assembleYkernel.xo
BINARY_CONTAINER_spmv_OBJS += $(TEMP_DIR)/storeYkernel.xo

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
$(TEMP_DIR)/loadNnzKernel.xo: $(XFLIB_DIR)/L2/src/hw/fp64/loadNnzKernel.cpp
	$(ECHO) "Compiling Kernel: loadNnzKernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(loadNnzKernel_VPP_FLAGS) $(VPP_FLAGS) -k loadNnzKernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/loadParXkernel.xo: $(XFLIB_DIR)/L2/src/hw/fp64/loadParXkernel.cpp
	$(ECHO) "Compiling Kernel: loadParXkernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(loadParXkernel_VPP_FLAGS) $(VPP_FLAGS) -k loadParXkernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/fwdParParamKernel.xo: $(XFLIB_DIR)/L2/src/hw/fp64/fwdParParamKernel.cpp
	$(ECHO) "Compiling Kernel: fwdParParamKernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(fwdParParamKernel_VPP_FLAGS) $(VPP_FLAGS) -k fwdParParamKernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/moveXkernel.xo: $(XFLIB_DIR)/L2/src/hw/fp64/moveXkernel.cpp
	$(ECHO) "Compiling Kernel: moveXkernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(moveXkernel_VPP_FLAGS) $(VPP_FLAGS) -k moveXkernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/selMultXkernel.xo: $(XFLIB_DIR)/L2/src/hw/fp64/selMultXkernel.cpp
	$(ECHO) "Compiling Kernel: selMultXkernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(selMultXkernel_VPP_FLAGS) $(VPP_FLAGS) -k selMultXkernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/loadRbParamKernel.xo: $(XFLIB_DIR)/L2/src/hw/fp64/loadRbParamKernel.cpp
	$(ECHO) "Compiling Kernel: loadRbParamKernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(loadRbParamKernel_VPP_FLAGS) $(VPP_FLAGS) -k loadRbParamKernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/rowAccKernel.xo: $(XFLIB_DIR)/L2/src/hw/fp64/rowAccKernel.cpp
	$(ECHO) "Compiling Kernel: rowAccKernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(rowAccKernel_VPP_FLAGS) $(VPP_FLAGS) -k rowAccKernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/assembleYkernel.xo: $(XFLIB_DIR)/L2/src/hw/fp64/assembleYkernel.cpp
	$(ECHO) "Compiling Kernel: assembleYkernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(assembleYkernel_VPP_FLAGS) $(VPP_FLAGS) -k assembleYkernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/storeYkernel.xo: $(XFLIB_DIR)/L2/src/hw/fp64/storeYkernel.cpp
	$(ECHO) "Compiling Kernel: storeYkernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(storeYkernel_VPP_FLAGS) $(VPP_FLAGS) -k storeYkernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'

ifneq (,$(shell echo $(XPLATFORM) | awk '/u280/'))
$(BUILD_DIR)/spmv.xclbin: $(BINARY_CONTAINER_spmv_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) -l $(VPP_FLAGS) --temp_dir $(BUILD_DIR) --report_dir $(BUILD_REPORT_DIR)/spmv $(VPP_LDFLAGS) $(VPP_LDFLAGS_spmv) -o '$@' $(+)
else 
$(BUILD_DIR)/spmv.xclbin: $(BINARY_CONTAINER_spmv_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) -l $(VPP_FLAGS) --temp_dir $(BUILD_DIR) --report_dir $(BUILD_REPORT_DIR)/spmv $(VPP_LDFLAGS) $(VPP_LDFLAGS_spmv) -o '$@' $(+)
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
