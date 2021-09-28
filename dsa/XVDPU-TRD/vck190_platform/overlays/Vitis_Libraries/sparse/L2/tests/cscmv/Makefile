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
TARGET ?= hw_emu

# ################### Setting up default value of DEVICE ##############################
DEVICE ?= xilinx_u280_xdma_201920_3

# ###################### Setting up default value of HOST_ARCH ####################### 
HOST_ARCH ?= x86

# #################### Checking if DEVICE in blacklist #############################

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
HOST_SRCS += $(XFLIB_DIR)/L2/src/sw/fp32/cscmv.cpp
HOST_SRCS += $(XFLIB_DIR)/../blas/L2/src/xcl2/xcl2.cpp

CXXFLAGS += -DSPARSE_maxParamDdrBlocks=1024 -DSPARSE_maxParamHbmBlocks=512 -DSPARSE_paramOffset=1024 -DSPARSE_maxColMemBlocks=128 -DSPARSE_maxColParBlocks=512 -DSPARSE_maxRowBlocks=512 -DSPARSE_dataType=float -DSPARSE_indexType=uint32_t -DSPARSE_logParEntries=2 -DSPARSE_parEntries=4 -DSPARSE_logParGroups=0 -DSPARSE_parGroups=1 -DSPARSE_dataBits=32 -DSPARSE_indexBits=32 -DSPARSE_hbmMemBits=256 -DSPARSE_ddrMemBits=512 -DSPARSE_hbmChannels=16 -DSPARSE_hbmChannelMegaBytes=256 -DSPARSE_printWidth=6 -DSPARSE_pageSize=4096
CXXFLAGS += -I$(XFLIB_DIR)/L2/include/sw/fp32
CXXFLAGS += -I$(XFLIB_DIR)/L1/include/sw
CXXFLAGS += -I$(XFLIB_DIR)/../blas/L2/include/xcl2
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
CXXFLAGS += -fmessage-length=0 -O0 
CXXFLAGS += -I$(CUR_DIR)/src/ 

ifeq ($(HOST_ARCH), x86)
LDFLAGS += -L$(XILINX_VIVADO)/lnx64/tools/fpo_v7_0 -Wl,--as-needed -lgmp -lmpfr -lIp_floating_point_v7_0_bitacc_cmodel
endif

# ################### Setting package and image directory #######################

EXE_NAME := cscmv.exe
EXE_FILE := $(BUILD_DIR)/$(EXE_NAME)
HOST_ARGS :=  $(BUILD_DIR)/cscmv.xclbin $(CUR_DIR)/data/app.bin

# ##################### Kernel compiler global settings ##########################
VPP_FLAGS += -t $(TARGET) --platform $(XPLATFORM) --save-temps --optimize 2
VPP_FLAGS += --jobs 8
ifneq (,$(shell echo $(XPLATFORM) | awk '/u280/'))
VPP_FLAGS += --config $(CUR_DIR)/compile_opts.cfg
endif

VPP_FLAGS += -I$(XFLIB_DIR)/L1/include/hw
VPP_FLAGS += -I$(XFLIB_DIR)/L1/include/sw
VPP_FLAGS += -I$(XFLIB_DIR)/L2/include/hw
VPP_FLAGS += -I$(XFLIB_DIR)/L2/include/hw

VPP_FLAGS += -I$(XFLIB_DIR)/L1/include/hw
VPP_FLAGS += -I$(XFLIB_DIR)/L2/include/hw/fp32
VPP_FLAGS += -I$(XFLIB_DIR)/../blas/L1/include/hw

ifneq (,$(shell echo $(XPLATFORM) | awk '/u280/'))
VPP_LDFLAGS_cscmv += --kernel_frequency 250
else
loadColKernel_VPP_FLAGS += --hls.clock 250000000:loadColKernel
bufTransColVecKernel_VPP_FLAGS += --hls.clock 250000000:bufTransColVecKernel
bufTransNnzColKernel_VPP_FLAGS += --hls.clock 250000000:bufTransNnzColKernel
xBarColKernel_VPP_FLAGS += --hls.clock 250000000:xBarColKernel
cscRowKernel_VPP_FLAGS += --hls.clock 250000000:cscRowKernel
readWriteHbmKernel_VPP_FLAGS += --hls.clock 250000000:readWriteHbmKernel
VPP_LDFLAGS_cscmv += --kernel_frequency 250
endif

VPP_FLAGS += -DSPARSE_maxParamDdrBlocks=1024 -DSPARSE_maxParamHbmBlocks=512 -DSPARSE_paramOffset=1024 -DSPARSE_maxColMemBlocks=128 -DSPARSE_maxColParBlocks=512 -DSPARSE_maxRowBlocks=512 -DSPARSE_dataType=float -DSPARSE_indexType=uint32_t -DSPARSE_logParEntries=2 -DSPARSE_parEntries=4 -DSPARSE_logParGroups=0 -DSPARSE_parGroups=1 -DSPARSE_dataBits=32 -DSPARSE_indexBits=32 -DSPARSE_hbmMemBits=256 -DSPARSE_ddrMemBits=512 -DSPARSE_hbmChannels=16 -DSPARSE_hbmChannelMegaBytes=256 -DSPARSE_printWidth=6 -DSPARSE_pageSize=4096 

# Kernel linker flags
VPP_LDFLAGS_cscmv_temp := --config $(CUR_DIR)/opts.cfg

ifneq (,$(shell echo $(XPLATFORM) | awk '/u280/'))
VPP_LDFLAGS_cscmv_temp += --config $(CUR_DIR)/conn_u280.cfg
endif
VPP_LDFLAGS_cscmv += $(VPP_LDFLAGS_cscmv_temp)

# ############################ Declaring Binary Containers ##########################

BINARY_CONTAINERS += $(BUILD_DIR)/cscmv.xclbin
BINARY_CONTAINER_cscmv_OBJS += $(TEMP_DIR)/loadColKernel.xo
BINARY_CONTAINER_cscmv_OBJS += $(TEMP_DIR)/bufTransColVecKernel.xo
BINARY_CONTAINER_cscmv_OBJS += $(TEMP_DIR)/bufTransNnzColKernel.xo
BINARY_CONTAINER_cscmv_OBJS += $(TEMP_DIR)/xBarColKernel.xo
BINARY_CONTAINER_cscmv_OBJS += $(TEMP_DIR)/cscRowKernel.xo
BINARY_CONTAINER_cscmv_OBJS += $(TEMP_DIR)/readWriteHbmKernel.xo

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
$(TEMP_DIR)/loadColKernel.xo: $(XFLIB_DIR)/L2/src/hw/fp32/loadColKernel.cpp
	$(ECHO) "Compiling Kernel: loadColKernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(loadColKernel_VPP_FLAGS) $(VPP_FLAGS) -k loadColKernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/bufTransColVecKernel.xo: $(XFLIB_DIR)/L2/src/hw/fp32/bufTransColVecKernel.cpp
	$(ECHO) "Compiling Kernel: bufTransColVecKernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(bufTransColVecKernel_VPP_FLAGS) $(VPP_FLAGS) -k bufTransColVecKernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/bufTransNnzColKernel.xo: $(XFLIB_DIR)/L2/src/hw/fp32/bufTransNnzColKernel.cpp
	$(ECHO) "Compiling Kernel: bufTransNnzColKernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(bufTransNnzColKernel_VPP_FLAGS) $(VPP_FLAGS) -k bufTransNnzColKernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/xBarColKernel.xo: $(XFLIB_DIR)/L2/src/hw/fp32/xBarColKernel.cpp
	$(ECHO) "Compiling Kernel: xBarColKernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(xBarColKernel_VPP_FLAGS) $(VPP_FLAGS) -k xBarColKernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/cscRowKernel.xo: $(XFLIB_DIR)/L2/src/hw/fp32/cscRowKernel.cpp
	$(ECHO) "Compiling Kernel: cscRowKernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(cscRowKernel_VPP_FLAGS) $(VPP_FLAGS) -k cscRowKernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/readWriteHbmKernel.xo: $(XFLIB_DIR)/L2/src/hw/fp32/readWriteHbmKernel.cpp
	$(ECHO) "Compiling Kernel: readWriteHbmKernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(readWriteHbmKernel_VPP_FLAGS) $(VPP_FLAGS) -k readWriteHbmKernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'

ifneq (,$(shell echo $(XPLATFORM) | awk '/u280/'))
$(BUILD_DIR)/cscmv.xclbin: $(BINARY_CONTAINER_cscmv_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) -l $(VPP_FLAGS) --temp_dir $(BUILD_DIR) --report_dir $(BUILD_REPORT_DIR)/cscmv $(VPP_LDFLAGS) $(VPP_LDFLAGS_cscmv) -o '$@' $(+)
else 
$(BUILD_DIR)/cscmv.xclbin: $(BINARY_CONTAINER_cscmv_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) -l $(VPP_FLAGS) --temp_dir $(BUILD_DIR) --report_dir $(BUILD_REPORT_DIR)/cscmv $(VPP_LDFLAGS) $(VPP_LDFLAGS_cscmv) -o '$@' $(+)
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
	XCL_EMULATION_MODE=$(TARGET) make -f makeGenBin.mk gen_data
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
