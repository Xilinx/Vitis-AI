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
# makefile-generator v1.0.2
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
TARGET ?= hw

# ################### Setting up default value of DEVICE ##############################
DEVICE ?= xilinx_u250_gen3x16_xdma_3_1_202020_1

# ###################### Setting up default value of HOST_ARCH ####################### 
HOST_ARCH ?= x86

# #################### Checking if DEVICE in whitelist ############################
ifneq ($(findstring u250, $(DEVICE)), u250)
$(warning [WARNING]: This project has not been tested for $(DEVICE). It may or may not work.)
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
HOST_SRCS += $(XFLIB_DIR)/L3/benchmarks/gemm/gemm_bench.cpp

CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=0
CXXFLAGS += -I$(XFLIB_DIR)/L3/include/sw
CXXFLAGS += -I$(XFLIB_DIR)/L3/benchmarks/gemm
CXXFLAGS += -I$(XFLIB_DIR)/L3/benchmarks
CXXFLAGS += -I$(XFLIB_DIR)/L2/include/streamingKernel
LDFLAGS += -luuid -lxrt_coreutil
CXXFLAGS += -I$(XFLIB_DIR)/L1/include/hw

ifeq ($(TARGET),sw_emu)
CXXFLAGS += -D SW_EMU_TEST
endif

ifeq ($(TARGET),hw_emu)
CXXFLAGS += -D HW_EMU_TEST
endif

# ######################### Host compiler global settings ############################
CXXFLAGS += -I$(XILINX_XRT)/include -I$(XILINX_VIVADO)/include -std=c++14 -O3 -Wall -Wno-unknown-pragmas -Wno-unused-label
LDFLAGS += -L$(XILINX_XRT)/lib -lOpenCL -lpthread -lrt -Wno-unused-label -Wno-narrowing -DVERBOSE
CXXFLAGS += -fmessage-length=0 
CXXFLAGS += -I$(CUR_DIR)/src/ 

ifeq ($(HOST_ARCH), x86)
LDFLAGS += -L$(XILINX_VIVADO)/lnx64/tools/fpo_v7_0 -Wl,--as-needed -lgmp -lmpfr -lIp_floating_point_v7_0_bitacc_cmodel
endif

# ################### Setting package and image directory #######################
PKG_OUT_DIR := package_$(TARGET)

EXE_NAME := gemm_bench.exe
EXE_FILE := $(BUILD_DIR)/$(EXE_NAME)
HOST_ARGS := $(BUILD_DIR)/gemmKernel.xclbin $(BUILD_DIR)/config_info.dat
PKG_HOST_ARGS = $(foreach args,$(HOST_ARGS),$(subst $(dir $(patsubst %/,%,$(args))),,$(args)))

# ##################### Kernel compiler global settings ##########################
VPP_FLAGS += -t $(TARGET) --platform $(XPLATFORM) --save-temps --optimize 2
VPP_FLAGS += --jobs 8
VPP_FLAGS += -I$(XFLIB_DIR)/L1/include/hw

VPP_FLAGS += -I$(XFLIB_DIR)/L1/include/hw
VPP_FLAGS += -I$(XFLIB_DIR)/L1/include/hw/xf_blas
VPP_FLAGS += -I$(XFLIB_DIR)/L1/include/hw/xf_blas/gemm
VPP_FLAGS += -I$(XFLIB_DIR)/L1/include/hw/xf_blas/helpers/utils
VPP_FLAGS += -I$(XFLIB_DIR)/L2/include/streamingKernel
VPP_FLAGS += -I$(XFLIB_DIR)/L2/include/streamingKernel/hw/xf_blas/gemmStreamKernel/
VPP_FLAGS += -I$(XFLIB_DIR)/L2/include/streamingKernel/hw/xf_blas

gemmCPlusXKernel_VPP_FLAGS += --hls.clock 300000000:gemmCPlusXKernel
gemmTagsKernel_VPP_FLAGS += --hls.clock 300000000:gemmTagsKernel
gemmTimerKernel_VPP_FLAGS += --hls.clock 300000000:gemmTimerKernel
gemmMergeKernel_VPP_FLAGS += --hls.clock 300000000:gemmMergeKernel
gemmAddsKernel_VPP_FLAGS += --hls.clock 300000000:gemmAddsKernel
gemmMulsKernel_VPP_FLAGS += --hls.clock 300000000:gemmMulsKernel
gemmSystolicArrayKernel_VPP_FLAGS += --hls.clock 300000000:gemmSystolicArrayKernel
gemmLoadStoreKernel_VPP_FLAGS += --hls.clock 300000000:gemmLoadStoreKernel


# Kernel linker flags
VPP_LDFLAGS_gemmKernel_temp := --config opts.cfg

ifneq (,$(shell echo $(XPLATFORM) | awk '/u250/'))
VPP_LDFLAGS_gemmKernel_temp += --kernel_frequency 250 --config $(CUR_DIR)/conn_u250.cfg
endif
VPP_LDFLAGS_gemmKernel += $(VPP_LDFLAGS_gemmKernel_temp)

# ############################ Declaring Binary Containers ##########################

BINARY_CONTAINERS += $(BUILD_DIR)/gemmKernel.xclbin
BINARY_CONTAINERS_PKG += $(BUILD_DIR)/gemmKernel.xclbin
BINARY_CONTAINER_gemmKernel_OBJS += $(TEMP_DIR)/gemmCPlusXKernel.xo
BINARY_CONTAINER_gemmKernel_OBJS += $(TEMP_DIR)/gemmTagsKernel.xo
BINARY_CONTAINER_gemmKernel_OBJS += $(TEMP_DIR)/gemmTimerKernel.xo
BINARY_CONTAINER_gemmKernel_OBJS += $(TEMP_DIR)/gemmMergeKernel.xo
BINARY_CONTAINER_gemmKernel_OBJS += $(TEMP_DIR)/gemmAddsKernel.xo
BINARY_CONTAINER_gemmKernel_OBJS += $(TEMP_DIR)/gemmMulsKernel.xo
BINARY_CONTAINER_gemmKernel_OBJS += $(TEMP_DIR)/gemmSystolicArrayKernel.xo
BINARY_CONTAINER_gemmKernel_OBJS += $(TEMP_DIR)/gemmLoadStoreKernel.xo

# ######################### Setting Targets of Makefile ################################

-include params.makefile

.PHONY: all clean cleanall docs emconfig
all: check_vpp check_platform  $(EXE_FILE) $(BINARY_CONTAINERS) emconfig

.PHONY: host
host: check_xrt check_sysroot $(EXE_FILE)

.PHONY: xclbin
xclbin: check_vpp check_sysroot $(BINARY_CONTAINERS)

.PHONY: build
build: xclbin

# ################ Setting Rules for Binary Containers (Building Kernels) ################
$(TEMP_DIR)/gemmCPlusXKernel.xo: $(XFLIB_DIR)/L2/src/streamingKernel/hw/gemmCPlusX.cpp
	$(ECHO) "Compiling Kernel: gemmCPlusXKernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(gemmCPlusXKernel_VPP_FLAGS) $(VPP_FLAGS) -k gemmCPlusXKernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/gemmTagsKernel.xo: $(XFLIB_DIR)/L2/src/streamingKernel/hw/gemmTags.cpp
	$(ECHO) "Compiling Kernel: gemmTagsKernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(gemmTagsKernel_VPP_FLAGS) $(VPP_FLAGS) -k gemmTagsKernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/gemmTimerKernel.xo: $(XFLIB_DIR)/L2/src/streamingKernel/hw/gemmTimerKernel.cpp
	$(ECHO) "Compiling Kernel: gemmTimerKernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(gemmTimerKernel_VPP_FLAGS) $(VPP_FLAGS) -k gemmTimerKernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/gemmMergeKernel.xo: $(XFLIB_DIR)/L2/src/streamingKernel/hw/gemmMerge.cpp
	$(ECHO) "Compiling Kernel: gemmMergeKernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(gemmMergeKernel_VPP_FLAGS) $(VPP_FLAGS) -k gemmMergeKernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/gemmAddsKernel.xo: $(XFLIB_DIR)/L2/src/streamingKernel/hw/gemmAdds.cpp
	$(ECHO) "Compiling Kernel: gemmAddsKernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(gemmAddsKernel_VPP_FLAGS) $(VPP_FLAGS) -k gemmAddsKernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/gemmMulsKernel.xo: $(XFLIB_DIR)/L2/src/streamingKernel/hw/gemmMuls.cpp
	$(ECHO) "Compiling Kernel: gemmMulsKernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(gemmMulsKernel_VPP_FLAGS) $(VPP_FLAGS) -k gemmMulsKernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/gemmSystolicArrayKernel.xo: $(XFLIB_DIR)/L2/src/streamingKernel/hw/gemmSystolicArray.cpp
	$(ECHO) "Compiling Kernel: gemmSystolicArrayKernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(gemmSystolicArrayKernel_VPP_FLAGS) $(VPP_FLAGS) -k gemmSystolicArrayKernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/gemmLoadStoreKernel.xo: $(XFLIB_DIR)/L2/src/streamingKernel/hw/gemmLoadStore.cpp
	$(ECHO) "Compiling Kernel: gemmLoadStoreKernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(gemmLoadStoreKernel_VPP_FLAGS) $(VPP_FLAGS) -k gemmLoadStoreKernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'

ifneq (,$(shell echo $(XPLATFORM) | awk '/u250/'))
$(BUILD_DIR)/gemmKernel.xclbin: $(BINARY_CONTAINER_gemmKernel_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) -l $(VPP_FLAGS) --temp_dir $(BUILD_DIR) --report_dir $(BUILD_REPORT_DIR)/gemmKernel $(VPP_LDFLAGS) $(VPP_LDFLAGS_gemmKernel) -o '$@' $(+)
else 
$(BUILD_DIR)/gemmKernel.xclbin: $(BINARY_CONTAINER_gemmKernel_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) -l $(VPP_FLAGS) --temp_dir $(BUILD_DIR) --report_dir $(BUILD_REPORT_DIR)/gemmKernel $(VPP_LDFLAGS) $(VPP_LDFLAGS_gemmKernel) -o '$@' $(+)
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
	XCL_EMULATION_MODE=$(TARGET) make dump_config
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
