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
HOST_SRCS += $(XFLIB_DIR)/L2/src/sw/cgSolver/test_cgSolver_spmv_jacobi.cpp
HOST_SRCS += $(XFLIB_DIR)/../blas/L2/src/xcl2/xcl2.cpp

CXXFLAGS += -DHW_CLK=4e-9 -DCG_numTasks=1 -DCG_dataType=double -DCG_numChannels=16 -DCG_parEntries=4 -DCG_instrBytes=64 -DSPARSE_dataType=double -DSPARSE_indexType=uint16_t -DSPARSE_parEntries=4 -DSPARSE_hbmChannels=16 -DSPARSE_hbmMemBits=256
CXXFLAGS += -I$(XFLIB_DIR)/L1/include/
CXXFLAGS += -I$(XFLIB_DIR)/L2/include/sw/cgSolver
CXXFLAGS += -I$(XFLIB_DIR)/L2/include/common/cgSolver
CXXFLAGS += -I$(XFLIB_DIR)/L2/include/sw/
CXXFLAGS += -I$(XFLIB_DIR)/../blas/L1/tests/sw/include/
CXXFLAGS += -I$(XFLIB_DIR)/../blas/L2/include/xcl2
CXXFLAGS += -I$(XFLIB_DIR)/../sparse/L2/include/sw/fp64
CXXFLAGS += -I$(XFLIB_DIR)/..

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
HOST_ARGS := $(BUILD_DIR)/cgSolver.xclbin 32 1e-12 $(CUR_DIR)/sig_dat/  $(CUR_DIR)/vec_dat ted_B

# ##################### Kernel compiler global settings ##########################
VPP_FLAGS += -t $(TARGET) --platform $(XPLATFORM) --save-temps --optimize 2
VPP_FLAGS += --hls.jobs 8
VPP_LDFLAGS += --vivado.synth.jobs 8 --vivado.impl.jobs 8
ifneq (,$(shell echo $(XPLATFORM) | awk '/u280/'))
VPP_FLAGS += 
endif

VPP_FLAGS += -I$(XFLIB_DIR)/L1/include/
VPP_FLAGS += -I$(XFLIB_DIR)/L1/include/hw/
VPP_FLAGS += -I$(XFLIB_DIR)/L1/include/hw/cgSolver
VPP_FLAGS += -I$(XFLIB_DIR)/L2/include/hw/cgSolver
VPP_FLAGS += -I$(XFLIB_DIR)/L2/include/hw/cgSolver/spmv
VPP_FLAGS += -I$(XFLIB_DIR)/L2/include/common/cgSolver
VPP_FLAGS += -I$(XFLIB_DIR)/L2/include/hw
VPP_FLAGS += -I$(XFLIB_DIR)/../sparse/L1/include/hw
VPP_FLAGS += -I$(XFLIB_DIR)/../sparse/L2/include/hw/fp64
VPP_FLAGS += -I$(XFLIB_DIR)/..
VPP_FLAGS += -I$(XFLIB_DIR)/../blas/L2/include/memKernel/hw/xf_blas
VPP_FLAGS += -I$(XFLIB_DIR)/../blas/L1/include/hw
VPP_FLAGS += -I$(XFLIB_DIR)/L2/tests/cgSolver/krnl_cgSolver_spmv_jacobi_16ch/kernel

ifneq (,$(shell echo $(XPLATFORM) | awk '/u280/'))
VPP_LDFLAGS_cgSolver += --kernel_frequency 333
else
krnl_timer_VPP_FLAGS += --hls.clock 333000000:krnl_timer
krnl_duplicate_VPP_FLAGS += --hls.clock 333000000:krnl_duplicate
krnl_dupTk4_VPP_FLAGS += --hls.clock 333000000:krnl_dupTk4
krnl_loadArbParam_VPP_FLAGS += --hls.clock 333000000:krnl_loadArbParam
krnl_loadAval_VPP_FLAGS += --hls.clock 333000000:krnl_loadAval
krnl_loadPkApar_VPP_FLAGS += --hls.clock 333000000:krnl_loadPkApar
krnl_storeApk_VPP_FLAGS += --hls.clock 333000000:krnl_storeApk
krnl_update_xk_VPP_FLAGS += --hls.clock 333000000:krnl_update_xk
krnl_update_rk_jacobi_VPP_FLAGS += --hls.clock 333000000:krnl_update_rk_jacobi
krnl_update_pk_VPP_FLAGS += --hls.clock 333000000:krnl_update_pk
krnl_control_VPP_FLAGS += --hls.clock 333000000:krnl_control
fwdParParamKernel_VPP_FLAGS += --hls.clock 333000000:fwdParParamKernel
moveXkernel_VPP_FLAGS += --hls.clock 333000000:moveXkernel
selMultXkernel_VPP_FLAGS += --hls.clock 333000000:selMultXkernel
rowAccKernel_VPP_FLAGS += --hls.clock 333000000:rowAccKernel
assembleYkernel_VPP_FLAGS += --hls.clock 333000000:assembleYkernel
endif

VPP_FLAGS += -DCG_numTasks=1 -DCG_dataType=double -DCG_instrBytes=64 -DCG_tkStrWidth=8 -DCG_parEntries=4 -DCG_numChannels=16 -DCG_vecParEntries=4 -DSPARSE_dataType=double -DSPARSE_dataBits=64 -DSPARSE_parEntries=4 -DSPARSE_indexType=uint16_t -DSPARSE_indexBits=16 -DSPARSE_maxRows=4096 -DSPARSE_maxCols=4096 -DSPARSE_accLatency=8 -DSPARSE_hbmMemBits=256 -DSPARSE_hbmChannels=16 

# Kernel linker flags
VPP_LDFLAGS_cgSolver_temp := --config $(CUR_DIR)/opts.cfg

ifneq (,$(shell echo $(XPLATFORM) | awk '/u280/'))
VPP_LDFLAGS_cgSolver_temp += --config $(CUR_DIR)/conn_u280.cfg
endif
VPP_LDFLAGS_cgSolver += $(VPP_LDFLAGS_cgSolver_temp)

# ############################ Declaring Binary Containers ##########################

BINARY_CONTAINERS += $(BUILD_DIR)/cgSolver.xclbin
BINARY_CONTAINER_cgSolver_OBJS += $(TEMP_DIR)/krnl_timer.xo
BINARY_CONTAINER_cgSolver_OBJS += $(TEMP_DIR)/krnl_duplicate.xo
BINARY_CONTAINER_cgSolver_OBJS += $(TEMP_DIR)/krnl_dupTk4.xo
BINARY_CONTAINER_cgSolver_OBJS += $(TEMP_DIR)/krnl_loadArbParam.xo
BINARY_CONTAINER_cgSolver_OBJS += $(TEMP_DIR)/krnl_loadAval.xo
BINARY_CONTAINER_cgSolver_OBJS += $(TEMP_DIR)/krnl_loadPkApar.xo
BINARY_CONTAINER_cgSolver_OBJS += $(TEMP_DIR)/krnl_storeApk.xo
BINARY_CONTAINER_cgSolver_OBJS += $(TEMP_DIR)/krnl_update_xk.xo
BINARY_CONTAINER_cgSolver_OBJS += $(TEMP_DIR)/krnl_update_rk_jacobi.xo
BINARY_CONTAINER_cgSolver_OBJS += $(TEMP_DIR)/krnl_update_pk.xo
BINARY_CONTAINER_cgSolver_OBJS += $(TEMP_DIR)/krnl_control.xo
BINARY_CONTAINER_cgSolver_OBJS += $(TEMP_DIR)/fwdParParamKernel.xo
BINARY_CONTAINER_cgSolver_OBJS += $(TEMP_DIR)/moveXkernel.xo
BINARY_CONTAINER_cgSolver_OBJS += $(TEMP_DIR)/selMultXkernel.xo
BINARY_CONTAINER_cgSolver_OBJS += $(TEMP_DIR)/rowAccKernel.xo
BINARY_CONTAINER_cgSolver_OBJS += $(TEMP_DIR)/assembleYkernel.xo

# ######################### Setting Targets of Makefile ################################

-include ./params.mk
.PHONY: all clean cleanall docs emconfig
all: check_vpp check_platform check_sysroot $(EXE_FILE) $(BINARY_CONTAINERS) emconfig

.PHONY: host
host: check_xrt check_sysroot $(EXE_FILE)

.PHONY: xclbin
xclbin: check_vpp check_sysroot $(BINARY_CONTAINERS)

.PHONY: build
build: xclbin

# ################ Setting Rules for Binary Containers (Building Kernels) ################
$(TEMP_DIR)/krnl_timer.xo: $(XFLIB_DIR)/L2/src/hw/cgSolver/krnl_timer.cpp
	$(ECHO) "Compiling Kernel: krnl_timer"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(krnl_timer_VPP_FLAGS) $(VPP_FLAGS) -k krnl_timer -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/krnl_duplicate.xo: $(XFLIB_DIR)/L2/src/hw/cgSolver/krnl_duplicate.cpp
	$(ECHO) "Compiling Kernel: krnl_duplicate"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(krnl_duplicate_VPP_FLAGS) $(VPP_FLAGS) -k krnl_duplicate -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/krnl_dupTk4.xo: $(XFLIB_DIR)/L2/src/hw/cgSolver/spmv/krnl_dupTk4.cpp
	$(ECHO) "Compiling Kernel: krnl_dupTk4"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(krnl_dupTk4_VPP_FLAGS) $(VPP_FLAGS) -k krnl_dupTk4 -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/krnl_loadArbParam.xo: $(XFLIB_DIR)/L2/src/hw/cgSolver/spmv/krnl_loadArbParam.cpp
	$(ECHO) "Compiling Kernel: krnl_loadArbParam"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(krnl_loadArbParam_VPP_FLAGS) $(VPP_FLAGS) -k krnl_loadArbParam -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/krnl_loadAval.xo: $(XFLIB_DIR)/L2/src/hw/cgSolver/spmv/krnl_loadAval.cpp
	$(ECHO) "Compiling Kernel: krnl_loadAval"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(krnl_loadAval_VPP_FLAGS) $(VPP_FLAGS) -k krnl_loadAval -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/krnl_loadPkApar.xo: $(XFLIB_DIR)/L2/src/hw/cgSolver/spmv/krnl_loadPkApar.cpp
	$(ECHO) "Compiling Kernel: krnl_loadPkApar"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(krnl_loadPkApar_VPP_FLAGS) $(VPP_FLAGS) -k krnl_loadPkApar -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/krnl_storeApk.xo: $(XFLIB_DIR)/L2/src/hw/cgSolver/spmv/krnl_storeApk.cpp
	$(ECHO) "Compiling Kernel: krnl_storeApk"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(krnl_storeApk_VPP_FLAGS) $(VPP_FLAGS) -k krnl_storeApk -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/krnl_update_xk.xo: $(XFLIB_DIR)/L2/src/hw/cgSolver/krnl_update_xk.cpp
	$(ECHO) "Compiling Kernel: krnl_update_xk"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(krnl_update_xk_VPP_FLAGS) $(VPP_FLAGS) -k krnl_update_xk -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/krnl_update_rk_jacobi.xo: $(XFLIB_DIR)/L2/src/hw/cgSolver/krnl_update_rk_jacobi.cpp
	$(ECHO) "Compiling Kernel: krnl_update_rk_jacobi"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(krnl_update_rk_jacobi_VPP_FLAGS) $(VPP_FLAGS) -k krnl_update_rk_jacobi -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/krnl_update_pk.xo: $(XFLIB_DIR)/L2/src/hw/cgSolver/krnl_update_pk.cpp
	$(ECHO) "Compiling Kernel: krnl_update_pk"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(krnl_update_pk_VPP_FLAGS) $(VPP_FLAGS) -k krnl_update_pk -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/krnl_control.xo: $(XFLIB_DIR)/L2/src/hw/cgSolver/krnl_control.cpp
	$(ECHO) "Compiling Kernel: krnl_control"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(krnl_control_VPP_FLAGS) $(VPP_FLAGS) -k krnl_control -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/fwdParParamKernel.xo: $(XFLIB_DIR)/../sparse/L2/src/hw/fp64/fwdParParamKernel.cpp
	$(ECHO) "Compiling Kernel: fwdParParamKernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(fwdParParamKernel_VPP_FLAGS) $(VPP_FLAGS) -k fwdParParamKernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/moveXkernel.xo: $(XFLIB_DIR)/../sparse/L2/src/hw/fp64/moveXkernel.cpp
	$(ECHO) "Compiling Kernel: moveXkernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(moveXkernel_VPP_FLAGS) $(VPP_FLAGS) -k moveXkernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/selMultXkernel.xo: $(XFLIB_DIR)/../sparse/L2/src/hw/fp64/selMultXkernel.cpp
	$(ECHO) "Compiling Kernel: selMultXkernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(selMultXkernel_VPP_FLAGS) $(VPP_FLAGS) -k selMultXkernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/rowAccKernel.xo: $(XFLIB_DIR)/../sparse/L2/src/hw/fp64/rowAccKernel.cpp
	$(ECHO) "Compiling Kernel: rowAccKernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(rowAccKernel_VPP_FLAGS) $(VPP_FLAGS) -k rowAccKernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/assembleYkernel.xo: $(XFLIB_DIR)/../sparse/L2/src/hw/fp64/assembleYkernel.cpp
	$(ECHO) "Compiling Kernel: assembleYkernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(assembleYkernel_VPP_FLAGS) $(VPP_FLAGS) -k assembleYkernel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'

ifneq (,$(shell echo $(XPLATFORM) | awk '/u280/'))

$(BUILD_DIR)/cgSolver.xclbin: $(BINARY_CONTAINER_cgSolver_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) -l $(VPP_FLAGS) --temp_dir $(TEMP_DIR) --report_dir $(BUILD_REPORT_DIR)/cgSolver $(VPP_LDFLAGS) $(VPP_LDFLAGS_cgSolver) -o '$@' $(+)
else 
$(BUILD_DIR)/cgSolver.xclbin: $(BINARY_CONTAINER_cgSolver_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) -l $(VPP_FLAGS) --temp_dir $(TEMP_DIR) --report_dir $(BUILD_REPORT_DIR)/cgSolver $(VPP_LDFLAGS) $(VPP_LDFLAGS_cgSolver) -o '$@' $(+)
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
