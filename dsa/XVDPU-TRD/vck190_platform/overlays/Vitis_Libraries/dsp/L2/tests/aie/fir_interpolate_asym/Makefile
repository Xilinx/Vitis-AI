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
	$(ECHO) "  make all TARGET=<aiesim/x86sim/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch64>"
	$(ECHO) "      Command to generate the design for specified Target and Shell."
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""
	$(ECHO) "  make sd_card TARGET=<aiesim/x86sim/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch64>"
	$(ECHO) "      Command to prepare sd_card files."
	$(ECHO) "      This target is only used in embedded device."
	$(ECHO) ""
	$(ECHO) "  make run TARGET=<aiesim/x86sim/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch64>"
	$(ECHO) "      Command to run application in emulation or on board."
	$(ECHO) ""
	$(ECHO) "  make build TARGET=<aiesim/x86sim/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch64>"
	$(ECHO) "      Command to build xclbin application."
	$(ECHO) ""
	$(ECHO) "  make host HOST_ARCH=<aarch64>"
	$(ECHO) "      Command to build host application."
	$(ECHO) ""
	$(ECHO) "  NOTE: For embedded devices, e.g. zcu102/zcu104/vck190, env variable SYSROOT, EDGE_COMMON_SW and PERL need to be set first, and HOST_ARCH is either aarch32 or aarch64. For example,"
	$(ECHO) "       export SYSROOT=< path-to-platform-sysroot >"
	$(ECHO) "       export EDGE_COMMON_SW=< path-to-rootfs-and-Image-files >"
	$(ECHO) "       export PERL=<path-to-perl-installation-location >"
	$(ECHO) ""

# ##################### Setting up default value of TARGET ##########################
TARGET ?= aiesim

# ################### Setting up default value of DEVICE ##############################
DEVICE ?= xilinx_vck190_base_202110_1

# ###################### Setting up default value of HOST_ARCH ####################### 
HOST_ARCH ?= aarch64

# #################### Checking if DEVICE in whitelist ############################
ifneq ($(findstring vck190, $(DEVICE)), vck190)
$(warning [WARNING]: This project has not been tested for $(DEVICE). It may or may not work.)
endif

# ######################## Setting up Project Variables #################################
MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
XF_PROJ_ROOT ?= $(shell bash -c 'export MK_PATH=$(MK_PATH); echo $${MK_PATH%/L2/*}')
CUR_DIR := $(patsubst %/,%,$(dir $(MK_PATH)))
XFLIB_DIR = $(XF_PROJ_ROOT)

# ################### Setting environment variables for versal flow ##################
ifneq ($(wildcard $(XILINX_VITIS)/aietools/.*),)
XILINX_VITIS_AIETOOLS := $(XILINX_VITIS)/aietools
LDXRT := -ladf_api_xrt
else
XILINX_VITIS_AIETOOLS := $(XILINX_VITIS)/cardano
LDXRT := -lcardano_api_xrt -lxaiengine -lxrt_core
endif

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

# ####################### Setting environment variables for embedded #####################
ifneq ($(HOST_ARCH), x86)
SYSROOT := $(SYSROOT)
PERL ?= perl
SDCARD := sd_card
EMU_DIR := $(SDCARD)/data/emulation
endif
# Setting customized_params in aiecompiler
TAG ?= UUT
UUT_KERNEL ?= fir_interpolate_asym
REF_KERNEL ?= fir_interpolate_asym_ref
UUT_GRAPH ?= fir_interpolate_asym_graph
REF_GRAPH ?= fir_interpolate_asym_ref_graph
NITER ?= 16
NITER_UUT ?= $(NITER)
NITER_REF ?= $$(($(NITER_UUT) / 2))
INPUT_WINDOW_VSIZE ?= 256
UUT_INPUT_WINDOW_VSIZE ?= $(INPUT_WINDOW_VSIZE)
REF_INPUT_WINDOW_VSIZE ?= $$(( $(UUT_INPUT_WINDOW_VSIZE) * $(NITER_UUT) / $(NITER_REF)))
DATA_TYPE ?= cint16
COEFF_TYPE ?= int16
COEFF_SIZE ?= 2
FIR_LEN ?= 9
SHIFT ?= 16
ROUND_MODE ?= 0
CASC_LEN ?= 1
DUAL_IP ?= 0
USE_COEFF_RELOAD ?= 0
SYMMETRY_FACTOR ?= 1
INTERPOLATE_FACTOR ?= 3
DECIMATE_FACTOR ?= 1
NUM_OUTPUTS ?= 1
DIFF_TOLERANCE ?= 0.0025
DATA_SEED ?= 1
DATA_STIM_TYPE ?= 0
STIM_TYPE ?= 0
DEVICE_FILE ?= 4x4
COEFF_STIM_TYPE ?= 0
UUT_OUT_DIR ?= ./aiesimulator_output
REF_OUT_DIR_x86 ?= ./aiesimulator_ref_output
UUT_SIM_FILE ?= ./data/uut_output.txt
REF_SIM_FILE ?= ./data/ref_output.txt
UUT_SIM_FILE2 ?= ./data/uut_output_2.txt
REF_SIM_FILE2 ?= ./data/ref_output_2.txt
UUT_OUT_FILE ?= $(UUT_OUT_DIR)/$(UUT_SIM_FILE)
REF_OUT_FILE ?= $(REF_OUT_DIR_X86)/$(REF_SIM_FILE)
UUT_OUT_FILE2 ?= $(UUT_OUT_DIR)/$(UUT_SIM_FILE2)
REF_OUT_FILE2 ?= $(REF_OUT_DIR_X86)/$(REF_SIM_FILE2)
INPUT_FILE ?= data/input.txt
COEFF_FILE ?= data/coeff.txt
LOC_COEFF_FILE ?= ./data/coeff.txt
LOC_INPUT_FILE ?= ./data/input.txt
LOG_FILE ?= ./logs/log_$(UUT_FILE_SUFFIX).txt
STATUS_LOG_FILE ?= ./logs/status_$(UUT_FILE_SUFFIX).txt
STATUS_FILE ?= $(STATUS_LOG_FILE)
DIFF_OUT_FILE ?= diff_output.txt
UUT_TARGET_COMPILE_ARGS ?= --xlopt=1 -stacksize=$$(( $(FIR_LEN) * $(INTERPOLATE_FACTOR) * $(COEFF_SIZE) + 1536))  -Xchess=llvm.xargs="-std=c++2a" -Xchess=main:backend.mist2.xargs="+NOdra" -Xchess=main:noodle.optim.olbb=20 -Xchess=main:backend.mist2.pnll="off" 
REF_TARGET_COMPILE_ARGS ?= 
WORK_DIR ?= ./Work
UUT_WORK_DIR ?= ./Work
REF_WORK_DIR ?= ./Work
PREPROC_ARGS ?= 
USING_REF ?= 
USING_UUT ?= -DUSING_UUT=1
TAG_PREPROC_ARGS ?= -DUUT_GRAPH=$($(TAG)_GRAPH) -DOUTPUT_FILE=$($(TAG)_SIM_FILE) -DOUTPUT_FILE2=$($(TAG)_SIM_FILE2) $(PREPROC_ARGS)  -DINPUT_WINDOW_VSIZE=$($(TAG)_INPUT_WINDOW_VSIZE) -DNITER=$(NITER_$(TAG)) $(USING_$(TAG))

# ######################### Setting up Host Variables #########################
#Include Required Host Source Files
CXXFLAGS += -I$(XFLIB_DIR)/L1/include/hw
CXXFLAGS += -I$(XILINX_VITIS_AIETOOLS)/include
CXXFLAGS += -Wno-int-to-pointer-cast

# ############################# Setting AIETARGET ##############################
ifeq ($(TARGET),aiesim)
AIETARGET := hw
else ifeq ($(TARGET),x86sim)
AIETARGET := x86sim
else ifeq ($(TARGET),hw_emu)
AIETARGET := hw
else
AIETARGET := $(TARGET)
endif

AIE_CXXFLAGS += --target=$(AIETARGET) --platform=$(XPLATFORM) --pl-freq=1000 --use-phy-shim=false
AIE_CXXFLAGS += -I $(XILINX_VITIS_AIETOOLS)/include
AIE_CXXFLAGS += -I $(XFLIB_DIR)/L1/include/aie
AIE_CXXFLAGS += -I $(XFLIB_DIR)/L1/src/aie
AIE_CXXFLAGS += -I $(XFLIB_DIR)/L1/tests/aie/inc
AIE_CXXFLAGS += -I $(XFLIB_DIR)/L1/tests/aie/src
AIE_CXXFLAGS += -I $(XFLIB_DIR)/L2/include/aie
AIE_CXXFLAGS += -I $(XFLIB_DIR)/L2/tests/aie/common/inc
AIE_CXXFLAGS += -I $(CUR_DIR)
AIE_CXXFLAGS += --dataflow -v --device=$(DEVICE_FILE) $($(TAG)_TARGET_COMPILE_ARGS) --test-iterations=$(NITER_$(TAG)) -workdir=$($(TAG)_WORK_DIR) -Xpreproc="$(TAG_PREPROC_ARGS) -DCOEFF_FILE=$(LOC_COEFF_FILE) -DINPUT_FILE=$(LOC_INPUT_FILE) -DDATA_TYPE=$(DATA_TYPE) -DCOEFF_TYPE=$(COEFF_TYPE) -DFIR_LEN=$(FIR_LEN) -DSHIFT=$(SHIFT) -DROUND_MODE=$(ROUND_MODE) -DCASC_LEN=$(CASC_LEN) -DDUAL_IP=$(DUAL_IP) -DUSE_COEFF_RELOAD=$(USE_COEFF_RELOAD) -DSTIM_TYPE=$(COEFF_STIM_TYPE) -DINTERPOLATE_FACTOR=$(INTERPOLATE_FACTOR) -DNUM_OUTPUTS=$(NUM_OUTPUTS) "

ifeq ($(TARGET),sw_emu)
CXXFLAGS += -D SW_EMU_TEST
endif

ifeq ($(TARGET),hw_emu)
CXXFLAGS += -D HW_EMU_TEST
endif

# ######################### Host compiler global settings ############################
CXXFLAGS += -I$(SYSROOT)/usr/include/xrt -I$(XILINX_HLS)/include -std=c++14 -O3 -Wall -Wno-unknown-pragmas -Wno-unused-label
CXXFLAGS += -fmessage-length=0 
CXXFLAGS += -I$(CUR_DIR)/src/ 
LDFLAGS += -L$(SYSROOT)/usr/lib -L$(XILINX_VITIS_AIETOOLS)/lib/aarch64.o $(LDXRT) -lxilinxopencl -lpthread -lc -lrt -ldl -lcrypt -lstdc++ -lxrt++ -lxrt_coreutil


# ################### Setting package and image directory #######################
EMBEDDED_PACKAGE_OUT = package_$(TARGET)
ifneq (,$(findstring zc706, $(XDEVICE)))
K_IMAGE := $(EDGE_COMMON_SW)/uImage
else
K_IMAGE := $(EDGE_COMMON_SW)/Image
endif

EXE_NAME := EXE_FILE := $(BUILD_DIR)/$(EXE_NAME)

# ##################### Kernel compiler global settings ##########################
VPP_FLAGS += -t $(TARGET) --platform $(XPLATFORM) --save-temps --optimize 2
VPP_FLAGS += --hls.jobs 8
VPP_LDFLAGS += --vivado.synth.jobs 8 --vivado.impl.jobs 8
VPP_FLAGS += -I$(XFLIB_DIR)/L1/include/hw




# Kernel args

# ############################ Declaring Binary Containers ##########################
AIE_CONTAINERS := ./libadf.a
AIE_CONTAINER_OBJS := $(AIE_CONTAINERS)

# ######################### Setting Targets of Makefile ################################
.PHONY: pre_build
pre_build:
	tclsh $(XFLIB_DIR)/L2/tests/aie/common/scripts/gen_input.tcl $(LOC_INPUT_FILE) $(INPUT_WINDOW_VSIZE) $(NITER_UUT) $(DATA_SEED) $(DATA_STIM_TYPE)
	TARGET=x86sim UUT_KERNEL=fir_interpolate_asym_ref UUT_SIM_FILE=./data/ref_output.txt make x86sim TARGET=x86sim TAG=REF
	make cleanall

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
xclbin: check_vpp pre_build check_xrt $(AIE_CONTAINERS) $(BINARY_CONTAINERS)
else
xclbin: check_vpp pre_build check_sysroot $(AIE_CONTAINERS) $(BINARY_CONTAINERS)
endif

.PHONY: aiesim
aiesim: $(AIE_CONTAINERS)
	$(AIESIMULATOR) --pkg-dir ./Work  --input-dir ./ --output-dir ./ --profile

.PHONY: x86sim
x86sim: $(AIE_CONTAINERS)
	$(X86SIMULATOR) --pkg-dir=./Work --input-dir=./ --output-dir=./


# ################## Setting Rules for AIE (Building Kernels) ####################

$(AIE_CONTAINERS): $(XFLIB_DIR)/L2/tests/aie/fir_interpolate_asym/test.cpp
	$(ECHO) "Compiling: libsdf.a"
	mkdir -p $(BUILD_DIR)
	$(AIECXX) $(libsdf.a_AIE_CXXFLAGS) $(AIE_CXXFLAGS) $^



# ################# Setting Rules for Host (Building Host Executable) ################
ifeq ($(HOST_ARCH), x86)
$(EXE_FILE): $(AIE_CONTAINERS) $(HOST_SRCS) | check_xrt 
else
$(EXE_FILE): $(AIE_CONTAINERS) $(HOST_SRCS) | check_sysroot 
endif

	mkdir -p $(BUILD_DIR)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(XPLATFORM) --od $(EMCONFIG_DIR)


# ########## Setting Essential Checks and Building Rules for versal flow ############

BUILD_OBJS :=
ifeq ($(TARGET),$(filter $(TARGET), x86sim))
	BUILD_OBJS := $(AIE_CONTAINERS)
endif
ifeq ($(TARGET),$(filter $(TARGET), aiesim))
	BUILD_OBJS := $(AIE_CONTAINERS)
endif
ifeq ($(TARGET),$(filter $(TARGET), hw_emu))
	BUILD_OBJS := $(AIE_CONTAINERS) $(BINARY_CONTAINERS) $(EXE_FILE) sd_card
endif
ifeq ($(TARGET),$(filter $(TARGET), hw))
	BUILD_OBJS := $(AIE_CONTAINERS) $(BINARY_CONTAINERS) $(EXE_FILE) sd_card
endif

.PHONY: build
build: pre_build $(BUILD_OBJS)

# ############### Setting Essential Checks And Running Rules For Vitis Vitis ############
run: build
ifeq ($(TARGET),$(filter $(TARGET), x86sim))
	$(X86SIMULATOR) --pkg-dir=./Work --input-dir=./ --output-dir=./
	mkdir -p logs
	grep -ve '[XT]' ./data/uut_output.txt > ./logs/uut_output.txt
	grep -ve '[XT]' ./data/ref_output.txt > ./logs/ref_output.txt
	tclsh $(XFLIB_DIR)/L2/tests/aie/common/scripts/diff.tcl ./logs/uut_output.txt ./logs/ref_output.txt ./logs/diff.txt $(DIFF_TOLERANCE)
endif
ifeq ($(TARGET),$(filter $(TARGET), aiesim))
	$(AIESIMULATOR) --pkg-dir=./Work  --input-dir=./ --output-dir=./ --profile
	mkdir -p logs
	grep -ve '[XT]' ./data/uut_output.txt > ./logs/uut_output.txt
	grep -ve '[XT]' ./data/ref_output.txt > ./logs/ref_output.txt
	tclsh $(XFLIB_DIR)/L2/tests/aie/common/scripts/diff.tcl ./logs/uut_output.txt ./logs/ref_output.txt ./logs/diff.txt $(DIFF_TOLERANCE)
	tclsh $(XFLIB_DIR)/L2/tests/aie/common/scripts/get_stats.tcl $(DATA_TYPE) $(COEFF_TYPE) $(FIR_LEN) $(INPUT_WINDOW_VSIZE) $(CASC_LEN) $(INTERPOLATE_FACTOR) $(DECIMATE_FACTOR) $(SYMMETRY_FACTOR) $(DUAL_IP) $(USE_COEFF_RELOAD) $(NUM_OUTPUTS) $(STATUS_FILE) ./ "filter" $(NITER_UUT);
endif
ifeq ($(TARGET),$(filter $(TARGET), hw_emu))
	export XCL_EMULATION_MODE=$(TARGET)
	$(PERL) $(XF_PROJ_ROOT)/ext/make_utility/embedded_run_emulation.pl "./$(EMBEDDED_PACKAGE_OUT)/launch_$(TARGET).sh -noc-memory-config qemu-memory-_ddr@0x00000000 -no-reboot" "./run_script.sh" "$(RESULT_STRING)" "$(RUN_LEVEL)" | tee embedded_run.log
endif
ifeq ($(TARGET),$(filter $(TARGET), hw))
ifeq ($(HOST_ARCH), x86)
	$(EXE_FILE) $(HOST_ARGS)
else
	$(ECHO) "Please copy the content of sd_card folder and data to an SD Card and run on the board"
endif
endif

# ######################### Preparing sdcard folder ##############################
EMBEDDED_EXEC_SCRIPT = run_script.sh
SD_FILES += $(EMBEDDED_EXEC_SCRIPT)
SD_FILES += $(EXE_FILE)
SD_FILES += emconfig.json
SD_FILES += $(DATA_FILE)
SD_FILES += $(AIE_CONTAINERS)
SD_FILES_WITH_PREFIX = $(foreach sd_file,$(SD_FILES),--package.sd_file $(sd_file))
SD_DIRS_WITH_PREFIX = $(foreach sd_dir,$(DATA_DIR),--package.sd_dir $(sd_dir))
PACKAGE_FILES := $(BINARY_CONTAINERS)
PACKAGE_FILES += $(AIE_CONTAINERS)
sd_card: host xclbin emconfig
ifneq ($(HOST_ARCH), x86)
	@echo "Generating sd_card folder...."
	mkdir -p $(EMBEDDED_PACKAGE_OUT)
	rm -rf run_script.sh
	@echo 'export LD_LIBRARY_PATH=/mnt:/tmp:$$LD_LIBRARY_PATH' >> run_script.sh
ifeq ($(TARGET), $(filter $(TARGET),sw_emu hw_emu))
	@echo 'export XCL_EMULATION_MODE=$(TARGET)' >> run_script.sh
endif
	@echo 'export XILINX_VITIS=/mnt' >> run_script.sh
	@echo 'export XILINX_XRT=/usr' >> run_script.sh
	@echo 'if [ -f platform_desc.txt  ]; then' >> run_script.sh
	@echo '         cp platform_desc.txt /etc/xocl.txt' >> run_script.sh
	@echo 'fi'  >> run_script.sh
	emconfigutil --platform $(DEVICE) --nd 1;
	@echo './$(EXE_NAME) $(PKG_HOST_ARGS)' >> run_script.sh
	@echo 'return_code=$$?' >> run_script.sh
	@echo 'if [ $$return_code -ne 0 ]; then' >> run_script.sh
	@echo '         echo "ERROR: Embedded host run failed, RC=$$return_code"' >> run_script.sh
	@echo 'fi' >> run_script.sh
	@echo 'echo "INFO: Embedded host run completed."' >> run_script.sh
	@echo 'exit $$return_code' >> run_script.sh
	chmod a+rx run_script.sh
	$(VPP) -t $(TARGET) --platform $(DEVICE) -o $(BINARY_CONTAINERS_PKG) -p $(PACKAGE_FILES)  --package.out_dir $(EMBEDDED_PACKAGE_OUT) --package.rootfs $(EDGE_COMMON_SW)/rootfs.ext4 --package.kernel_image $(K_IMAGE) --package.boot_mode sd $(SD_FILES_WITH_PREFIX) $(SD_DIRS_WITH_PREFIX)
	@echo "### ***** sd_card generation done! ***** ###"
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
	-$(RMDIR) $(BUILD_DIR) sd_card* build_dir.* emconfig.json *.html $(TEMP_DIR) $(CUR_DIR)/reports *.csv *.run_summary $(CUR_DIR)/*.raw package_* run_script.sh .ipcache *.str
	-$(RMDIR) $(XFLIB_DIR)/common/data/*.xe2xd* $(XFLIB_DIR)/common/data/*.orig*

	-$(RMDIR) $(AIE_CONTAINERS) $(CUR_DIR)/Work $(CUR_DIR)/*.xpe $(CUR_DIR)/hw.o $(CUR_DIR)/*.xsa $(CUR_DIR)/xnwOut aiesimulator_output .AIE_SIM_CMD_LINE_OPTIONS

clean: cleanh
