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

# ######################### Setting up Host Variables #########################
#Include Required Host Source Files
HOST_SRCS += $(XFLIB_DIR)/L2/tests/aie/Filter2D/16bit_aie_8bit_pl_2core/host.cpp
HOST_SRCS += $(CUR_DIR)/Work/ps/c_rts/aie_control_xrt.cpp

CXXFLAGS += -D__PS_ENABLE_AIE__
CXXFLAGS += -I$(SYSROOT)/usr/include/xrt/
CXXFLAGS += -I$(SYSROOT)/usr/include/
CXXFLAGS += -I$(SYSROOT)/usr/include/opencv4
CXXFLAGS += -I$(XFLIB_DIR)/L1/include/aie
CXXFLAGS += -I$(CUR_DIR)
LDFLAGS += -L$(SYSROOT)/usr/lib/
LDFLAGS += -L$(XFLIB_DIR)/L1/lib/sw/aarch64-linux
LDFLAGS += -lsmartTilerStitcher
LDFLAGS += --sysroot=$(SYSROOT)
CXXFLAGS += -I$(XFLIB_DIR)/ext/xcl2/
CXXFLAGS += -I$(XFLIB_DIR)/L1/include

LDFLAGS += -lopencv_videoio -lopencv_imgcodecs -lopencv_core -lopencv_imgproc -lopencv_features2d -lopencv_flann -lopencv_video -lopencv_calib3d -lopencv_highgui

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

AIE_CXXFLAGS += --target=$(AIETARGET) --platform=$(XPLATFORM) --pl-freq=400 
AIE_CXXFLAGS += -I $(XILINX_VITIS_AIETOOLS)/include
AIE_CXXFLAGS += -I $(XFLIB_DIR)/L1/include/aie

ifeq ($(TARGET),sw_emu)
CXXFLAGS += -D SW_EMU_TEST
endif

ifeq ($(TARGET),hw_emu)
CXXFLAGS += -D HW_EMU_TEST
endif

# ######################### Host compiler global settings ############################
CXXFLAGS += -I$(SYSROOT)/usr/include/xrt -I$(XILINX_HLS)/include -std=c++14 -O3 -Wall -Wno-unknown-pragmas -Wno-unused-label
CXXFLAGS += -fmessage-length=0 --sysroot=$(SYSROOT)
CXXFLAGS += -I$(CUR_DIR)/src/ 
LDFLAGS += -L$(SYSROOT)/usr/lib -L$(XILINX_VITIS_AIETOOLS)/lib/aarch64.o $(LDXRT) -lxilinxopencl -lpthread -lc -lrt -ldl -lcrypt -lstdc++ -lxrt++ -lxrt_coreutil


# ################### Setting package and image directory #######################
EMBEDDED_PACKAGE_OUT = package_$(TARGET)
ifneq (,$(findstring zc706, $(XDEVICE)))
K_IMAGE := $(EDGE_COMMON_SW)/uImage
else
K_IMAGE := $(EDGE_COMMON_SW)/Image
endif

EXE_NAME := host.elf
EXE_FILE := $(BUILD_DIR)/$(EXE_NAME)
ifeq ($(TARGET),$(filter $(TARGET),hw_emu))
HOST_ARGS := kernel.xclbin HD.jpg

PKG_HOST_ARGS = $(foreach args,$(HOST_ARGS),$(subst $(dir $(patsubst %/,%,$(args))),,$(args)))
endif
ifeq ($(TARGET),$(filter $(TARGET),hw))
HOST_ARGS := kernel.xclbin 4k.jpg

PKG_HOST_ARGS = $(foreach args,$(HOST_ARGS),$(subst $(dir $(patsubst %/,%,$(args))),,$(args)))
endif

# ##################### Kernel compiler global settings ##########################
VPP_FLAGS += -t $(TARGET) --platform $(XPLATFORM) --save-temps --optimize 2
VPP_FLAGS += --hls.jobs 8
VPP_LDFLAGS += --vivado.synth.jobs 8 --vivado.impl.jobs 8
ifneq (,$(shell echo $(XPLATFORM) | awk '/vck190/'))
AIE_LDFLAGS += --clock.freqHz 400000000:Tiler_top_1
AIE_LDFLAGS += --clock.freqHz 400000000:stitcher_top_1
AIE_LDFLAGS += --clock.defaultTolerance 0.001
AIE_LDFLAGS += --config $(CUR_DIR)/system.cfg
AIE_LDFLAGS += $(XFLIB_DIR)/L1/lib/hw/8_16/tiler.xo
AIE_LDFLAGS += $(XFLIB_DIR)/L1/lib/hw/8_16/stitcher.xo
endif

ifneq (,$(shell echo $(XPLATFORM) | awk '/vck190/'))
VPP_FLAGS += 
endif

VPP_FLAGS += -I$(XFLIB_DIR)/L1/include/


ifneq ($(HOST_ARCH), x86)
VPP_LDFLAGS_kernel += --clock.defaultFreqHz 400000000
else
VPP_LDFLAGS_kernel += --kernel_frequency 400
endif


# Kernel args

# ############################ Declaring Binary Containers ##########################

BINARY_CONTAINERS += $(BUILD_DIR)/kernel_pkg.xclbin
BINARY_CONTAINERS_PKG += $(BUILD_DIR)/kernel.xclbin
AIE_CONTAINERS := ./libadf.a
AIE_CONTAINER_OBJS := $(AIE_CONTAINERS)

# ######################### Setting Targets of Makefile ################################
DATA_FILE += $(XFLIB_DIR)/data/4k.jpg
DATA_FILE += $(XFLIB_DIR)/data/HD.jpg

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
xclbin: check_vpp  check_xrt $(AIE_CONTAINERS) $(BINARY_CONTAINERS)
else
xclbin: check_vpp  check_sysroot $(AIE_CONTAINERS) $(BINARY_CONTAINERS)
endif

.PHONY: aiesim
aiesim: $(AIE_CONTAINERS)
	$(AIESIMULATOR) --pkg-dir ./Work  --input-dir ./ --output-dir ./ --profile

.PHONY: x86sim
x86sim: $(AIE_CONTAINERS)
	$(X86SIMULATOR) --pkg-dir=./Work --input-dir=./ --output-dir=./


# ################## Setting Rules for AIE (Building Kernels) ####################

$(AIE_CONTAINERS): $(XFLIB_DIR)/L2/tests/aie/Filter2D/16bit_aie_8bit_pl_2core/graph.cpp
	$(ECHO) "Compiling: libsdf.a"
	mkdir -p $(BUILD_DIR)
	$(AIECXX) $(libsdf.a_AIE_CXXFLAGS) $(AIE_CXXFLAGS) $^


$(BUILD_DIR)/kernel_pkg.xclbin: $(BINARY_CONTAINER_kernel_OBJS) $(AIE_CONTAINER_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) -l $(VPP_FLAGS) --temp_dir $(TEMP_DIR) --report_dir $(BUILD_REPORT_DIR)/kernel $(VPP_LDFLAGS) $(VPP_LDFLAGS_kernel) $(AIE_LDFLAGS) -o '$@' $^


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
build: $(BUILD_OBJS)

# ############### Setting Essential Checks And Running Rules For Vitis Vitis ############
run: build
ifeq ($(TARGET),$(filter $(TARGET), x86sim))
	$(X86SIMULATOR) --pkg-dir=./Work --input-dir=. --output-dir=.
endif
ifeq ($(TARGET),$(filter $(TARGET), aiesim))
	$(AIESIMULATOR) --pkg-dir=./Work  --input-dir=. --output-dir=. --profile
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
	$(VPP) -t $(TARGET) --platform $(DEVICE) -o $(BINARY_CONTAINERS_PKG) -p $(PACKAGE_FILES) --package.defer_aie_run --package.out_dir $(EMBEDDED_PACKAGE_OUT) --package.rootfs $(EDGE_COMMON_SW)/rootfs.ext4 --package.kernel_image $(K_IMAGE) --package.boot_mode sd $(SD_FILES_WITH_PREFIX) $(SD_DIRS_WITH_PREFIX)
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
