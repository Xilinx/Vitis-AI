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
# vitis makefile-generator v2.0.2

############################## Help Section ##############################
.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<hw/hw_emu/> DEVICE=<FPGA platform> HOST_ARCH=<aarch64 >"
	$(ECHO) "      Command to generate the design for specified Target and Shell."
	$(ECHO) "      By default, HOST_ARCH=aarch64. HOST_ARCH is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make sd_card TARGET=<hw/hw_emu/> DEVICE=<FPGA platform> HOST_ARCH=<aarch64 >"
	$(ECHO) "      Command to prepare sd_card files."
	$(ECHO) "      By default, HOST_ARCH=aarch64. HOST_ARCH is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make run TARGET=<hw/hw_emu/> DEVICE=<FPGA platform> HOST_ARCH=<aarch64 >"
	$(ECHO) "      Command to run application in emulation."
	$(ECHO) "      By default, HOST_ARCH=aarch64. HOST_ARCH required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make xclbin TARGET=<hw/hw_emu/> DEVICE=<FPGA platform> HOST_ARCH=<aarch64 >"
	$(ECHO) "      Command to build xclbin application."
	$(ECHO) "      By default, HOST_ARCH=aarch64. HOST_ARCH is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make host HOST_ARCH=<hw/hw_emu/>"
	$(ECHO) "      Command to build host application."
	$(ECHO) "      By default, HOST_ARCH=aarch64. HOST_ARCH is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  NOTE: For embedded devices, e.g. zcu102/zcu104/vck190, env variable SYSROOT, EDGE_COMMON_SW and PERL need to be set first, and HOST_ARCH is either aarch32 or aarch64. For example,"
	$(ECHO) "       export SYSROOT=< path-to-platform-sysroot >"
	$(ECHO) "       export EDGE_COMMON_SW=< path-to-rootfs-and-Image-files >"
	$(ECHO) "       export PERL=<path-to-perl-installation-location >"
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""

############################## Setting up Project Variables ##############################

MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
XF_PROJ_ROOT ?= $(shell bash -c 'export MK_PATH=$(MK_PATH); echo $${MK_PATH%/L2/*}')
CUR_DIR := $(patsubst %/,%,$(dir $(MK_PATH)))
XFLIB_DIR = $(XF_PROJ_ROOT)

# setting devault value
TARGET ?= hw_emu
DEVICE ?= xilinx_vck190_base_202110_1
HOST_ARCH ?= aarch64

# ###dddd################# Checking if DEVICE in whitelist ############################
PLATFORM_ALLOWLIST +=  vck190
PLATFORM_BLOCKLIST += 

include ./utils.mk

TEMP_DIR := _x_temp.$(TARGET).$(XDEVICE)
TEMP_REPORT_DIR := $(CUR_DIR)/reports/_x.$(TARGET).$(XDEVICE)
BUILD_DIR := build_dir.$(TARGET).$(XDEVICE)
BUILD_REPORT_DIR := $(CUR_DIR)/reports/_build.$(TARGET).$(XDEVICE)
EMCONFIG := $(BUILD_DIR)/emconfig.json
XCLBIN_DIR := $(CUR_DIR)/$(BUILD_DIR)
export XCL_BINDIR = $(XCLBIN_DIR)

EXE_FILE_DEPS :=
BINARY_CONTAINERS_DEPS :=
RUN_DEPS :=

# get global setting
ifeq ($(HOST_ARCH), x86)
CXXFLAGS += -fmessage-length=0 -I$(CUR_DIR)/src/ -I$(XILINX_XRT)/include -I$(XILINX_HLS)/include -std=c++14 -O3 -Wall -Wno-unknown-pragmas -Wno-unused-label 
LDFLAGS += -pthread -L$(XILINX_XRT)/lib -L$(XILINX_HLS)/lnx64/tools/fpo_v7_0  -Wl,--as-needed -lOpenCL -lxrt_coreutil -lgmp -lmpfr -lIp_floating_point_v7_0_bitacc_cmodel 
VPP_FLAGS += -t $(TARGET) --platform $(XPLATFORM) --save-temps 
VPP_LDFLAGS += --optimize 2 -R 2 
else ifeq ($(HOST_ARCH), aarch64)
CXXFLAGS += -I$(CUR_DIR)/src/ -fmessage-length=0 --sysroot=$(SYSROOT)  -I$(SYSROOT)/usr/include/xrt -I$(XILINX_HLS)/include -std=c++14 -O3 -Wall -Wno-unknown-pragmas -Wno-unused-label 
LDFLAGS += -pthread -L$(SYSROOT)/usr/lib -L$(XILINX_VITIS_AIETOOLS)/lib/aarch64.o -Wl,--as-needed -lxilinxopencl -lxrt_coreutil 
VPP_FLAGS += -t $(TARGET) --platform $(XPLATFORM) --save-temps 
VPP_LDFLAGS += --optimize 2 -R 2 
endif

# aie template
XILINX_VITIS_AIETOOLS ?=
XILINX_VITIS_AIELIBS ?=
ifneq ($(wildcard $(XILINX_VITIS)/aietools/.*),)
XILINX_VITIS_AIETOOLS := $(XILINX_VITIS)/aietools
XILINX_VITIS_AIELIBS := -ladf_api_xrt
else
XILINX_VITIS_AIETOOLS := $(XILINX_VITIS)/cardano
XILINX_VITIS_AIELIBS:= -lcardano_api_xrt -lxaiengine -lxrt_core
endif
CXXFLAGS += -I$(XILINX_VITIS_AIETOOLS)/include
LDFLAGS += $(XILINX_VITIS_AIELIBS)
# Setting customized_params in aiecompiler

############################ setting AIE Compiler ###########################
ifneq ($(filter aiesim hw_emu, $(TARGET)),)
AIETARGET := hw
else
AIETARGET := $(TARGET)
endif
AIE_CXXFLAGS += --target=$(AIETARGET) --platform=$(XPLATFORM) --pl-freq=300 -pl-axi-lite=true
AIE_CXXFLAGS += -I $(XILINX_VITIS_AIETOOLS)/include
AIE_CONTAINER = $(TEMP_DIR)/libadf.a
AIE_CXXFLAGS += -I $(XFLIB_DIR)/L1/include/aie

EXE_FILE_DEPS += $(AIE_CONTAINER)
BINARY_CONTAINERS_DEPS += $(AIE_CONTAINER)
########################## Setting up Host Variables ##########################
ifeq ($(TARGET),sw_emu)
CXXFLAGS += -D SW_EMU_TEST
endif
ifeq ($(TARGET),hw_emu)
CXXFLAGS += -D HW_EMU_TEST
endif

#Inclue Required Host Source Files
HOST_SRCS += $(XFLIB_DIR)/L2/tests/aie/pixelwise_mul/16bit_aie/host.cpp $(CUR_DIR)/Work/ps/c_rts/aie_control_xrt.cpp 
CXXFLAGS +=  -D __PS_ENABLE_AIE__
CXXFLAGS +=  -I $(SYSROOT)/usr/include/xrt/ -I $(XFLIB_DIR)/L1/include/aie/ -I $(CUR_DIR) -I $(XFLIB_DIR)/ext/xcl2/ -I $(XFLIB_DIR)/L1/include
CXXFLAGS += --sysroot=$(SYSROOT)
LDFLAGS +=  -L $(SYSROOT)/usr/lib/
LDFLAGS +=  -l opencv_videoio -l opencv_imgcodecs -l opencv_core -l opencv_imgproc -l opencv_features2d -l opencv_flann -l opencv_video -l opencv_calib3d -l opencv_highgui
LDFLAGS += --sysroot=$(SYSROOT)

EXE_NAME := host.elf
EXE_FILE := $(BUILD_DIR)/$(EXE_NAME)
EXE_FILE_DEPS += $(HOST_SRCS) $(EXE_FILE_DEPS)

########################## Kernel compiler global settings ##########################
ifneq (,$(shell echo $(XPLATFORM) | awk '/vck190/'))
VPP_FLAGS +=   
VPP_FLAGS +=  -I $(XFLIB_DIR)/L1/include/
VPP_LDFLAGS +=   --config $(CUR_DIR)/system.cfg
VPP_PACKAGE +=   --package.defer_aie_run

else 
VPP_FLAGS +=  -I $(XFLIB_DIR)/L1/include/

endif

############################## Setting Rules for AIE (Building Kernels) ##################
$(AIE_CONTAINER): $(XFLIB_DIR)/L2/tests/aie/pixelwise_mul/16bit_aie/graph.cpp 
	$(ECHO) "Compiling: libadf.a"
	mkdir -p $(dir $@)
	$(AIECXX) $(AIE_CXXFLAGS) $^
	mv ./libadf.a $(AIE_CONTAINER)

VPP_FLAGS_mm2s +=  
VPP_FLAGS_mm2s += --hls.clock 300000000:mm2s
VPP_FLAGS_bmm2s +=  
VPP_FLAGS_bmm2s += --hls.clock 300000000:bmm2s
VPP_FLAGS_s2mm +=  
VPP_FLAGS_s2mm += --hls.clock 300000000:s2mm

#######
########################## binary container global settings ##########################
BINARY_CONTAINER_kernel_OBJS += $(TEMP_DIR)/mm2s.xo
BINARY_CONTAINERS_DEPS += $(BINARY_CONTAINER_kernel_OBJS)
BINARY_CONTAINER_kernel_OBJS += $(TEMP_DIR)/bmm2s.xo
BINARY_CONTAINERS_DEPS += $(BINARY_CONTAINER_kernel_OBJS)
BINARY_CONTAINER_kernel_OBJS += $(TEMP_DIR)/s2mm.xo
BINARY_CONTAINERS_DEPS += $(BINARY_CONTAINER_kernel_OBJS)
ifeq ($(HOST_ARCH), x86)
BINARY_CONTAINERS += $(BUILD_DIR)/kernel.xclbin
else
BINARY_CONTAINERS += $(BUILD_DIR)/kernel_pkg.xclbin
BINARY_CONTAINERS_PKG += $(BUILD_DIR)/kernel.xclbin
endif

# ################ Setting Rules for Binary Containers (Building Kernels) ################
$(TEMP_DIR)/mm2s.xo: $(XFLIB_DIR)/L2/tests/aie/pixelwise_mul/16bit_aie/mm2s.cpp 
	$(ECHO) "Compiling Kernel: mm2s"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(VPP_FLAGS_mm2s) $(VPP_FLAGS) -k mm2s -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/bmm2s.xo: $(XFLIB_DIR)/L2/tests/aie/pixelwise_mul/16bit_aie/bmm2s.cpp 
	$(ECHO) "Compiling Kernel: bmm2s"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(VPP_FLAGS_bmm2s) $(VPP_FLAGS) -k bmm2s -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(TEMP_DIR)/s2mm.xo: $(XFLIB_DIR)/L2/tests/aie/pixelwise_mul/16bit_aie/s2mm.cpp 
	$(ECHO) "Compiling Kernel: s2mm"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(VPP_FLAGS_s2mm) $(VPP_FLAGS) -k s2mm -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' '$<'
$(BINARY_CONTAINERS): $(BINARY_CONTAINERS_DEPS)
	mkdir -p $(BUILD_DIR)
	$(VPP) -l $(VPP_FLAGS) --temp_dir $(BUILD_DIR) --report_dir $(BUILD_REPORT_DIR)/kernel $(VPP_LDFLAGS)  $(VPP_LDFLAGS_kernel) $(AIE_LDFLAGS)   -o $@ $^

############################## Setting Rules for Host (Building Host Executable) ##############################
ifeq ($(HOST_ARCH), x86)
$(EXE_FILE): $(EXE_FILE_DEPS) |  check_xrt
	mkdir -p $(BUILD_DIR)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

else
$(EXE_FILE): $(EXE_FILE_DEPS) |  check_sysroot
	mkdir -p $(BUILD_DIR)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

endif

$(EMCONFIG):
	emconfigutil --platform $(XPLATFORM) --od $(BUILD_DIR)
############################## Preparing sdcard folder ##############################
ifneq ($(HOST_ARCH), x86)
ifneq (,$(findstring zc706, $(XDEVICE)))
K_IMAGE := $(SYSROOT)/../../uImage
else
K_IMAGE := $(SYSROOT)/../../Image
endif
RUN_SCRIPT := $(BUILD_DIR)/run_script.sh
$(RUN_SCRIPT):
	rm -rf $(RUN_SCRIPT)
	@echo 'export LD_LIBRARY_PATH=/mnt:/tmp:$$LD_LIBRARY_PATH' >> $(RUN_SCRIPT)
ifneq ($(filter sw_emu hw_emu, $(TARGET)),)
	@echo 'export XCL_EMULATION_MODE=$(TARGET)' >> $(RUN_SCRIPT)
endif
	@echo 'export XILINX_VITIS=/mnt' >> $(RUN_SCRIPT)
	@echo 'export XILINX_XRT=/usr' >> $(RUN_SCRIPT)
	@echo 'if [ -f platform_desc.txt  ]; then' >> $(RUN_SCRIPT)
	@echo '        cp platform_desc.txt /etc/xocl.txt' >> $(RUN_SCRIPT)
	@echo 'fi' >> $(RUN_SCRIPT)
	@echo './$(EXE_NAME) $(PKG_HOST_ARGS)' >> $(RUN_SCRIPT)
	@echo 'return_code=$$?' >> $(RUN_SCRIPT)
	@echo 'if [ $$return_code -ne 0 ]; then' >> $(RUN_SCRIPT)
	@echo '        echo "ERROR: Embedded host run failed, RC=$$return_code"' >> $(RUN_SCRIPT)
	@echo 'else' >> $(RUN_SCRIPT)
	@echo '        echo "INFO: TEST PASSED, RC=0"' >> $(RUN_SCRIPT)
	@echo 'fi' >> $(RUN_SCRIPT)
	@echo 'echo "INFO: Embedded host run completed."' >> $(RUN_SCRIPT)
	@echo 'exit $$return_code' >> $(RUN_SCRIPT)
DATA_FILE := 
DATA_DIR := 
SD_FILES += $(RUN_SCRIPT)
SD_FILES += $(EXE_FILE)
SD_FILES += $(EMCONFIG)
SD_FILES += xrt.ini
SD_FILES += $(DATA_FILE)# where define DATAFILE in json
SD_FILES_WITH_PREFIX = $(foreach sd_file,$(SD_FILES), $(if $(filter $(sd_file),$(wildcard $(sd_file))), --package.sd_file $(sd_file)))
SD_DIRS_WITH_PREFIX = $(foreach sd_dir,$(DATA_DIR),--package.sd_dir $(sd_dir))
PACKAGE_FILES := $(BINARY_CONTAINERS)
PACKAGE_FILES += $(AIE_CONTAINER)
SD_CARD := $(CUR_DIR)/package_$(TARGET)
$(SD_CARD): $(EXE_FILE) $(BINARY_CONTAINERS) $(RUN_SCRIPT) $(EMCONFIG)
	@echo "Generating sd_card folder...."
	mkdir -p $(SD_CARD)
	chmod a+rx $(BUILD_DIR)/run_script.sh
	$(VPP) -t $(TARGET) --platform $(DEVICE) -o $(BINARY_CONTAINERS_PKG) -p $(PACKAGE_FILES) $(VPP_PACKAGE) --package.out_dir  $(SD_CARD) --package.rootfs $(SYSROOT)/../../rootfs.ext4 --package.kernel_image $(K_IMAGE) --package.boot_mode sd  $(SD_FILES_WITH_PREFIX) $(SD_DIRS_WITH_PREFIX)
	@echo "### ***** sd_card generation done! ***** ###"

.PHONY: sd_card
sd_card: $(SD_CARD)

endif
############################## Setting Essential Checks and Building Rules ##############################
ifneq (,$(filter x86sim aiesim, $(TARGET)))
RUN_DEPS += $(AIE_CONTAINER)
else
RUN_DEPS += $(SD_CARD)
RUN_DEPS += $(BINARY_CONTAINERS) $(AIE_CONTAINER)  $(EXE_FILE)  $(EMCONFIG)
endif

run: check_device  $(RUN_DEPS)
#x86sim
ifeq ($(TARGET), x86sim)
	$(X86SIMULATOR) --pkg-dir=./Work 
	
endif
#aiesim
ifeq ($(TARGET), aiesim)
	$(AIESIMULATOR) --pkg-dir=./Work --profile 
	
endif
#hw_emu
ifeq ($(TARGET), hw_emu)
ifeq ($(HOST_ARCH), x86)
	XCL_EMULATION_MODE=$(TARGET) $(EXE_FILE) $(HOST_ARGS)
	
else
	export XCL_EMULATION_MODE=$(TARGET)
	@echo $(RUN_DEPS)
	$(SD_CARD)/launch_$(TARGET).sh -no-reboot -run-app $(notdir $(RUN_SCRIPT)) 
	grep "TEST PASSED, RC=0" $(SD_CARD)/qemu_output.log || exit 1
	
endif
endif
#hw
ifeq ($(TARGET), hw)
ifeq ($(HOST_ARCH), x86)
	$(EXE_FILE) $(HOST_ARGS)
	
else
	$(ECHO) "Please copy the content of sd_card folder and data to an SD Card and run on the board"
endif
endif

############################## Setting Targets ##############################

.PHONY: all clean cleanall emconfig
emconfig: $(EMCONFIG)
ifeq ($(HOST_ARCH), x86)
all:  check_vpp check_platform check_xrt $(EXE_FILE) $(BINARY_CONTAINERS) emconfig
else
all:  check_vpp check_platform check_sysroot $(EXE_FILE) $(BINARY_CONTAINERS) emconfig sd_card
endif

.PHONY: host
ifeq ($(HOST_ARCH), x86)
host:  check_xrt $(EXE_FILE)
else
host:  check_sysroot $(EXE_FILE)
endif

.PHONY: xclbin
ifeq ($(HOST_ARCH), x86)
xclbin:  check_vpp check_xrt $(BINARY_CONTAINERS) $(AIE_CONTAINER)
else
xclbin:  check_vpp check_sysroot $(BINARY_CONTAINERS) $(AIE_CONTAINER)
endif

.PHONY: x86sim
x86sim: $(AIE_CONTAINER)
	$(X86SIMULATOR) --pkg-dir=./Work

############################## Cleaning Rules ##############################
cleanh:
	-$(RMDIR) $(EXE_FILE) vitis_* TempConfig system_estimate.xtxt *.rpt .run/
	-$(RMDIR) src/*.ll _xocc_* .Xil dltmp* xmltmp* *.log *.jou *.wcfg *.wdb sample_link.ini sample_compile.ini obj*  bin* *.csv *.jpg *.jpeg *.png

cleank:
	-$(RMDIR) $(BUILD_DIR)/*.xclbin _vimage *xclbin.run_summary qemu-memory-_* emulation/ _vimage/ pl*start_simulation. sh *.xclbin
	-$(RMDIR) _x_temp.*/_x.* _x_temp.*/.Xil _x_temp.*/profile_summary.* xo_* _x*
	-$(RMDIR) _x_temp.*/dltmp* _x_temp.*/kernel_info.dat _x_temp.*/*.log
	-$(RMDIR) _x_temp.* 

cleanall: cleanh cleank
	-$(RMDIR) $(BUILD_DIR) sd_card* build_dir.* emconfig.json *.html $(TEMP_DIR) $(CUR_DIR)/reports *.csv *.run_summary  $(CUR_DIR)/*.raw package_*   $(BUILD_DIR)/run_script.sh .ipcache *.str
	-$(RMDIR) $(XFLIB_DIR)/common/data/*.xe2xd* $(XFLIB_DIR)/common/data/*.orig*
	-$(RMDIR) $(AIE_CONTAINER) aiesimulator_output .AIE_SIM_CMD_LINE_OPTIONS $(CUR_DIR)/Work $(CUR_DIR)/*.xpe $(CUR_DIR)/hw.o $(CUR_DIR)/*.xsa $(CUR_DIR)/xnwOut

clean: cleanh