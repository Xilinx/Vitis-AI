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
#
############################## Help Section ##############################
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
	$(ECHO) "  make sd_card TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86>"
	$(ECHO) "      Command to prepare sd_card files."
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

############################## Setting up Project Variables ##############################
MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
XF_PROJ_ROOT ?= $(shell bash -c 'export MK_PATH=$(MK_PATH); echo $${MK_PATH%L3/tests/convertCsrCsc/*}')
CUR_DIR := $(patsubst %/,%,$(dir $(MK_PATH)))
XFLIB_DIR = $(XF_PROJ_ROOT)

ifneq ($(wildcard $(XILINX_VITIS)/aietools/.*),)
XILINX_VITIS_AIETOOLS := $(XILINX_VITIS)/aietools
LDXRT := -ladf_api_xrt
else
XILINX_VITIS_AIETOOLS := $(XILINX_VITIS)/cardano
LDXRT := -lcardano_api_xrt
endif

SYSROOT := $(SYSROOT)
TARGET ?= sw_emu
DEVICE ?= xilinx_u50_gen3x16_xdma_201920_3
HOST_ARCH ?= x86

ifneq ($(findstring u50, $(DEVICE)), u50)
$(warning [WARNING]: This project has not been tested for $(DEVICE). It may or may not work.)
endif

include ./utils.mk
XDEVICE := $(call device2xsa, $(DEVICE))
TEMP_DIR := _x_temp.$(TARGET).$(XDEVICE)
TEMP_REPORT_DIR := $(CUR_DIR)/reports/_x.$(TARGET).$(XDEVICE)
BUILD_DIR := build_dir.$(TARGET).$(XDEVICE)
BUILD_REPORT_DIR := $(CUR_DIR)/reports/_build.$(TARGET).$(XDEVICE)
EMCONFIG_DIR := $(BUILD_DIR)

PERL ?= perl
XCLBIN_DIR := $(CUR_DIR)/$(BUILD_DIR)
export XCL_BINDIR = $(XCLBIN_DIR)

# Setting tools
VPP := v++
SDCARD := sd_card
EMU_DIR := $(SDCARD)/data/emulation


############################## Setting up Host Variables ##############################
#Include Required Host Source Files
HOST_SRCS += $(XFLIB_DIR)/L3/tests/convertCsrCsc/test_convert.cpp
HOST_SRCS += $(XFLIB_DIR)/ext/xcl2/xcl2.cpp

CXXFLAGS += -I$(XFLIB_DIR)/L3/include
LDFLAGS += -L$(XFLIB_DIR)/L3/lib

LDFLAGS += -lgraphL3

LDFLAGS += -L$(XILINX_XRM)/lib -lxrm 

CXXFLAGS += -I$(XFLIB_DIR)/L2/include
CXXFLAGS += -I$(XFLIB_DIR)/../database/L1/include/hw
CXXFLAGS += -I$(XFLIB_DIR)/../quantitative_finance/L1/include
CXXFLAGS += -I$(XFLIB_DIR)/../utils/L1/include
CXXFLAGS += -I$(XFLIB_DIR)/ext/xcl2


ifeq ($(TARGET),sw_emu)
CXXFLAGS += -D SW_EMU_TEST
endif

ifeq ($(TARGET),hw_emu)
CXXFLAGS += -D HW_EMU_TEST
endif
# Host compiler global settings
CXXFLAGS += -I$(XILINX_XRT)/include -I$(XILINX_VIVADO)/include -std=c++11 -O3 -Wall -Wno-unknown-pragmas -Wno-unused-label
LDFLAGS += -L$(XILINX_XRT)/lib -lOpenCL -lpthread -lrt -Wno-unused-label -Wno-narrowing -DVERBOSE
CXXFLAGS += -fmessage-length=0 -O3 -I$(XILINX_XRM)/include
CXXFLAGS += -I$(CUR_DIR)/src/ 


ifeq ($(HOST_ARCH), x86)
#LDFLAGS += -L$(XILINX_VIVADO)/lnx64/tools/fpo_v7_0 -Wl,--as-needed -lgmp -lmpfr -lIp_floating_point_v7_0_bitacc_cmodel
endif
PKG_OUT_DIR := package_$(TARGET)
ifneq (,$(findstring zc706, $(XDEVICE)))
K_IMAGE := $(SYSROOT)/../../uImage
else
K_IMAGE := $(SYSROOT)/../../Image
endif

EXE_NAME := host.exe
EXE_FILE := $(BUILD_DIR)/$(EXE_NAME)
HOST_ARGS := -offset $(CUR_DIR)/data/csr_offsets.txt -indice $(CUR_DIR)/data/csr_columns.txt -goldenOffset $(CUR_DIR)/data/csc_offsets.txt -goldenIndice $(CUR_DIR)/data/csc_rows.txt
PKG_HOST_ARGS = $(foreach args,$(HOST_ARGS),$(subst $(dir $(patsubst %/,%,$(args))),,$(args)))
LIBRARY_PATH = $(CUR_DIR)/../../lib:$(XILINX_XRT)/lib:$(XILINX_XRM)/lib:$(LD_LIBRARY_PATH)


############################## Setting up Kernel Variables ##############################
# Kernel compiler global settings
VPP_FLAGS += -t $(TARGET) --platform $(XPLATFORM) --save-temps
LDCLFLAGS += --optimize 2 --jobs 8
ifneq (,$(shell echo $(XPLATFORM) | awk '/u50/'))
VPP_FLAGS += --config $(CUR_DIR)/conn_u50.ini
endif

VPP_FLAGS += -I$(XFLIB_DIR)/L2/include
VPP_FLAGS += -I$(XFLIB_DIR)/../database/L1/include/hw
VPP_FLAGS += -I$(XFLIB_DIR)/../quantitative_finance/L1/include
VPP_FLAGS += -I$(XFLIB_DIR)/../utils/L1/include

VPP_FLAGS += -I$(XFLIB_DIR)/L2/include
VPP_FLAGS += -I$(XFLIB_DIR)/L2/tests/convert_csc_csr/kernel



convertCsrCsc_kernel_VPP_FLAGS +=  -D KERNEL_NAME=convertCsrCsc_kernel
XFREQUENCY ?= 300.0
VIVADO_FREQUENCY =$(shell echo $$(( $(XFREQUENCY) * 1000000 )))
VPP_CFLAGS += --clock.defaultFreqHz $(VIVADO_FREQUENCY)

############################## Declaring Binary Containers ##############################


BINARY_CONTAINERS += $(BUILD_DIR)/convertCsrCsc_kernel.xclbin

BINARY_CONTAINERS_PKG += $(BUILD_DIR)/convertCsrCsc_kernel.xclbin
BINARY_CONTAINER_convertCsrCsc_kernel_OBJS += $(TEMP_DIR)/convertCsrCsc_kernel.xo

############################## Setting Targets ##############################
CP = cp -rf

.PHONY: all clean cleanall docs emconfig
all: check_vpp check_platform  $(EXE_FILE) $(BINARY_CONTAINERS) emconfig sd_card


.PHONY: host
host: check_xrt check_sysroot $(EXE_FILE)

.PHONY: xclbin
xclbin: check_vpp check_sysroot $(BINARY_CONTAINERS)

.PHONY: build
build: xclbin

############################## Setting Rules for Binary Containers (Building Kernels) ##############################


$(TEMP_DIR)/convertCsrCsc_kernel.xo: $(XFLIB_DIR)/L2/tests/convert_csc_csr/kernel/convertCsrCsc_kernel.cpp
	$(ECHO) "Compiling Kernel: convertCsrCsc_kernel"
	mkdir -p $(TEMP_DIR)
	$(VPP) $(convertCsrCsc_kernel_VPP_FLAGS) $(VPP_FLAGS) --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -c -k convertCsrCsc_kernel -I'$(<D)' -o'$@' '$<'


$(BUILD_DIR)/convertCsrCsc_kernel.xclbin: $(BINARY_CONTAINER_convertCsrCsc_kernel_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) $(VPP_FLAGS) --temp_dir $(BUILD_DIR) --report_dir $(BUILD_REPORT_DIR)/convertCsrCsc_kernel -l $(LDCLFLAGS) $(LDCLFLAGS_convertCsrCsc_kernel) -o'$@' $(+)

############################## Setting Rules for Host (Building Host Executable) ##############################
$(EXE_FILE): $(HOST_SRCS) | check_xrt
	mkdir -p $(BUILD_DIR)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(XPLATFORM) --od $(EMCONFIG_DIR)

############################## Setting Essential Checks and Building Rules ##############################
############################## Setting Essential Checks and Running Rules ##############################
RESULT_VAL ?= 
RUN_LEVEL ?= 7
run: all
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
ifeq ($(HOST_ARCH), x86)
	$(CP) $(EMCONFIG_DIR)/emconfig.json .
	XCL_EMULATION_MODE=$(TARGET) $(EXE_FILE) $(HOST_ARGS)
else
	export XCL_EMULATION_MODE=$(TARGET)
	$(PERL) $(XF_PROJ_ROOT)/ext/make_utility/embedded_run_emulation.pl "./$(PKG_OUT_DIR)/launch_$(TARGET).sh -no-reboot" "./run_script.sh" "$(RESULT_VAL)" "$(RUN_LEVEL)" | tee embedded_run.log 
endif
else
ifeq ($(HOST_ARCH), x86)
	LD_LIBRARY_PATH=$(LIBRARY_PATH) $(EXE_FILE) $(HOST_ARGS)
else
	$(ECHO) "Please copy the content of sd_card folder and data to an SD Card and run on the board"
endif
endif

############################## Preparing sdcard folder ##############################
PACKAGE_FILES := run_script.sh
PACKAGE_FILES += $(EXE_FILE)
PACKAGE_FILES += emconfig.json
PACKAGE_FILES += $(DATA_FILE)
SD_FILES_WITH_PREFIX = $(foreach sd_file,$(PACKAGE_FILES),--package.sd_file $(sd_file))
SD_DIRS_WITH_PREFIX = $(foreach sd_dir,$(DATA_DIR),--package.sd_dir $(sd_dir))
sd_card: $(EXE_FILE) $(BINARY_CONTAINERS) emconfig
ifneq ($(HOST_ARCH), x86)
	@echo "Generating sd_card folder...."
	mkdir -p $(PKG_OUT_DIR)
	rm -rf run_script.sh
	@echo 'export LD_LIBRARY_PATH=/mnt:/tmp:$$LD_LIBRARY_PATH' >> run_script.sh
ifeq ($(TARGET), $(filter $(TARGET),sw_emu hw_emu))
	@echo 'export XCL_EMULATION_MODE=$(TARGET)' >> run_script.sh
endif
	@echo 'export XILINX_VITIS=/mnt' >> run_script.sh
	@echo 'export XILINX_XRT=/usr' >> run_script.sh
	@echo 'cp platform_desc.txt /etc/xocl.txt' >> run_script.sh
	emconfigutil --platform $(DEVICE) --nd 1;
	@echo './$(EXE_NAME) $(PKG_HOST_ARGS)' >> run_script.sh
	@echo 'return_code=$$?' >> run_script.sh
	@echo 'if [ $$return_code -ne 0 ]; then' >> run_script.sh
	@echo '         echo "ERROR: Embedded host run failed, RC=$$return_code"' >> run_script.sh
	@echo 'fi' >> run_script.sh
	@echo 'echo "INFO: Embedded host run completed."' >> run_script.sh
	@echo 'exit $$return_code' >> run_script.sh
	chmod a+rx run_script.sh
	v++ -t $(TARGET) --platform $(DEVICE) -o $(BINARY_CONTAINERS) -p $(BINARY_CONTAINERS) --package.out_dir $(PKG_OUT_DIR) --package.rootfs $(SYSROOT)/../../rootfs.ext4 --package.sd_file $(K_IMAGE) $(SD_FILES_WITH_PREFIX) $(SD_DIRS_WITH_PREFIX)
	@echo "### ***** sd_card generation done! ***** ###"
endif
############################## Cleaning Rules ##############################
cleanh:
	-$(RMDIR) $(EXE_FILE) vitis_* TempConfig system_estimate.xtxt *.rpt .run/
	-$(RMDIR) src/*.ll _xocc_* .Xil dltmp* xmltmp* *.log *.jou *.wcfg *.wdb sample_link.ini sample_compile.ini obj* bin* *.csv *.jpg *.jpeg *.png

cleank:
	-$(RMDIR) $(BUILD_DIR)/*.xclbin _vimage *xclbin.run_summary qemu-memory-_* emulation/ _vimage/ pl* start_simulation.sh *.xclbin
	-$(RMDIR) _x_temp.*/_x.* _x_temp.*/.Xil _x_temp.*/profile_summary.* xo_* _x*
	-$(RMDIR) _x_temp.*/dltmp* _x_temp.*/kernel_info.dat _x_temp.*/*.log 
	-$(RMDIR) _x_temp.* 

cleanall: cleanh cleank
	-$(RMDIR) $(BUILD_DIR) sd_card* build_dir.* emconfig.json *.html $(TEMP_DIR) $(CUR_DIR)/reports *.csv *.run_summary $(CUR_DIR)/*.raw package_* run_script.sh .ipcache *.str
	-$(RMDIR) $(XFLIB_DIR)/common/data/*.xe2xd* $(XFLIB_DIR)/common/data/*.orig*
	-$(RMDIR) $(AIE_CONTAINERS) $(CUR_DIR)/Work $(CUR_DIR)/*.xpe $(CUR_DIR)/hw.o $(CUR_DIR)/*.xsa $(CUR_DIR)/xnwOut

clean: cleanh
