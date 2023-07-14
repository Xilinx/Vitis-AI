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
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86>"
	$(ECHO) "      Command to generate the design for specified Target and Shell."
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""
	$(ECHO) "  make sd_card TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64>"
	$(ECHO) "      Command to prepare sd_card files."
	$(ECHO) "      This target is only used in embedded device."
	$(ECHO) ""
	$(ECHO) "  make run TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86>"
	$(ECHO) "      Command to run application in emulation or on board."
	$(ECHO) ""
	$(ECHO) "  make build TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86>"
	$(ECHO) "      Command to build xclbin application."
	$(ECHO) ""
	$(ECHO) "  make host HOST_ARCH=<aarch32/aarch64/x86>"
	$(ECHO) "      Command to build host application."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH is required for embedded devices"
	$(ECHO) ""
	$(ECHO) "  NOTE: For embedded devices, e.g. zcu102/zcu104/vck190, env variable SYSROOT, EDGE_COMMON_SW and PERL need to be set first, and HOST_ARCH is either aarch32 or aarch64. For example,"
	$(ECHO) "       export SYSROOT=< path-to-platform-sysroot >"
	$(ECHO) "       export EDGE_COMMON_SW=< path-to-rootfs-and-Image-files >"
	$(ECHO) "       export PERL=<path-to-perl-installation-location >"
	$(ECHO) ""

# ##################### Setting up default value of TARGET ##########################
TARGET ?= sw_emu

# ################### Setting up default value of DEVICE ##############################
DEVICE ?= xilinx_u200_xdma_201830_2

# ###################### Setting up default value of HOST_ARCH ####################### 
HOST_ARCH ?= x86

# #################### Checking if DEVICE in blacklist #############################
#ifeq ($(findstring u280, $(DEVICE)), u280)
#$(error [ERROR]: This project is not supported for $(DEVICE).)
#endif
ifeq ($(findstring u250, $(DEVICE)), u250)
$(error [ERROR]: This project is not supported for $(DEVICE).)
endif

# #################### Checking if DEVICE in whitelist ############################
ifneq ($(findstring u200, $(DEVICE)), u200)
ifneq ($(findstring aws-vu9p-f1, $(DEVICE)), aws-vu9p-f1)
ifneq ($(findstring vck190, $(DEVICE)), vck190)
ifneq ($(findstring zcu102, $(DEVICE)), zcu102)
ifneq ($(findstring zcu104, $(DEVICE)), zcu104)
ifneq ($(findstring u50, $(DEVICE)), u50)
$(warning [WARNING]: This project has not been tested for $(DEVICE). It may or may not work.)
endif
endif
endif
endif
endif
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

# ####################### Setting environment variables for embedded #####################
ifneq ($(HOST_ARCH), x86)
SYSROOT := $(SYSROOT)
PERL ?= perl
SDCARD := sd_card
EMU_DIR := $(SDCARD)/data/emulation
endif

# ######################### Setting up Host Variables #########################
#Include Required Host Source Files
HOST_SRCS += $(CUR_DIR)/xf_sort_nms_tb.cpp
HOST_SRCS += $(CUR_DIR)/ext/xcl2/xcl2.cpp

CXXFLAGS +=
CXXFLAGS += -I$(CUR_DIR)/ext/xcl2/
CXXFLAGS += -I$(CUR_DIR)/../../include/pl/

LDFLAGS += -lopencv_videoio -lopencv_imgcodecs -lopencv_core -lopencv_imgproc -lopencv_features2d -lopencv_flann -lopencv_video -lopencv_calib3d -lopencv_highgui


ifeq ($(TARGET),sw_emu)
CXXFLAGS += -D SW_EMU_TEST
endif

ifeq ($(TARGET),hw_emu)
CXXFLAGS += -D HW_EMU_TEST
endif

# ######################### Host compiler global settings ############################
ifneq ($(HOST_ARCH), x86)
CXXFLAGS += --sysroot=$(SYSROOT)
CXXFLAGS += -I$(SYSROOT)/usr/include/xrt -I$(XILINX_HLS)/include -std=c++14 -O3 -Wall -Wno-unknown-pragmas -Wno-unused-label
else
CXXFLAGS += -I$(XILINX_XRT)/include -I$(XILINX_HLS)/include -std=c++11 -O3 -Wall -Wno-unknown-pragmas -Wno-unused-label
endif
LDFLAGS += -L$(XILINX_XRT)/lib -lOpenCL -lpthread -lrt -Wno-unused-label -Wno-narrowing -DVERBOSE
CXXFLAGS += -fmessage-length=0 -O3
CXXFLAGS += -I$(CUR_DIR)/src/ 

ifeq ($(HOST_ARCH), x86)
LDFLAGS += -L$(XILINX_HLS)/lnx64/tools/fpo_v7_0 -Wl,--as-needed -lgmp -lmpfr -lIp_floating_point_v7_0_bitacc_cmodel
endif
ifneq (,$(shell echo $(XPLATFORM) | awk '/u50/'))
CXXFLAGS += -I$(OPENCV_INCLUDE)
LDFLAGS += -L$(OPENCV_LIB)
else ifneq (,$(shell echo $(XPLATFORM) | awk '/vck190/'))
CXXFLAGS += -I$(SYSROOT)/usr/include
CXXFLAGS += -I$(SYSROOT)/usr/include/opencv4
CXXFLAGS += --sysroot=$(SYSROOT)
LDFLAGS += -L$(SYSROOT)/usr/lib
LDFLAGS += -L${SYSROOT}/lib
else ifneq (,$(shell echo $(XPLATFORM) | awk '/u200/'))
CXXFLAGS += -I$(OPENCV_INCLUDE)
LDFLAGS += -L$(OPENCV_LIB)
else ifneq (,$(shell echo $(XPLATFORM) | awk '/aws-vu9p-f1/'))
CXXFLAGS += -I$(OPENCV_INCLUDE)
LDFLAGS += -L$(OPENCV_LIB)
else ifneq (,$(shell echo $(XPLATFORM) | awk '/zcu102/'))
CXXFLAGS += -I$(SYSROOT)/usr/include
CXXFLAGS += -I$(SYSROOT)/usr/include/opencv4
CXXFLAGS += --sysroot=$(SYSROOT)
LDFLAGS += -L$(SYSROOT)/usr/lib
LDFLAGS += -L${SYSROOT}/lib
else ifneq (,$(shell echo $(XPLATFORM) | awk '/zcu104/'))
CXXFLAGS += -I$(SYSROOT)/usr/include
CXXFLAGS += -I$(SYSROOT)/usr/include/opencv4
CXXFLAGS += --sysroot=$(SYSROOT)
LDFLAGS += -L$(SYSROOT)/usr/lib
LDFLAGS += -L${SYSROOT}/lib
endif

# ################### Setting package and image directory #######################
EMBEDDED_PACKAGE_OUT = package_$(TARGET)
ifneq (,$(findstring zc706, $(XDEVICE)))
K_IMAGE := $(EDGE_COMMON_SW)/uImage
else
K_IMAGE := $(EDGE_COMMON_SW)/Image
endif

EXE_NAME := sort_nms
EXE_FILE := $(BUILD_DIR)/$(EXE_NAME)
ifeq ($(TARGET),$(filter $(TARGET),sw_emu))
HOST_ARGS := $(XFLIB_DIR)/data/128x128_1.png $(XFLIB_DIR)/data/128x128_2.png

PKG_HOST_ARGS = $(foreach args,$(HOST_ARGS),$(subst $(dir $(patsubst %/,%,$(args))),,$(args)))
LIBRARY_PATH =$(OPENCV_LIB):$(LD_LIBRARY_PATH):$(XILINX_XRT)/lib
endif
ifeq ($(TARGET),$(filter $(TARGET),hw_emu))
HOST_ARGS := $(XFLIB_DIR)/data/128x128_1.png $(XFLIB_DIR)/data/128x128_2.png

PKG_HOST_ARGS = $(foreach args,$(HOST_ARGS),$(subst $(dir $(patsubst %/,%,$(args))),,$(args)))
LIBRARY_PATH =$(OPENCV_LIB):$(LD_LIBRARY_PATH):$(XILINX_XRT)/lib
endif
ifeq ($(TARGET),$(filter $(TARGET),hw))
HOST_ARGS := $(XFLIB_DIR)/data/4k_1.jpg $(XFLIB_DIR)/data/4k_2.jpg

PKG_HOST_ARGS = $(foreach args,$(HOST_ARGS),$(subst $(dir $(patsubst %/,%,$(args))),,$(args)))
LIBRARY_PATH =$(OPENCV_LIB):$(LD_LIBRARY_PATH):$(XILINX_XRT)/lib
endif

# ##################### Kernel compiler global settings ##########################
VPP_FLAGS += -t $(TARGET) --platform $(XPLATFORM) --save-temps --optimize 2
VPP_FLAGS += --hls.jobs 8
VPP_LDFLAGS += --vivado.synth.jobs 8 --vivado.impl.jobs 8
VPP_FLAGS += -I$(CUR_DIR)/../../include/pl/


VPP_FLAGS += 
VPP_FLAGS += 

# Kernel args

# ############################ Declaring Binary Containers ##########################

ifeq ($(HOST_ARCH), x86)
BINARY_CONTAINERS += $(BUILD_DIR)/krnl_sort_nms.xclbin
BINARY_CONTAINERS_PKG += $(BUILD_DIR)/krnl_sort_nms.xclbin
else
BINARY_CONTAINERS += $(BUILD_DIR)/krnl_sort_nms_xo.xclbin
BINARY_CONTAINERS_PKG += $(BUILD_DIR)/krnl_sort_nms.xclbin
endif
BINARY_CONTAINER_krnl_sort_nms_OBJS += $(TEMP_DIR)/sort_nms_accel.xo

# ######################### Setting Targets of Makefile ################################
DATA_FILE += $(XFLIB_DIR)/data/4k_1.jpg
DATA_FILE += $(XFLIB_DIR)/data/4k_2.jpg
DATA_FILE += $(XFLIB_DIR)/data/128x128_1.png
DATA_FILE += $(XFLIB_DIR)/data/128x128_2.png

.PHONY: all clean cleanall docs emconfig
ifeq ($(HOST_ARCH), x86)
all:  check_version check_vpp check_platform check_xrt $(EXE_FILE) $(BINARY_CONTAINERS) emconfig
else
all:  check_version check_vpp check_platform check_sysroot $(EXE_FILE) $(BINARY_CONTAINERS) emconfig  sd_card
endif

.PHONY: check_env
check_env:
ifneq (,$(shell echo $(XPLATFORM) | awk '/u200/'))
ifeq (,$(OPENCV_INCLUDE))
	@echo "Cannot find OpenCV include path. Please set OPENCV_INCLUDE variable" && false
endif
endif
ifneq (,$(shell echo $(XPLATFORM) | awk '/u50/'))
ifeq (,$(OPENCV_INCLUDE))
	@echo "Cannot find OpenCV include path. Please set OPENCV_INCLUDE variable" && false
endif
endif
ifneq (,$(shell echo $(XPLATFORM) | awk '/aws-vu9p-f1/'))
ifeq (,$(OPENCV_INCLUDE))
	@echo "Cannot find OpenCV include path. Please set OPENCV_INCLUDE variable" && false
endif
endif
ifneq (,$(shell echo $(XPLATFORM) | awk '/u200/'))
ifeq (,$(OPENCV_LIB))
	@echo "Cannot find Opencv lib path. Please set OPENCV_LIB variable" && false
endif
endif
ifneq (,$(shell echo $(XPLATFORM) | awk '/u50/'))
ifeq (,$(OPENCV_LIB))
	@echo "Cannot find Opencv lib path. Please set OPENCV_LIB variable" && false
endif
endif
ifneq (,$(shell echo $(XPLATFORM) | awk '/aws-vu9p-f1/'))
ifeq (,$(OPENCV_LIB))
	@echo "Cannot find Opencv lib path. Please set OPENCV_LIB variable" && false
endif
endif
ifneq (,$(shell echo $(XPLATFORM) | awk '/zcu102/'))
ifeq (,$(EDGE_COMMON_SW))
	@echo "Please set EDGE_COMMON_SW variable" && false
endif
endif
ifneq (,$(shell echo $(XPLATFORM) | awk '/zcu104/'))
ifeq (,$(EDGE_COMMON_SW))
	@echo "Please set EDGE_COMMON_SW variable" && false
endif
endif
ifneq (,$(shell echo $(XPLATFORM) | awk '/vck190/'))
ifeq (,$(EDGE_COMMON_SW))
	@echo "Please set EDGE_COMMON_SW variable" && false
endif
endif

.PHONY: host
ifeq ($(HOST_ARCH), x86)
host:  check_env check_xrt $(EXE_FILE)
else
host:  check_env check_sysroot $(EXE_FILE)
endif

.PHONY: xo
xo: $(TEMP_DIR)/sort_nms_accel.xo | check_vpp check_platform
	cp -f $(TEMP_DIR)/sort_nms_accel.xo $(CUR_DIR)


.PHONY: xclbin
ifeq ($(HOST_ARCH), x86)
xclbin: check_vpp  check_xrt  $(BINARY_CONTAINERS)
else
xclbin: check_vpp  check_sysroot  $(BINARY_CONTAINERS)
endif

.PHONY: build
build: xclbin

# ################ Setting Rules for Binary Containers (Building Kernels) ################
$(TEMP_DIR)/sort_nms_accel.xo: $(CUR_DIR)/xf_sort_nms_accel.cpp
	$(ECHO) "Compiling Kernel: sort_nms_accel"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(sort_nms_accel_VPP_FLAGS) $(VPP_FLAGS) -k sort_nms_accel -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' $^


$(BINARY_CONTAINERS): $(BINARY_CONTAINER_krnl_sort_nms_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) -l $(VPP_FLAGS) --temp_dir $(TEMP_DIR) --report_dir $(BUILD_REPORT_DIR)/krnl_sort_nms $(VPP_LDFLAGS) $(VPP_LDFLAGS_krnl_sort_nms) -o '$@' $(+)


# ################# Setting Rules for Host (Building Host Executable) ################
ifeq ($(HOST_ARCH), x86)
$(EXE_FILE):  $(HOST_SRCS) | check_xrt check_env
else
$(EXE_FILE):  $(HOST_SRCS) | check_sysroot check_env
endif

	mkdir -p $(BUILD_DIR)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(XPLATFORM) --od $(EMCONFIG_DIR)


# ###############Setting Essential Checks And Running Rules For Vitis Flow #############
# Setting default value for vitis flow
RESULT_VAL ?= 
RUN_LEVEL ?= 7
run: all
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
ifeq ($(HOST_ARCH), x86)
	$(CP) $(EMCONFIG_DIR)/emconfig.json .
	LD_LIBRARY_PATH=$(LIBRARY_PATH) XCL_EMULATION_MODE=$(TARGET) $(EXE_FILE) $(HOST_ARGS)
else
ifeq ($(findstring vck190, $(DEVICE)), vck190)
	export XCL_EMULATION_MODE=$(TARGET)
	$(PERL) $(XF_PROJ_ROOT)/ext/make_utility/embedded_run_emulation.pl "./$(EMBEDDED_PACKAGE_OUT)/launch_$(TARGET).sh -noc-memory-config qemu-memory-_ddr@0x00000000 -no-reboot" "./run_script.sh" "$(RESULT_STRING)" "$(RUN_LEVEL)" | tee embedded_run.log
else
	$(PERL) $(XF_PROJ_ROOT)/ext/make_utility/embedded_run_emulation.pl "./$(EMBEDDED_PACKAGE_OUT)/launch_$(TARGET).sh -no-reboot" "./run_script.sh" "$(RESULT_VAL)" "$(RUN_LEVEL)" | tee embedded_run.log 
endif
endif
else
	LD_LIBRARY_PATH=$(LIBRARY_PATH)
ifeq ($(HOST_ARCH), x86)
	$(EXE_FILE) $(HOST_ARGS)
else
	$(ECHO) "Please copy the content of sd_card folder and data to an SD Card and run on the board"
endif
endif

# ######################### Preparing sdcard folder ##############################
EMBEDDED_EXEC_SCRIPT = run_script.sh
PACKAGE_FILES += $(EMBEDDED_EXEC_SCRIPT)
PACKAGE_FILES += $(EXE_FILE)
PACKAGE_FILES += emconfig.json
PACKAGE_FILES += $(DATA_FILE)
SD_FILES_WITH_PREFIX = $(foreach sd_file,$(PACKAGE_FILES),--package.sd_file $(sd_file))
SD_DIRS_WITH_PREFIX = $(foreach sd_dir,$(DATA_DIR),--package.sd_dir $(sd_dir))
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
ifeq ($(findstring vck190, $(DEVICE)), vck190)
	$(VPP) -t $(TARGET) --platform $(DEVICE) -o $(BINARY_CONTAINERS_PKG) -p $(BINARY_CONTAINERS)  --package.out_dir $(EMBEDDED_PACKAGE_OUT) --package.rootfs $(EDGE_COMMON_SW)/rootfs.ext4 --package.kernel_image $(K_IMAGE) --package.boot_mode sd $(SD_FILES_WITH_PREFIX) $(SD_DIRS_WITH_PREFIX)
else
	$(VPP) -t $(TARGET) --platform $(DEVICE) -o $(BINARY_CONTAINERS_PKG) -p $(BINARY_CONTAINERS) --package.out_dir $(EMBEDDED_PACKAGE_OUT) --package.rootfs $(EDGE_COMMON_SW)/rootfs.ext4 --package.sd_file $(K_IMAGE) $(SD_FILES_WITH_PREFIX) $(SD_DIRS_WITH_PREFIX)
endif
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
