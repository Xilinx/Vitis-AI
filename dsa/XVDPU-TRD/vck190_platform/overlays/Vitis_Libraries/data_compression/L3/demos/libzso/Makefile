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
#

.SECONDEXPANSION:

# ------------------------------------------------------------
#						Help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform>"
	$(ECHO) "      Command to generate the design for specified Target and Device."
	$(ECHO) ""
	$(ECHO) "  make run TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform>"
	$(ECHO) "      Command to run application in emulation."
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make lib"
	$(ECHO) "      Command to build libz.so (ZLIB Shared Object File)."
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""

# ------------------------------------------------------------
#						Build Environment Setup

include ./utils.mk

TOOL_VERSION ?= 2019.2

#check environment setup
ifndef XILINX_VITIS
  XILINX_VITIS = /opt/xilinx/Vitis/$(TOOL_VERSION)
  export XILINX_VITIS
endif
ifndef XILINX_VIVADO
  XILINX_VIVADO = /opt/xilinx/Vivado/$(TOOL_VERSION)
  export XILINX_VIVADO
endif
ifndef XILINX_XRT
  XILINX_XRT = /opt/xilinx/xrt
  export XILINX_XRT
endif

.PHONY: check_vivado
check_vivado:
ifeq (,$(wildcard $(XILINX_VIVADO)/bin/vivado))
	@echo "Cannot locate Vivado installation. Please set XILINX_VIVADO variable." && false
endif

.PHONY: check_vpp
check_vpp:
ifeq (,$(wildcard $(XILINX_VITIS)/bin/v++))
	@echo "Cannot locate Vitis installation. Please set XILINX_VITIS variable." && false
endif

.PHONY: check_xrt
check_xrt:
ifeq (,$(wildcard $(XILINX_XRT)/lib/libxilinxopencl.so))
	@echo "Cannot locate XRT installation. Please set XILINX_XRT variable." && false
endif

export PATH := $(XILINX_VITIS)/bin:$(XILINX_XRT)/bin:$(PATH)

ifeq (,$(LD_LIBRARY_PATH))
  LD_LIBRARY_PATH := $(XILINX_XRT)/lib
else
  LD_LIBRARY_PATH := $(XILINX_XRT)/lib:$(LD_LIBRARY_PATH)
endif
ifneq (,$(wildcard $(XILINX_VITIS)/bin/ldlibpath.sh))
  export LD_LIBRARY_PATH := $(shell $(XILINX_VITIS)/bin/ldlibpath.sh $(XILINX_VITIS)/lib/lnx64.o):$(LD_LIBRARY_PATH)
endif

# sw_emu, hw_emu, hw
TARGET ?= sw_emu
ifeq ($(filter $(TARGET),sw_emu hw hw_emu),)
  $(error TARGET is not sw_emu, hw_emu or hw)
endif

# ZLIB Version
ZLIBVER ?= zlib-1.2.7
ifeq ($(filter $(ZLIBVER), zlib zlib-1.2.7 zlib-1.2.11),)
	$(error ZLIBVER is not zlib, zlib-1.2.7 or zlib-1.2.11)
endif 

# Target device
DEVICE ?= xilinx_u200_xdma_201920_1

ifneq (,$(wildcard $(DEVICE)))
  # Use DEVICE as a file path
  XPLATFORM := $(DEVICE)
else
  # Use DEVICE as a file name pattern
  DEVICE_L := $(shell echo $(DEVICE) | tr A-Z a-z)
  # Match the name
ifneq (,$(PLATFORM_REPO_PATHS))
  XPLATFORMS := $(foreach p, $(subst :, ,$(PLATFORM_REPO_PATHS)), $(wildcard $(p)/*/*.xpfm))
  XPLATFORM := $(strip $(foreach p, $(XPLATFORMS), $(shell echo $(p) | awk '$$1 ~ /$(DEVICE_L)/')))
endif
ifeq (,$(XPLATFORM))
  XPLATFORMS := $(wildcard $(XILINX_VITIS)/platforms/*/*.xpfm)
  XPLATFORM := $(strip $(foreach p, $(XPLATFORMS), $(shell echo $(p) | awk '$$1 ~ /$(DEVICE_L)/')))
endif
ifeq (,$(XPLATFORM))
  XPLATFORMS := $(wildcard /opt/xilinx/platforms/*/*.xpfm)
  XPLATFORM := $(strip $(foreach p, $(XPLATFORMS), $(shell echo $(p) | awk '$$1 ~ /$(DEVICE_L)/')))
endif
endif

define MSG_PLATFORM
No platform matched pattern '$(DEVICE)'.
Available platforms are: $(XPLATFORMS)
To add more platform directories, set the PLATFORM_REPO_PATHS variable.
endef
export MSG_PLATFORM

define MSG_DEVICE
More than one platform matched: $(XPLATFORM)
Please set DEVICE variable more accurately to select only one platform file. For example: DEVICE='u200.*xdma'
endef
export MSG_DEVICE

.PHONY: check_platform
check_platform:
ifeq (,$(XPLATFORM))
	@echo "$${MSG_PLATFORM}" && false
endif
ifneq (,$(word 2,$(XPLATFORM)))
	@echo "$${MSG_DEVICE}" && false
endif

XDEVICE := $(basename $(notdir $(firstword $(XPLATFORM))))

$(info $$XDEVICE is ${XDEVICE})
$(info $$PLATFORM is ${XPLATFORM})
$(info ${XILINX_XRT})
# ------------------------------------------------------------
#						Directory Setup

MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
XF_PROJ_ROOT ?= $(shell bash -c 'export MK_PATH=$(MK_PATH); echo $${MK_PATH%L3/demos/libzso/*}')
CUR_DIR := $(patsubst %/,%,$(dir $(MK_PATH)))

XFLIB_DIR := $(shell readlink -f $(XF_PROJ_ROOT))

BUILD_DIR := $(CUR_DIR)/build
TEMP_DIR := $(CUR_DIR)/_x_temp.$(TARGET).$(XDEVICE)
SRC_DIR := $(XFLIB_DIR)/L3/demos/libzso/src
TB_DIR := $(XFLIB_DIR)/L2/tests/src/
ZLIBDIR := $(XFLIB_DIR)/common/thirdParty/$(ZLIBVER)/

include ./config.mk

# ------------------------------------------------------------
#                      kernel setup

KSRC_DIR = $(XFLIB_DIR)/L2/src/

VPP = $(XILINX_VITIS)/bin/v++

# HLS src files
HLS_SRC_DIR = $(XFLIB_DIR)/L1/include/hw

# Compilation flags
VPP_FLAGS = -I$(HLS_SRC_DIR) \
			-I$(KSRC_DIR) \
			-I$(XFLIB_DIR)/L2/include/

VPP_FLAGS += -I$(XFLIB_DIR)/L2/tests/src/
VPP_FLAGS += -I$(XFLIB_DIR)/../security/L1/include

VPP_FLAGS += -t $(TARGET) --platform $(XPLATFORM) --save-temps
VPP_FLAGS += --config $(CUR_DIR)/advanced.cfg \
			 -DMULTIPLE_BYTES=$(MULTIPLE_BYTES)

VPP_DIRS = --temp_dir $(TEMP_DIR)/_x.$(TARGET) \
		--report_dir $(CUR_DIR)/reports/_x.$(TARGET)

# Linking flags
VPP_LINK_FLAGS = --kernel_frequency 250 --optimize 2 --config $(CUR_DIR)/connectivity.cfg

VPP_LINK_DIRS = --temp_dir $(TEMP_DIR)/_build.$(TARGET)\
				--report_dir $(CUR_DIR)/reports/_build.$(TARGET)/

XCLBIN_FILE = $(BUILD_DIR)/xclbin_$(XDEVICE)_$(TARGET)/compress_decompress.xclbin

XO_FILES_C = $(TEMP_DIR)/xilGzipCompBlock.xo \
		  	 
XO_FILES_D = $(TEMP_DIR)/xilGzipS2MM.xo \
		     $(TEMP_DIR)/xilGzipMM2S.xo \
		     $(TEMP_DIR)/xilDecompress.xo

COMPRESS_KERNEL_SRCS = $(KSRC_DIR)/gzip_compress_multicore_mm.cpp
DATAREADER_KERNEL_SRCS = $(TB_DIR)/gzip_dm_s2mm.cpp
DATAWRITER_KERNEL_SRCS = $(TB_DIR)/gzip_dm_mm2s.cpp
DECOMPRESS_KERNEL_SRCS = $(KSRC_DIR)/zlib_decompress.cpp

COMPRESS_KERNEL_NAME = xilGzipCompBlock
DATAREADER_KERNEL_NAME = xilGzipS2MM
DATAWRITER_KERNEL_NAME = xilGzipMM2S
DECOMPRESS_KERNEL_NAME = xilDecompress

KERNELS += $(COMPRESS_KERNEL_NAME) $(CHECKSUM_KERNEL_NAME)
KERNELS += $(DATAREADER_KERNEL_NAME) $(DATAWRITER_KERNEL_NAME) $(DECOMPRESS_KERNEL_NAME)


# ------------------------------------------------------------
#                      kernel rules

# Building kernel
$(TEMP_DIR)/xilGzipCompBlock.xo: $(COMPRESS_KERNEL_SRCS) $(HLS_SRC_DIR)
	@echo -e "----\nCompiling compression kernel..."
	mkdir -p $(TEMP_DIR)
	$(VPP) $(VPP_FLAGS) $(VPP_DIRS) -c -k $(COMPRESS_KERNEL_NAME) \
		-I'$(<D)' -o'$@' '$<'

$(TEMP_DIR)/xilGzipS2MM.xo: $(DATAREADER_KERNEL_SRCS) $(HLS_SRC_DIR)
	@echo -e "----\nCompiling data reader kernel $*..."
	mkdir -p $(TEMP_DIR)
	$(VPP) $(VPP_FLAGS) $(VPP_DIRS) -c -k $(DATAREADER_KERNEL_NAME) \
	    -I'$(<D)' -o'$@' '$<'

$(TEMP_DIR)/xilGzipMM2S.xo: $(DATAWRITER_KERNEL_SRCS) $(HLS_SRC_DIR)
	@echo -e "----\nCompiling data writer kernel $*..."
	mkdir -p $(TEMP_DIR)
	$(VPP) $(VPP_FLAGS) $(VPP_DIRS) -c -k $(DATAWRITER_KERNEL_NAME) \
	    -I'$(<D)' -o'$@' '$<'

$(TEMP_DIR)/xilDecompress.xo: $(DECOMPRESS_KERNEL_SRCS) $(HLS_SRC_DIR)
	@echo -e "----\nCompiling decompression kernel $*..."
	mkdir -p $(TEMP_DIR)
	$(VPP) $(VPP_FLAGS) $(VPP_DIRS) -c -k $(DECOMPRESS_KERNEL_NAME) \
	    -I'$(<D)' -o'$@' '$<'


# xclbin Binary creation
$(XCLBIN_FILE): $(XO_FILES_C) $(XO_FILES_D)
	mkdir -p $(BUILD_DIR)
	$(VPP) $(VPP_FLAGS) $(VPP_LINK_FLAGS) $(VPP_LINK_DIRS) -l \
	       -o'$@' $(+)


# ------------------------------------------------------------
#                       host setup

OBJ_DIR_BASE ?= obj
BIN_DIR_BASE ?= bin

BIN_DIR_SUFFIX ?= _$(XDEVICE)

OBJ_DIR = $(CUR_DIR)/$(OBJ_DIR_BASE)$(BIN_DIR_SUFFIX)

#Checks for g++
CXX := g++
ifeq ($(HOST_ARCH), x86)
ifneq ($(shell expr $(shell g++ -dumpversion) \>= 5), 1)
ifndef XILINX_VIVADO
$(error [ERROR]: g++ version older. Please use 5.0 or above)
else
CXX := $(XILINX_VIVADO)/tps/lnx64/gcc-6.2.0/bin/g++
$(warning [WARNING]: g++ version older. Using g++ provided by the tool : $(CXX))
endif
endif
endif

EXE_NAME := xzlib 
ifeq ($(ZLIBVER),zlib-1.2.7)
LIBZLIB_NAME := libz.so.1.2.7
endif

#EXE_EXT = exe

CXXFLAGS +=-I$(CUR_DIR)/src/
CXXFLAGS +=-I$(XFLIB_DIR)/L3/include/
CXXFLAGS +=-I$(XFLIB_DIR)/L1/include/hw/
CXXFLAGS +=-I$(XILINX_XRT)/include/
CXXFLAGS +=-I$(XFLIB_DIR)/common/libs/xcl2/
CXXFLAGS +=-I$(XFLIB_DIR)/common/libs/cmdparser/
CXXFLAGS +=-I$(XFLIB_DIR)/common/libs/logger/
#CXXFLAGS +=-I$(XFLIB_DIR)/L3/demos/zlib_app/hadoop/

#Host and Common sources
#SRCS += host.cpp
EXTRA_OBJS += xil_zlib xcl2 cmdlineparser logger adler32 crc32 deflate zutil trees compress uncompr inflate inftrees inffast zpipe xlibz
EXTRA_OBJS += infback gzclose gzlib gzread gzwrite 
xil_zlib_SRCS = $(XFLIB_DIR)/L3/src/zlib.cpp
xcl2_SRCS = $(XFLIB_DIR)/common/libs/xcl2/xcl2.cpp
cmdlineparser_SRCS = $(XFLIB_DIR)/common/libs/cmdparser/cmdlineparser.cpp
logger_SRCS = $(XFLIB_DIR)/common/libs/logger/logger.cpp

# ThirdParty ZLIB Source Files
adler32_SRCS = $(ZLIBDIR)/adler32.c
crc32_SRCS = $(ZLIBDIR)/crc32.c
deflate_SRCS = $(ZLIBDIR)/deflate.c
zutil_SRCS = $(ZLIBDIR)/zutil.c
trees_SRCS = $(ZLIBDIR)/trees.c
compress_SRCS = $(ZLIBDIR)/compress.c
uncompr_SRCS = $(ZLIBDIR)/uncompr.c
inflate_SRCS = $(ZLIBDIR)/inflate.c
inftrees_SRCS = $(ZLIBDIR)/inftrees.c
inffast_SRCS = $(ZLIBDIR)/inffast.c
infback_SRCS = $(ZLIBDIR)/infback.c
gzclose_SRCS = $(ZLIBDIR)/gzclose.c
gzlib_SRCS = $(ZLIBDIR)/gzlib.c
gzread_SRCS = $(ZLIBDIR)/gzread.c
gzwrite_SRCS = $(ZLIBDIR)/gzwrite.c
zpipe_SRCS = $(ZLIBDIR)/examples/zpipe.c
xlibz_SRCS = $(ZLIBDIR)/xlibz.cpp

CXXFLAGS += -std=c++11 -fmessage-length=0 \
		-DXDEVICE=$(XDEVICE) \
	    -Wall -Wno-unknown-pragmas -Wno-unused-label \
		-Wno-unused-function
CXXFLAGS += -O3	   

ifeq ($(ENABLE_XRM), yes) 
CXXFLAGS += -I/opt/xilinx/xrm/include
endif

LDFLAGS += -L$(XILINX_XRT)/lib/ -lOpenCL 
LDFLAGS += -lrt -Wno-unused-label -Wno-narrowing
LDFLAGS += -Wno-unused-function

ifeq ($(ENABLE_XRM), yes)
LDFLAGS += -L/opt/xilinx/xrm/lib/
LDFLAGS += -lxrm
endif

EXE_FILE = $(BUILD_DIR)/$(EXE_NAME)
LIBZLIB = $(BUILD_DIR)/$(LIBZLIB_NAME)

# ------------------------------------------------------------
#                       libz rules
OBJ_FILES = $(foreach s,$(SRCS),$(OBJ_DIR)/$(basename $(s)).o)

$(OBJ_DIR)/%.o: CXXFLAGS += $($(*)_CXXFLAGS)

$(OBJ_FILES): $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo -e "----\nCompiling object $*..."
	mkdir -p $(@D) 
	$(CXX) -fPIC -D_LARGEFILE64_SOURCE=1 -DHAVE_HIDDEN -DPIC -o $@ -c $< $(CXXFLAGS) $(LDFLAGS)

EXTRA_OBJ_FILES = $(foreach f,$(EXTRA_OBJS),$(OBJ_DIR)/$(f).o)

$(EXTRA_OBJ_FILES): $(OBJ_DIR)/%.o: $$($$(*)_SRCS) $$($$(*)_HDRS)
	@echo -e "----\nCompiling extra object $@..."
	mkdir -p $(@D)
	$(CXX) -fPIC -D_LARGEFILE64_SOURCE=1 -DHAVE_HIDDEN -DPIC -o $@ -c $< $(CXXFLAGS)

#------------------------------------------------------------
#                      host rules
CXX = g++
HOST_SRCS += host.cpp
HOST_OBJ_FILES = $(OBJ_DIR)/host.o

$(HOST_OBJ_FILES): $(OBJ_DIR)/host.o $(SRC_DIR)/host.cpp 
	@echo -e "----\nCompiling object host.cpp..."
	mkdir -p $(@D) 
	$(CXX) -fPIC -o $@ -c $< $(CXXFLAGS) $(LDFLAGS)

$(EXE_FILE): $(HOST_OBJ_FILES)
	@echo -e "----\nCompiling host executable..."
	mkdir -p $(BUILD_DIR)
	$(CXX) -fPIC -D_LARGEFILE64_SOURCE=1 -DHAVE_HIDDEN -o $@ $^ -L. $(LIBZLIB_NAME) $(LDXRM) -lboost_system -lboost_filesystem -lOpenCL

# ------------------------------------------------------------
#                      build rules

.PHONY: all help host xclbin cleanh cleank cleanall clean lib 

all: host xclbin

host: $(EXE_FILE)

xclbin: $(XCLBIN_FILE)

cleank:
	rm -f _x_temp*/*.xo
	rm -f $(BUILD_DIR)/*.xclbin
	rm -rf _x_temp*/_x.* _x_temp*/.Xil _x_temp*/profile_summary.* sample_*
	rm -rf _x_temp*/dltmp* _x_temp*/kernel_info.dat _x_temp*/*.log
	
cleanh:
	rm -rf $(EXE_FILE)
	rm -rf $(EXE_NAME)
	rm -rf $(LIBZLIB_NAME)
	rm -rf $(ZLIBVER)
	rm -rf *.so
	rm -rf *.so.1
	rm -rf *.zlib
	rm -rf *.orig
	rm -rf obj_/
	rm -rf listdir*
	-$(RMDIR) $(EXE_FILE)
	-$(RMDIR) vitis_* TempConfig system_estimate.xtxt *.rpt .run/
	-$(RMDIR) src/*.ll _vxx_* .Xil dltmp* xmltmp* *.log *.jou *.wcfg *.wdb

cleanall: cleanh cleank
	rm -rf $(BUILD_DIR)
	-$(RMDIR) _x_temp* $(CUR_DIR)/reports *.csv *.run_summary *.txt.* $(CUR_DIR)/obj_*
	-$(RMDIR) $(XFLIB_DIR)/common/data/*.xe2xd* $(XFLIB_DIR)/common/data/*.orig* 

clean: cleanh

# ------------------------------------------------------------
#                      simulation run

$(BUILD_DIR)/emconfig.json :
		emconfigutil --platform $(XPLATFORM) --od $(BUILD_DIR)

HOST_ARGS = -t ./sample.txt

ifeq ($(TARGET),sw_emu)
  RUN_ENV = export XCL_EMULATION_MODE=sw_emu
  EMU_CONFIG = $(BUILD_DIR)/emconfig.json
else ifeq ($(TARGET),hw_emu)
  RUN_ENV = export XCL_EMULATION_MODE=hw_emu
  EMU_CONFIG = $(BUILD_DIR)/emconfig.json
else ifeq ($(TARGET),hw)
  RUN_ENV = echo "TARGET=hw"
  EMU_CONFIG =
endif

RUN_ENV += export LD_LIBRARY_PATH=$(PWD):$(LD_LIBRARY_PATH)
RUN_ENV += export XILINX_LIBZ_XCLBIN=$(XCLBIN_FILE)
RUN_ENV += export XRT_INI_PATH=$(PWD)/xrt.ini

run: xclbin lib host $(EMU_CONFIG)
	$(RUN_ENV);  
ifeq ($(TARGET), $(filter $(TARGET), sw_emu hw_emu))
	XCL_EMULATION_MODE=$(TARGET) ./run.sh $(EXE_FILE) $(XFLIB_DIR) $(XCLBIN_FILE)
else 
	./run.sh $(EXE_FILE) $(XFLIB_DIR) $(XCLBIN_DIR)
endif
