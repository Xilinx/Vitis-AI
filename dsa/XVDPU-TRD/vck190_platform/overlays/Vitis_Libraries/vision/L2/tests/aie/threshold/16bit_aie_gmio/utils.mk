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
#+-------------------------------------------------------------------------------
# The following parameters are assigned with default values. These parameters can
# be overridden through the make command line
#+-------------------------------------------------------------------------------

REPORT := no
PROFILE := no
DEBUG := no

#'estimate' for estimate report generation
#'system' for system report generation
ifneq ($(REPORT), no)
VPP_LDFLAGS += --report estimate
VPP_LDFLAGS += --report system
endif

#Generates profile summary report
ifeq ($(PROFILE), yes)
VPP_LDFLAGS += --profile_kernel data:all:all:all
endif

#Generates debug summary report
ifeq ($(DEBUG), yes)
VPP_LDFLAGS += --dk protocol:all:all:all
endif

#Check environment setup
ifndef XILINX_VITIS
  XILINX_VITIS = /opt/xilinx/Vitis/$(TOOL_VERSION)
  export XILINX_VITIS
endif
ifndef XILINX_XRT
  XILINX_XRT = /opt/xilinx/xrt
  export XILINX_XRT
endif

check_device:
	@set -eu; \
	inallowlist=False; \
	inblocklist=False; \
	for dev in $(PLATFORM_ALLOWLIST); \
	    do if [[ $$(echo $(XDEVICE) | grep $$dev) != "" ]]; \
		then inallowlist=True; fi; \
	done ;\
	for dev in $(PLATFORM_BLOCKLIST); \
	    do if [[ $$(echo $(XDEVICE) | grep $$dev) != "" ]]; \
		then inblocklist=True; fi; \
	done ;\
	if [[ $$inallowlist == False ]]; \
	    then echo "[Warning]: The device $(XDEVICE) not in allowlist."; \
	fi; \
	if [[ $$inblocklist == True ]]; \
	    then echo "[ERROR]: The device $(XDEVICE) in blocklist."; exit 1;\
	fi;

#get HOST_ARCH by DEVICE
HOST_ARCH_temp = $(shell platforminfo -p $(DEVICE) | grep 'CPU Type' | sed 's/.*://' | sed '/ai_engine/d' | sed 's/^[[:space:]]*//')
$(warning HOST_ARCH_temp:$(HOST_ARCH_temp))
ifeq ($(HOST_ARCH_temp), x86)
HOST_ARCH := x86
else ifeq ($(HOST_ARCH_temp), cortex-a9)
HOST_ARCH := aarch32
else ifeq ($(HOST_ARCH_temp), cortex-a*)
HOST_ARCH := aarch64
endif

#Checks for Device Family
ifeq ($(HOST_ARCH), aarch32)
	DEV_FAM = 7Series
else ifeq ($(HOST_ARCH), aarch64)
	DEV_FAM = Ultrascale
endif

B_NAME = $(shell dirname $(XPLATFORM))

#Checks for Correct architecture
ifneq ($(HOST_ARCH), $(filter $(HOST_ARCH),aarch64 aarch32 x86))
$(error HOST_ARCH variable not set, please set correctly and rerun)
endif

check_version:
ifneq (, $(shell which git))
ifneq (,$(wildcard $(XFLIB_DIR)/.git))
	@cd $(XFLIB_DIR) && git log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit -n 1 && cd -
endif
endif

#Checks for SYSROOT
check_sysroot:
ifneq ($(HOST_ARCH), x86)
ifndef SYSROOT
	$(error SYSROOT ENV variable is not set, please set ENV variable correctly and rerun)
endif
endif

#Checks for g++
CXX := g++
ifeq ($(HOST_ARCH), x86)
ifneq ($(shell expr $(shell g++ -dumpversion) \>= 5), 1)
ifndef XILINX_VIVADO
$(error [ERROR]: g++ version too old. Please use 5.0 or above)
else
CXX := $(XILINX_VIVADO)/tps/lnx64/gcc-6.2.0/bin/g++
ifeq ($(LD_LIBRARY_PATH),)
export LD_LIBRARY_PATH := $(XILINX_VIVADO)/tps/lnx64/gcc-6.2.0/lib64
else
export LD_LIBRARY_PATH := $(XILINX_VIVADO)/tps/lnx64/gcc-6.2.0/lib64:$(LD_LIBRARY_PATH)
endif
$(warning [WARNING]: g++ version too old. Using g++ provided by the tool: $(CXX))
endif
endif
else ifeq ($(HOST_ARCH), aarch64)
CXX := $(XILINX_VITIS)/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-g++
else ifeq ($(HOST_ARCH), aarch32)
CXX := $(XILINX_VITIS)/gnu/aarch32/lin/gcc-arm-linux-gnueabi/bin/arm-linux-gnueabihf-g++
endif

#Check OS and setting env
OSDIST = $(shell lsb_release -i |awk -F: '{print tolower($$2)}' | tr -d ' \t' )
OSREL = $(shell lsb_release -r |awk -F: '{print tolower($$2)}' |tr -d ' \t')

ifeq ($(OSDIST), centos)
ifeq (7,$(shell echo $(OSREL) | awk -F. '{print tolower($$1)}' ))
ifeq ($(HOST_ARCH), x86)
CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=0
endif
endif
endif

#Setting VPP
VPP := v++

#Cheks for aiecompiler
AIECXX := aiecompiler
AIESIMULATOR := aiesimulator
X86SIMULATOR := x86simulator

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
ifeq ($(HOST_ARCH), x86)
ifeq (,$(LD_LIBRARY_PATH))
LD_LIBRARY_PATH := $(XILINX_XRT)/lib
else
LD_LIBRARY_PATH := $(XILINX_XRT)/lib:$(LD_LIBRARY_PATH)
endif
else # aarch64
ifeq (,$(LD_LIBRARY_PATH))
LD_LIBRARY_PATH := $(SYSROOT)/usr/lib 
else
LD_LIBRARY_PATH := $(SYSROOT)/usr/lib:$(LD_LIBRARY_PATH) 
endif
endif

ifneq (,$(wildcard $(DEVICE)))
# Use DEVICE as a file path
XPLATFORM := $(DEVICE)
else
# Use DEVICE as a file name pattern
# 1. search paths specified by variable
ifneq (,$(PLATFORM_REPO_PATHS))
# 1.1 as exact name
XPLATFORM := $(strip $(foreach p, $(subst :, ,$(PLATFORM_REPO_PATHS)), $(wildcard $(p)/$(DEVICE)/$(DEVICE).xpfm)))
# 1.2 as a pattern
ifeq (,$(XPLATFORM))
XPLATFORMS := $(foreach p, $(subst :, ,$(PLATFORM_REPO_PATHS)), $(wildcard $(p)/*/*.xpfm))
XPLATFORM := $(strip $(foreach p, $(XPLATFORMS), $(shell echo $(p) | awk '$$1 ~ /$(DEVICE)/')))
endif # 1.2
endif # 1
# 2. search Vitis installation
ifeq (,$(XPLATFORM))
# 2.1 as exact name
XPLATFORM := $(strip $(wildcard $(XILINX_VITIS)/platforms/$(DEVICE)/$(DEVICE).xpfm))
# 2.2 as a pattern
ifeq (,$(XPLATFORM))
XPLATFORMS := $(wildcard $(XILINX_VITIS)/platforms/*/*.xpfm)
XPLATFORM := $(strip $(foreach p, $(XPLATFORMS), $(shell echo $(p) | awk '$$1 ~ /$(DEVICE)/')))
endif # 2.2
endif # 2
# 3. search default locations
ifeq (,$(XPLATFORM))
# 3.1 as exact name
XPLATFORM := $(strip $(wildcard /opt/xilinx/platforms/$(DEVICE)/$(DEVICE).xpfm))
# 3.2 as a pattern
ifeq (,$(XPLATFORM))
XPLATFORMS := $(wildcard /opt/xilinx/platforms/*/*.xpfm)
XPLATFORM := $(strip $(foreach p, $(XPLATFORMS), $(shell echo $(p) | awk '$$1 ~ /$(DEVICE)/')))
endif # 3.2
endif # 3
endif

define MSG_PLATFORM
No platform matched pattern '$(DEVICE)'.
Available platforms are: $(XPLATFORMS)
To add more platform directories, set the PLATFORM_REPO_PATHS variable or point DEVICE variable to the full path of platform .xpfm file.
endef
export MSG_PLATFORM

define MSG_DEVICE
More than one platform matched: $(XPLATFORM)
Please set DEVICE variable more accurately to select only one platform file, or set DEVICE variable to the full path of the platform .xpfm file.
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
#Check ends

#   device2xsa - create a filesystem friendly name from device name
#   $(1) - full name of device
XDEVICE = $(strip $(patsubst %.xpfm, % , $(shell basename $(DEVICE))))


# Cleaning stuff
RM = rm -f
RMDIR = rm -rf

MV = mv -f
CP = cp -rf
ECHO:= @echo
