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
#

MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CUR_DIR := $(patsubst %/,%,$(dir $(MK_PATH)))
XF_PROJ_ROOT ?= $(shell bash -c 'export MK_PATH=$(MK_PATH); echo $${MK_PATH%L1/tests/*}')

# MK_INC_BEGIN hls_common.mk

.PHONY: help

help::
	@echo ""
	@echo "Makefile Usage:"
	@echo ""
	@echo "  make run CSIM=1 CSYNTH=1 COSIM=1 DEVICE=<FPGA platform> PLATFORM_REPO_PATHS=<path to platform directories>"
	@echo "      Command to run the selected tasks for specified device."
	@echo ""
	@echo "      Valid tasks are CSIM, CSYNTH, COSIM, VIVADO_SYN, VIVADO_IMPL"
	@echo ""
	@echo "      DEVICE is case-insensitive and support awk regex."
	@echo "      For example, \`make run DEVICE='u200.*xdma' COSIM=1\`"
	@echo "      It can also be an absolute path to platform file."
	@echo ""
	@echo "      PLATFORM_REPO_PATHS variable is used to specify the paths in which the platform files will be"
	@echo "      searched for."
	@echo ""
	@echo "  make run CSIM=1 CSYNTH=1 COSIM=1 XPART=<FPGA part name>"
	@echo "      Alternatively, the FPGA part can be speficied via XPART."
	@echo "      For example, \`make run XPART='xcu200-fsgd2104-2-e' COSIM=1\`"
	@echo "      When XPART is set, DEVICE will be ignored."
	@echo ""
	@echo "  make clean "
	@echo "      Command to remove the generated files."
	@echo ""

# MK_INC_END hls_common.mk

# MK_INC_BEGIN vivado.mk

TOOL_VERSION ?= 2020.2

ifeq (,$(XILINX_VIVADO))
XILINX_VIVADO = /opt/xilinx/Vivado/$(TOOL_VERSION)
endif
export XILINX_VIVADO

.PHONY: check_vivado
check_vivado:
ifeq (,$(wildcard $(XILINX_VIVADO)/bin/vivado))
	@echo "Cannot locate Vivado installation. Please set XILINX_VIVADO variable." && false
endif

export PATH := $(XILINX_VIVADO)/bin:$(PATH)

# MK_INC_END vivado.mk

DEVICE ?= u200
# MK_INC_BEGIN vitis_set_part.mk

.PHONY: check_part

ifeq (,$(XPART))
# MK_INC_BEGIN vitis.mk

TOOL_VERSION ?= 2020.2

ifeq (,$(XILINX_VITIS))
XILINX_VITIS = /opt/xilinx/Vitis/$(TOOL_VERSION)
endif
export XILINX_VITIS
.PHONY: check_vpp
check_vpp:
ifeq (,$(wildcard $(XILINX_VITIS)/bin/v++))
	@echo "Cannot locate Vitis installation. Please set XILINX_VITIS variable." && false
endif

ifeq (,$(XILINX_XRT))
XILINX_XRT = /opt/xilinx/xrt
endif
export XILINX_XRT
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
Avaialble platforms are: $(XPLATFORMS)
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

# MK_INC_END vitis_set_platform.mk
ifeq (1, $(words $(XPLATFORM)))
# Query the part name of device
ifneq (,$(wildcard $(XILINX_VITIS)/bin/platforminfo))
override XPART := $(shell $(XILINX_VITIS)/bin/platforminfo --json="hardwarePlatform.devices[0].fpgaPart" --platform $(firstword $(XPLATFORM)) | sed 's/^[^:]*://g' | sed 's/[^a-zA-Z0-9]/-/g' | sed 's/-\+/-/g')
endif
endif
check_part: | check_platform
ifeq (,$(XPART))
	@echo "XPART is not set and cannot be inferred. Please run \`make help\` for usage info." && false
endif
else # XPART
check_part:
	@echo "XPART is directly set to $(XPART)"
endif # XPART

.PHONY: run setup runhls clean

CSIM ?= 0
CSYNTH ?= 0
COSIM ?= 0
VIVADO_SYN ?= 0
VIVADO_IMPL ?= 0
QOR_CHECK ?= 0


# at least RTL synthesis before check QoR
ifeq (1,$(QOR_CHECK))
ifeq (0,$(VIVADO_IMPL))
override VIVADO_SYN := 1
endif
endif

# need synthesis before cosim or vivado
ifeq (1,$(VIVADO_IMPL))
override CSYNTH := 1
endif

ifeq (1,$(VIVADO_SYN))
override CSYNTH := 1
endif

ifeq (1,$(COSIM))
override CSYNTH := 1
endif

run: setup runhls

setup: | check_part
	@rm -f ./settings.tcl
	@if [ -n "$$CLKP" ]; then echo 'set CLKP $(CLKP)' >> ./settings.tcl ; fi
	@echo 'set XPART $(XPART)' >> ./settings.tcl
	@echo 'set CSIM $(CSIM)' >> ./settings.tcl
	@echo 'set CSYNTH $(CSYNTH)' >> ./settings.tcl
	@echo 'set COSIM $(COSIM)' >> ./settings.tcl
	@echo 'set VIVADO_SYN $(VIVADO_SYN)' >> ./settings.tcl
	@echo 'set VIVADO_IMPL $(VIVADO_IMPL)' >> ./settings.tcl
	@echo 'set QOR_CHECK $(QOR_CHECK)' >> ./settings.tcl
	@echo 'set XF_PROJ_ROOT "$(XF_PROJ_ROOT)"' >> ./settings.tcl
	@echo "Configured: settings.tcl"
	@echo "----"
	@cat ./settings.tcl
	@echo "----"

runhls: setup | check_vivado
	vitis_hls -f run_hls.tcl;

clean:
	rm -rf *.prj *_hls.log settings.tcl

