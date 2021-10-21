#
# Copyright 2019 Xilinx, Inc.
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
	@echo "  make OP=dot XPART=<FPGA part name> run CSIM=1 CSYNTH=1 COSIM=1 BENCHMAR=1"
	@echo "      Command to run the selected tasks for specified device."
	@echo ""
	@echo "      Valid tasks are CSIM, CSYNTH, COSIM, BENCHMARK"
	@echo ""
	@echo "      OP is the operator to be tested, mandatory."
	@echo ""
	@echo "      For example, \`make run XPART='xcu200-fsgd2104-2-e' OP=dot BENCHMARK=1\`"
	@echo ""
	@echo "  make clean "
	@echo "      Command to remove the generated files."
	@echo ""

# MK_INC_END hls_common.mk

# MK_INC_BEGIN vivado.mk

TOOL_VERSION ?= 2019.1

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

# MK_INC_BEGIN vitis_set_part.mk

.PHONY: check_part

XPART ?= 'xcu200-fsgd2104-2-e'
check_part:
	@echo "XPART is directly set to $(XPART)"

# MK_INC_END vitis_set_part.mk

# MK_INC_BEGIN hls_test_rules.mk

.PHONY: run setup runhls clean

ifeq (,$(OP))
	@echo "OP is not set and cannot be inferred. Please run \`make help\` for usage info." && false
endif

BENCHMARK ?= 1

PFLAGS = --override --operator $(OP) --xpart $(XPART)

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

ifeq (1,$(BENCHMARK))
PFLAGS += --benchmark
endif

ifeq (1,$(COSIM))
PFLAGS += --cosim
endif

ifeq (1,$(CSYNTH))
PFLAGS += --csynth
endif

run: setup runhls

setup: | check_part

PYTHON ?= python3
runhls: setup | check_vivado
	$(PYTHON) run_test.py $(PFLAGS)

clean:
	rm -rf out_test/

.PHONY: check
check: run

# MK_INC_END hls_test_rules.mk
