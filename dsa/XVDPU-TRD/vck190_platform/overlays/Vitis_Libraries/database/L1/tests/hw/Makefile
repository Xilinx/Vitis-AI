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

#

.PHONY: help

help::
	@echo ""
	@echo "Makefile Usage:"
	@echo ""
	@echo "  make run DEVICE=<FPGA platform> CSIM=1 CSYNTH=1 COSIM=1"
	@echo "      Command to run the selected tasks for specified device."
	@echo ""
	@echo "      Valid tasks are CSIM, CSYNTH, COSIM, VIVADO_SYN, VIVADO_IMPL"
	@echo ""
	@echo "      DEVICE is case-insensitive and support awk regex."
	@echo "      For example, \`make run DEVICE='u200.*qdma' CSIM=1\`"
	@echo "      It can also be an absolute path to platform file."
	@echo ""
	@echo "  make run XPART=<FPGA part name> CSIM=1"
	@echo "      Alternatively, the FPGA part can be speficied via XPART."
	@echo "      For example, \`make run XPART='xcu200-fsgd2104-2-e' CSIM=1\`"
	@echo "      When XPART is set, DEVICE will be ignored."
	@echo ""
	@echo "  make clean "
	@echo "      Command to remove the generated files."
	@echo ""

DEVICE ?= u200

CSIM ?= 1
CSYNTH ?= 1
COSIM ?= 1
VIVADO_SYN ?= 0
VIVADO_IMPL ?= 0
QOR_CHECK ?= 0

export CSIM CSYNTH COSIM VIVADO_SYN VIVADO_IMPL QOR_CHECK DEVICE XPART

SUBDIRS = $(dir $(shell find ./ -name "run_hls.tcl" ! -path '*/jks.*'))

$(info SUBDIRS are $(SUBDIRS))

.PHONY: run setup clean

run: $(SUBDIRS)

setup: $(SUBDIRS)

clean: $(SUBDIRS)

$(SUBDIRS)::
	make -C $@ $(MAKECMDGOALS)
