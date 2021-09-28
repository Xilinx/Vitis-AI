#
# Copyright 2018-2019 Xilinx, Inc.
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
	@echo "  make check DEVICE=<FPGA platform>"
	@echo "      Command to synthesis and run the test for specified Device."
	@echo ""
	@echo "      DEVICE is case-insensitive and support awk regex."
	@echo "      For example, \`make check DEVICE='u200'\`"
	@echo "      The default value is U200"
	@echo ""
	@echo "  make clean "
	@echo "      Command to remove the generated files."
	@echo ""

DEVICE ?= U200

CSIM ?= 1
CSYNTH ?= 1
COSIM ?= 1
VIVADO_SYN ?= 0
VIVADO_IMPL ?= 0
QOR_CHECK ?= 0

export CSIM CSYNTH COSIM VIVADO_SYN VIVADO_IMPL QOR_CHECK DEVICE

#SUBDIRS = $(wildcard */)
SUBDIRS = $(dir $(shell find ./ -name "run_hls.tcl"))

$(info SUBDIRS are $(SUBDIRS))

.PHONY: check run setup clean

check: $(SUBDIRS)
run: check

setup: $(SUBDIRS)

clean: $(SUBDIRS)

$(SUBDIRS)::
	make -C $@ $(MAKECMDGOALS)
