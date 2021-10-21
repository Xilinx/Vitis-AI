#
# Copyright 2020 Xilinx, Inc.
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
DEVICE ?= xilinx_u200_xdma_201830_2
XDEVICE=$(basename $(notdir $(DEVICE)))
SS ?= 1
LN ?= 1
VERSION ?=
TARGET ?= sw_emu

MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CUR_DIR := $(patsubst %/,%,$(dir $(MK_PATH)))
PROJ_DIR := $(CUR_DIR)/../
BUILD_DIR := $(PROJ_DIR)/build_dir.$(TARGET).$(XDEVICE)

DATA_EXE := $(BUILD_DIR)/gendata$(VERSION).exe
DATA_SRC = gendata.cpp
DATA_HDR = x_utils.hpp
DATA_DIR = $(BUILD_DIR)/demo_data$(VERSION)
DATA_FILE = $(DATA_DIR)/.stamp$(SS)_$(LN)$(VERSION)

.PHONY: clean data
data: $(DATA_FILE)

uniq = $(if $1,$(firstword $1) $(call uniq,$(filter-out $(firstword $1),$1)))
inc_paths = $(foreach _,$1,$(patsubst %xf_database/,%,$(dir $(abspath $_))))
inc_opts = $(patsubst %,-I%,$(call uniq,$(call inc_paths,$1)))

$(DATA_EXE): $(DATA_SRC) $(DATA_HDR)
	mkdir -p $(BUILD_DIR)
	g++ -std=c++11 -o $@ $(DATA_SRC) \
		-I${XILINX_VIVADO}/include  \
                -I$(XILINX_XRT)/include \
                -pthread  -L$(XILINX_XRT)/lib \
                -g -Wall -Wno-unknown-pragmas -Wno-unused-label \
                -DUSER_DEBUG -DXDEVICE=$(XDEVICE) 

$(DATA_FILE): $(DATA_EXE)
	mkdir -p $(DATA_DIR)
	touch $(DATA_FILE)
	$(abspath $(DATA_EXE)) -ss $(SS) -ln $(LN) -out $(DATA_DIR) 
	rm -f $(DATA_DIR)/input.txt $(DATA_DIR)/output.txt
	ls $(DATA_DIR)/*.dat > $(DATA_DIR)/input.txt
	sed -i 's/^/\/home\/nimbix\//g' $(DATA_DIR)/input.txt
	sed 's/input/output/' $(DATA_DIR)/input.txt > $(DATA_DIR)/output.txt

clean:
	rm -f $(DATA_EXE)

cleanall:
	rm -rf $(DATA_DIR) $(DATA_EXE) 

