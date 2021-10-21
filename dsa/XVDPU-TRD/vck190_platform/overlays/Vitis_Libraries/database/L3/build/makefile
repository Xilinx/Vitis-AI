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
TARGET ?= hw
DEVICE ?= xilinx_u280_xdma_201920_3
XDEVICE=$(basename $(notdir $(DEVICE)))
XPLATFORM ?= $(XDEVICE)

MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CUR_DIR := $(patsubst %/,%,$(dir $(MK_PATH)))
XF_PROJ_ROOT ?= $(shell bash -c 'export MK_PATH=$(MK_PATH); echo $${MK_PATH%L3/*}')
XF_LIB_DIR := $(abspath $(XF_PROJ_ROOT))

uniq = $(if $1,$(firstword $1) $(call uniq,$(filter-out $(firstword $1),$1)))
inc_paths = $(foreach _,$1,$(patsubst %xf_database/,%,$(dir $(abspath $_))))
inc_opts = $(patsubst %,-I%,$(call uniq,$(call inc_paths,$1)))

#Checks for g++
HOST_ARCH ?= x86
ifeq ($(HOST_ARCH), x86)
ifneq ($(shell expr $(shell g++ -dumpversion) \>= 5), 1)
ifndef XILINX_VIVADO
$(error [ERROR]: g++ version too old. Please install 5.0 or above)
else
CXX := $(XILINX_VIVADO)/tps/lnx64/gcc-6.2.0/bin/g++
$(warning [WARNING]: g++ version too old. Using g++ provided by the tool : $(CXX))
endif #XILINX_VIVADO
else
CXX := g++
endif
else ifeq ($(HOST_ARCH), aarch64)
CXX := $(XILINX_VITIS)/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-g++
else ifeq ($(HOST_ARCH), aarch32)
CXX := $(XILINX_VITIS)/gnu/aarch32/lin/gcc-arm-linux-gnueabi/bin/arm-linux-gnueabihf-g++
endif

#-------------------------------------------------------------------------------#
# User targets
.PHONY: all lib xclbin clean cleanall
all: lib xclbin

#-------------------------------------------------------------------------------#
#For library
LIB_SRCS = $(XF_LIB_DIR)/L3/src/sw/gqe_table.cpp \
       $(XF_LIB_DIR)/L3/src/sw/gqe_ocl.cpp \
       $(XF_LIB_DIR)/L3/src/sw/gqe_join_config.cpp \
       $(XF_LIB_DIR)/L3/src/sw/gqe_join.cpp \
       $(XF_LIB_DIR)/L3/src/sw/gqe_aggr_config.cpp \
       $(XF_LIB_DIR)/L3/src/sw/gqe_aggr.cpp

HDRS = $(XF_LIB_DIR)/L3/include/sw/xf_database/gqe_table.hpp \
       $(XF_LIB_DIR)/L3/include/sw/xf_database/gqe_ocl.hpp \
       $(XF_LIB_DIR)/L3/include/sw/xf_database/gqe_join_config.hpp \
       $(XF_LIB_DIR)/L3/include/sw/xf_database/gqe_join_strategy.hpp \
       $(XF_LIB_DIR)/L3/include/sw/xf_database/gqe_join.hpp \
       $(XF_LIB_DIR)/L3/include/sw/xf_database/gqe_aggr_config.hpp \
       $(XF_LIB_DIR)/L3/include/sw/xf_database/gqe_aggr_strategy.hpp \
       $(XF_LIB_DIR)/L3/include/sw/xf_database/gqe_aggr.hpp

LIB_SO := libvt_database_gqe.so
$(LIB_SO): $(LIB_SRCS) $(HDRS)
	$(CXX) -o $@ -fPIC -shared -g -std=c++14 -Wall -Wno-unknown-pragmas -Wno-unused-label \
		-Wno-narrowing -Wno-sign-compare \
		-I${XILINX_VITIS}/include \
		-I${XILINX_HLS}/include \
                -I$(XILINX_XRT)/include \
                -I$(XF_LIB_DIR)/L1/include/hw/ \
                -I$(XF_LIB_DIR)/L1/include/sw/ \
                -I$(XF_LIB_DIR)/L2/include/sw/ \
                -I$(XF_LIB_DIR)/L3/include/sw/ \
                -pthread -lOpenCL \
		$(LIB_SRCS)

#-------------------------------------------------------------------------------#
#For xclbin

XCLBIN_FILE_JOIN := $(XF_LIB_DIR)/L3/tests/gqe/join/build_dir.$(TARGET).$(XDEVICE)/gqe_join_part.xclbin
XCLBIN_FILE_AGGR := $(XF_LIB_DIR)/L3/tests/gqe/aggregate/build_dir.$(TARGET).$(XDEVICE)/gqe_aggr_part.xclbin
$(XCLBIN_FILE_JOIN):
	make -C $(XF_LIB_DIR)/L3/tests/gqe/join/ xclbin DEVICE=$(DEVICE) TARGET=$(TARGET)
$(XCLBIN_FILE_AGGR):
	make -C $(XF_LIB_DIR)/L3/tests/gqe/aggregate/ xclbin DEVICE=$(DEVICE) TARGET=$(TARGET)

XCLBIN_FILE_JOIN_DET := vt_database_join-$(XDEVICE).xclbin
XCLBIN_FILE_AGGR_DET := vt_database_aggr-$(XDEVICE).xclbin

$(XCLBIN_FILE_JOIN_DET): $(XCLBIN_FILE_JOIN)
	@echo "copy $(notdir $(XCLBIN_FILE_JOIN))"
	cp $(XCLBIN_FILE_JOIN) $(XCLBIN_FILE_JOIN_DET)

$(XCLBIN_FILE_AGGR_DET): $(XCLBIN_FILE_AGGR)
	@echo "copy $(notdir $(XCLBIN_FILE_AGGR))"
	cp $(XCLBIN_FILE_AGGR) $(XCLBIN_FILE_AGGR_DET)

#-------------------------------------------------------------------------------#

lib: $(LIB_SO)

xclbin: $(XCLBIN_FILE_JOIN_DET) $(XCLBIN_FILE_AGGR_DET)

clean:
	rm -rf $(LIB_SO)

cleanall:
	rm -rf $(LIB_SO) \
	$(XCLBIN_FILE_JOIN_DET) \
	$(XCLBIN_FILE_AGGR_DET)


