
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


ifndef XILINX_XRT
$(error "XILINX_XRT should be set on or after 2019.2 release.")
endif

ifndef XILINX_XCL2_DIR
$(error "XILINX_XCL2_DIR should be set to the directory containing xcl2")
endif

TARGET =
XF_PROJ_ROOT =
SRC_DIR = .
HOST_ARGS =
RUN_ENV =
INCLUDE_DIR = ../include
L2_INCLUDE_DIR = ../../L2/include/
BUILD_DIR = ./output

HOST_SRCS := $(shell find $(SRC_DIR) -name '*.cpp')
HOST_SRCS += $(shell find $(XILINX_XCL2_DIR) -name '*.cpp')

CXXFLAGS = -std=c++11 -O3 -g -Wall -Wno-unknown-pragmas -Wno-unused-label -fPIC
CXXFLAGS += -I$(L2_INCLUDE_DIR)
CXXFLAGS += -I$(INCLUDE_DIR)
CXXFLAGS += -Imodels/cf_black_scholes/include
CXXFLAGS += -Imodels/cf_black_scholes_merton/include
CXXFLAGS += -Imodels/cf_garman_kohlhagen/include
CXXFLAGS += -Imodels/hcf/include
CXXFLAGS += -Imodels/m76/include
CXXFLAGS += -Imodels/heston_fd/include
CXXFLAGS += -Imodels/mc_american/include
CXXFLAGS += -Imodels/mc_european/include
CXXFLAGS += -Imodels/binomial_tree/include
CXXFLAGS += -Imodels/mc_european_dje/include
CXXFLAGS += -Imodels/binomial_tree/include
CXXFLAGS += -Imodels/pop_mcmc/include
CXXFLAGS += -Imodels/hjm/include
CXXFLAGS += -Imodels/lmm/include
CXXFLAGS += -Imodels/cds/include
CXXFLAGS += -Imodels/hullwhite/include
CXXFLAGS += -Iutils/linear_interpolation/include
CXXFLAGS += -I$(XILINX_XCL2_DIR)
CXXFLAGS += -I$(XILINX_XRT)/include

LDFLAGS = -shared -lxilinxopencl -lpthread -lrt -lstdc++ -L$(XILINX_XRT)/lib

EXE_NAME := libxilinxfintech.so
EXE_FILE := $(BUILD_DIR)/$(EXE_NAME)


.PHONY: all clean cleanall run

all: $(EXE_FILE)

clean:
	rm -Rf $(EXE_FILE)

cleanall: clean

run: all

$(EXE_FILE): $(HOST_SRCS)
	mkdir -p $(BUILD_DIR)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)
