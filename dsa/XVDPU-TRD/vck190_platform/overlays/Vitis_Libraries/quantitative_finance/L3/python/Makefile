
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

ifndef XILINX_FINTECH_L3_INC
$(error "XILINX_FINTECH_L3_INC is not set")
endif

ifndef XILINX_FINTECH_L2_INC
$(error "XILINX_FINTECH_L2_INC is not set")
endif


EXE_NAME = xf_fintech_python
EXE_EXT ?= so
EXE_FILE ?= $(EXE_NAME)$(if $(EXE_EXT),.,)$(EXE_EXT)

SRC_DIR = .
HOST_ARGS =
RUN_ENV =
INCLUDE_DIR = ../include
OUTPUT_DIR = ./output
PYTHON_INC_FLAGS = $(shell python3-config --includes) \
                         -I/usr/local/include/python3.6m/

SRCS := $(shell find $(SRC_DIR) -maxdepth 4 -name '*.cpp')
OBJ_FILES := $(addsuffix .o, $(basename $(SRCS)))
EXTRA_OBJS :=

CPPFLAGS = -DVITIS_PLATFORM=$(VITIS_PLATFORM) \
			-std=c++11 -O3 -g -Wall -Wno-unknown-pragmas -c -fPIC \
			-I$(XILINX_FINTECH_L3_INC) \
			-I$(XILINX_FINTECH_L2_INC) \
			-I$(XILINX_XCL2_DIR) \
			-I$(XILINX_XRT)/include \
			$(PYTHON_INC_FLAGS)

LDFLAGS = -shared  -lxilinxopencl -lxilinxfintech -lpthread -lrt -lstdc++ -L$(XILINX_XRT)/lib -L$(XILINX_FINTECH_LIB_DIR)


.PHONY: output all clean cleanall run

all: output $(EXE_FILE)

output:
	@mkdir -p ${OUTPUT_DIR}

clean:
	@$(RM) -rf $(OUTPUT_DIR)

cleanall: clean

run:
	@echo "library not executable"


%.o:%.cpp
	@echo $(notdir $(@))
	$(CXX) $(CPPFLAGS) -o ${OUTPUT_DIR}/$(notdir $(@)) -c $<


$(EXE_FILE): $(OBJ_FILES)
	$(CXX)   $(addprefix ${OUTPUT_DIR}/,$(notdir $(OBJ_FILES))) -o ${OUTPUT_DIR}/$@ $(LDFLAGS)
	
