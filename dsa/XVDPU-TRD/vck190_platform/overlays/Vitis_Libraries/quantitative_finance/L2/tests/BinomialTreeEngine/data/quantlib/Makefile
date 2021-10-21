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

EXE_NAME = bt_quantlib
EXE_EXT ?= exe
EXE_FILE ?= $(EXE_NAME)$(if $(EXE_EXT),.,)$(EXE_EXT)

SRC_DIR = .
HOST_ARGS =
RUN_ENV =
INCLUDE_DIR = ../
XF_FINTECH_L2_DIR = ../../../../include
OUTPUT_DIR = ./output

SRCS := $(shell find $(SRC_DIR) -maxdepth 1 -name '*.cpp')
OBJ_FILES := $(addsuffix .o, $(basename $(SRCS)))
EXTRA_OBJS :=

CPPFLAGS = -std=c++11 -O3 -Wall -Wno-unknown-pragmas -c -I$(INCLUDE_DIR) -I$(XF_FINTECH_L2_DIR)
LDFLAGS = -lpthread -lstdc++ -lQuantLib -L/usr/local/lib

# not used
KERNELS=
KSRC_DIR=


.PHONY: output all clean cleanall run

all: output $(EXE_FILE)

output:
	@mkdir -p ${OUTPUT_DIR}

clean:
	@$(RM) -rf $(OUTPUT_DIR)

cleanall: clean

run:
	${OUTPUT_DIR}/$(EXE_FILE) $(HOST_ARGS)

host:
xclbin:
build:
	@echo "unused targets"


%.o:%.cpp
	@echo $(notdir $(@))
	$(CXX) $(CPPFLAGS) -o ${OUTPUT_DIR}/$(notdir $(@)) -c $<


$(EXE_FILE): $(OBJ_FILES)
	$(CXX) $(LDFLAGS) -o ${OUTPUT_DIR}/$@ $(addprefix ${OUTPUT_DIR}/,$(notdir $(OBJ_FILES)))
