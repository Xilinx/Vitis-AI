## Copyright 2019 Xilinx Inc.
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

# ----------------------------------------------------------
#  Global Info
# ----------------------------------------------------------

## Vitis-AI ROOT folder
ifeq "$(origin VAI_ALVEO_ROOT)" "undefined"
VAI_ALVEO_ROOT := ../../../../
endif

# Source Dir
COMMON_REPO := $(shell pwd)
DST_DIR := $(VAI_ALVEO_ROOT)/algorithms/fall_detection/libs
$(shell mkdir -p $(DST_DIR))

# Include Paths
INCLUDE += -I ${CONDA_PREFIX}/include/
INCLUDE += -I $(VAI_ALVEO_ROOT)/apps/aks

# Library Paths
LIB_PATHS += -L $(CONDA_PREFIX)/lib
LIB_PATHS += -L $(VAI_ALVEO_ROOT)/apps/aks/libs
# Libraries
LIBS += -lpthread -lAks -lrt

# Compiler Flags
CXXFLAGS := -O3 -Wno-unused -Wno-maybe-uninitialized -Wno-unknown-pragmas -w
CCOPTS := -fPIC -shared 

# Kernel
APP_NAME := FallDetectionEvaluation
APP_SRCS := $(COMMON_REPO)/*.cpp

# Target
target := lib$(APP_NAME).so
APP_LIB := $(DST_DIR)/$(target)

$(APP_LIB): $(APP_SRCS)
	@echo -e "make: $(target)"
	@$(CXX) $(CXXFLAGS) $^ ${INCLUDE} $(LIB_PATHS) $(LIBS) $(CCOPTS) -o $@

.PHONY: all
all: $(LIB_NAME)

.PHONY: clean

clean:
	rm -rf *.so
