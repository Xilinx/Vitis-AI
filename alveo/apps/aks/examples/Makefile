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

## Vitis-AI ROOT folder
ifeq "$(origin VAI_ALVEO_ROOT)" "undefined"
VAI_ALVEO_ROOT := ../../
endif

PWD := $(shell pwd)
###
# Includes
###
# Common
INCLUDE += -I $(CONDA_PREFIX)/include
# AKS
INCLUDE += -I $(VAI_ALVEO_ROOT)/apps/aks
###
# Libraries
###
LIB_PATHS := -L $(CONDA_PREFIX)/lib -L$(VAI_ALVEO_ROOT)/apps/aks/libs
# Common
LIBS := -lpthread -ldl -lz
# Boost
LIBS += -lboost_thread -lboost_system -lboost_filesystem
# Others
LIBS += -lglog -ljson-c -lhdf5_cpp -lhdf5 -lpthread -lpng16 -lprotobuf -lcaffe-nv
# Opencv
LIBS += -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

###
# Compiler
###
CXX := g++ 
# Flags
CXXOPTS :=-O3
CCOPTS := -D __USE_XOPEN2K8 -std=c++11 -fPIC -static-libgcc -static-libstdc++

###
# Examples
###
AKS_EXES := $(patsubst %.cpp,bin/%.exe,$(wildcard *.cpp))

###
# Make Targets
###
examples: $(AKS_EXES)
all: $(AKS_EXES)
	
.PHONY := examples

bin/%.exe: %.cpp
	@echo -e "make: $<"
	@mkdir -p bin
	@$(CXX) $(CXXOPT) $(CCOPTS) $< -o $@ $(INCLUDE) $(LIB_PATHS) $(LIBS) -lAks

clean:
	@echo -e "make: examples $@"
	@$(RM) -r bin/*.exe
	
ultraclean:
	@echo -e "make: examples $@"
	@$(RM) -r bin

