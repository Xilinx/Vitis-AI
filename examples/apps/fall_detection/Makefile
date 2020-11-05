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

## uuid_copy@UUID_1.0 undifne issue due to conflict between the docker and conda libs
TMP_LD_PATH=${LD_LIBRARY_PATH}
LD_LIBRARY_PATH :=

BUILD := $(VAI_ALVEO_ROOT)/algorithms/fall_detection/build
$(shell mkdir -p $(BUILD))

###
INCLUDE += -I$(CONDA_PREFIX)/include
# AKS
INCLUDE += -I $(VAI_ALVEO_ROOT)/apps/aks

CXX := g++ #$(GXX)
# Flags
CXXOPT :=-O3
CCOPTS :=-D __USE_XOPEN2K8 -std=c++11 -fPIC -static-libgcc -static-libstdc++

#####################
#        Linker
#####################
# common
LIB_PATHS := -L $(CONDA_PREFIX)/lib
# Add libs under current directory
LIB_PATHS += -L libs
# Add libs unders aks
LIB_PATHS += -L $(VAI_ALVEO_ROOT)/apps/aks/libs

# Common
LIBS := -lpthread -ldl -lz
# Boost
LIBS += -lboost_thread -lboost_system -lboost_filesystem
# Others
LIBS += -lglog -ljson-c -lhdf5_cpp -lhdf5 -lpthread -lpng16 -lprotobuf -lcaffe-nv
# Opencv
LIBS += -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
# Opencv video
LIBS += -lopencv_videoio -lopencv_video
# Custom
LIBS += -lAks

#####################
#      Rules/Targets
#####################

AKS_KERNELS := $(wildcard kernel_src/*/)
SRC := ./src/main.cc

TARGET := $(BUILD)/fall_detection.exe

$(TARGET): $(SRC)
	@echo -e "\n[BUILD] Building $@ \n\n"
	$(CXX) $(CXXOPT) $(CCOPTS) -std=c++11 $(SRC) $(INCLUDE) $(LIB_PATHS) $(LIBS) -o $@
	@echo -e "\n[BUILD] Done\n\n"

all: $(TARGET) $(AKS_KERNELS)

$(AKS_KERNELS):
	@echo -e 	
	@echo -e "make: $@"
	@$(MAKE) --no-print-directory -C $@

.PHONY: all $(AKS_KERNELS) clean

LD_LIBRARY_PATH :=${TMP_LD_PATH}	

clean :
	rm -rf build/*exe
	rm -rf libs/*
