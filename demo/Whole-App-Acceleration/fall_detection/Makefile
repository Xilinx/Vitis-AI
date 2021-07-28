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

TMP_LD_PATH=${LD_LIBRARY_PATH}
LD_LIBRARY_PATH :=

###
# AKS
INCLUDE += -I /usr/include/aks
INCLUDE += -I /usr/include/xir
INCLUDE += -I /usr/include/vart
INCLUDE += -I src/include

CXX := g++ #$(GXX)
# Flags
CXXOPT :=-O3
CCOPTS :=-D __USE_XOPEN2K8 -std=c++17 -fPIC -static-libgcc -static-libstdc++

#####################
#        Linker
#####################
# common
LIB_PATHS := -L /usr/lib
LIB_PATHS += -L /usr/lib/x86_64-linux-gnu
LIB_PATHS += -L $(CONDA_PREFIX)/lib
# Add libs under current directory
LIB_PATHS += -L libs
# Add libs unders aks
LIB_PATHS += -L $(VAI_HOME)/tools/AKS/libs

# Common
LIBS := -lpthread -ldl -lz
# Boost
LIBS += -lboost_thread -lboost_system -lboost_filesystem
# Opencv
LIBS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs
# Opencv video
LIBS += -lopencv_videoio -lopencv_video
# AKS
LIBS += -laks -lvart-runner -lxir -lunilog

#####################
#      Rules/Targets
#####################
SRC := ./src/*.cpp
BIN := ./src/bin
$(shell mkdir -p $(BIN))
TARGET := $(BIN)/main.exe

$(TARGET): $(SRC)
	@echo -e "\n[BUILD] Building $@ \n"
	$(CXX) $(CXXOPT) $(CCOPTS) $(SRC) $(INCLUDE) $(LIB_PATHS) $(LIBS) -o $@
	@echo -e "\n[BUILD] Done\n"

LD_LIBRARY_PATH :=${TMP_LD_PATH}	

clean:
	rm -rf src/bin/*.exe
