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

VPATH:= .
SRC  := dpu_xrt.cpp
OBJ  := $(patsubst %.cpp, %.o, $(notdir $(SRC)))

flavor := release

AOL_MAJOR_VERSION=1
AOL_MINOR_VERSION=1
AOL_REVISION_VERSION=0

ifeq ($(flavor), debug) 
CXXFLAGS = -g -DDPUDRV_DBG -DDPUDRV_INFO
BUILD	= debug
else 
CXXFLAGS = -O2
BUILD	= release
endif


ifneq ($(SYSROOT), )
    CC        =   $(CROSS_COMPILE)gcc
    CXX       =   $(CROSS_COMPILE)g++
    CXXFLAGS  +=   --sysroot=$(SYSROOT) -isystem$(SYSROOT)/usr/include/xrt -fPIC
    LDFLAGS   +=   --sysroot=$(SYSROOT) -lxrt_core -Wl,-no-undefined
else
    CXXFLAGS  += -std=c++11 -fPIC
    LDFLAGS   += -pthread -lxrt_core -Wl,-no-undefined
endif

include ../../../../Makefile.cfg

all: $(BUILD) libdpuaol.so 


libdpuaol.so : $(OBJ)
	@echo "Linking" $@
	@$(CXX) $(CXXFLAGS) -shared -Wl,-soname,$@.$(AOL_MAJOR_VERSION) $(addprefix $(BUILD)/, $^) -o $(BUILD)/$@.$(AOL_MAJOR_VERSION).$(AOL_MINOR_VERSION).$(AOL_REVISION_VERSION)  $(LDFLAGS)
	cd $(BUILD); rm -f $@
	cd $(BUILD); rm -f $@.$(AOL_MAJOR_VERSION)
	cd $(BUILD); ln -s $@.$(AOL_MAJOR_VERSION).$(AOL_MINOR_VERSION).$(AOL_REVISION_VERSION) $@.$(AOL_MAJOR_VERSION)
	cd $(BUILD); ln -s $@.$(AOL_MAJOR_VERSION) $@

$(OBJ):%.o : %.cpp
	@echo "compiling" $<
	@$(CXX) -c $(CXXFLAGS) $< -o $(BUILD)/$@

$(BUILD):
	@mkdir -p $@
clean:
	rm -rf debug release


