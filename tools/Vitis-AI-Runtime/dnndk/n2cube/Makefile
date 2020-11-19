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

HINEON_MAJOR_VERSION=1
HINEON_MINOR_VERSION=0
HINEON_REVISION_VERSION=0

N2CUBE_MAJOR_VERSION=4
N2CUBE_MINOR_VERSION=3
N2CUBE_REVISION_VERSION=0

ifneq ($(SYSROOT), )
    CC        =   $(CROSS_COMPILE)gcc
    CXX       =   $(CROSS_COMPILE)g++
    CFLAGS    =   --sysroot=$(SYSROOT) -I$(SYSROOT)/usr/include -I$(SYSROOT)/usr/local/include -I. -I./include -fPIC
    LDFLAGS   =   --sysroot=$(SYSROOT) -L$(SYSROOT)/usr/local/lib/ -L./build -Wl,-no-undefined
else
    CFLAGS   +=   -I. -I./include -fPIC
    LDFLAGS  +=   -L./build -Wl,-no-undefined
endif

CUR_DIR   =   $(shell pwd)
SRC_PATH  =   $(CUR_DIR)/src
BUILD     =   $(CUR_DIR)/build
COMMON_PATH = $(CUR_DIR)/../common
VPATH = $(SRC_PATH):$(COMMON_PATH)

include ../Makefile.cfg

COMMON_DIR = $(shell find $(COMMON_PATH) -maxdepth 3 -type d)
COMMON_SRCS_C    =  $(foreach dir, $(COMMON_DIR), $(wildcard $(dir)/*.c))
COMMON_OBJS_C = $(patsubst %.c, %.o, $(notdir $(COMMON_SRCS_C)))

DIRS      =   $(shell find $(SRC_PATH) -maxdepth 1 -type d)
SRCS_C    +=  $(foreach dir, $(DIRS), $(wildcard $(dir)/*.c))
SRCS_CPP  +=  $(foreach dir, $(DIRS), $(wildcard $(dir)/*.cpp))
OBJS_C    =   $(patsubst %.c, %.o, $(notdir $(SRCS_C)))
OBJS_CPP  =   $(patsubst %.cpp, %.o, $(notdir $(SRCS_CPP)))
n2cube_tmp1 = $(OBJS_C) $(OBJS_CPP)
n2cube_tmp2 = $(subst dputils.o,,$(n2cube_tmp1))
n2cube_tmp4 = $(subst neonopt.o,,$(n2cube_tmp2))
OBJS_N2CUBE = $(n2cube_tmp4)

LDFLAGS_HINEON   += $(LDFLAGS)  -pthread
LDFLAGS_N2CUBE   += $(LDFLAGS)  -pthread -lhineon -ldpuaol

CPPFLAGS = $(CFLAGS) -std=c++11

all: $(BUILD) libdpuaol.so libhineon.so libn2cube.so
 
libn2cube.so: $(OBJS_N2CUBE) $(COMMON_OBJS_C)
	$(CXX) $(CFLAGS) $(addprefix $(BUILD)/, $^) -shared -Wl,-soname,$@.$(N2CUBE_MAJOR_VERSION) -o $(BUILD)/$@.$(N2CUBE_MAJOR_VERSION).$(N2CUBE_MINOR_VERSION).$(N2CUBE_REVISION_VERSION) $(LDFLAGS_N2CUBE)
	cd $(BUILD); rm -f $@
	cd $(BUILD); rm -f $@.$(N2CUBE_MAJOR_VERSION)
	cd $(BUILD); ln -s $@.$(N2CUBE_MAJOR_VERSION).$(N2CUBE_MINOR_VERSION).$(N2CUBE_REVISION_VERSION) $@.$(N2CUBE_MAJOR_VERSION)
	cd $(BUILD); ln -s $@.$(N2CUBE_MAJOR_VERSION) $@

libhineon.so: neonopt.o
	$(CXX) $(CFLAGS) $(addprefix $(BUILD)/, $^) -shared -Wl,-soname,$@.$(HINEON_MAJOR_VERSION) -o $(BUILD)/$@.$(HINEON_MAJOR_VERSION).$(N2CUBE_MINOR_VERSION).$(HINEON_REVISION_VERSION) $(LDFLAGS_HINEON)
	cd $(BUILD); rm -f $@
	cd $(BUILD); rm -f $@.$(HINEON_MAJOR_VERSION)
	cd $(BUILD); ln -s $@.$(HINEON_MAJOR_VERSION).$(N2CUBE_MINOR_VERSION).$(HINEON_REVISION_VERSION) $@.$(HINEON_MAJOR_VERSION)
	cd $(BUILD); ln -s $@.$(HINEON_MAJOR_VERSION) $@

libdpuaol.so:
	@if [ -z "$(SYSROOT)" ]; then \
		make -C $(AOL_PATH); \
	else \
		make SYSROOT=$(SYSROOT) CROSS_COMPILE=$(CROSS_COMPILE) -C $(AOL_PATH); \
	fi
	cp -d $(AOL_PATH)/release/libdpuaol.so* $(BUILD)

$(OBJS_C):%.o : %.c
	$(CC) -c $(CFLAGS) $< -o $(BUILD)/$@

$(COMMON_OBJS_C):%.o : %.c
	$(CC) -c  $(CFLAGS) $< -o $(BUILD)/$@

$(OBJS_CPP):%.o : %.cpp
	$(CXX) -c $(CPPFLAGS) $< -o $(BUILD)/$@

$(BUILD):
	mkdir -p $@

clean:
	$(RM) $(BUILD)/*
	make clean -C ./src/aol/xrt/
	make clean -C ./src/aol/linux/

