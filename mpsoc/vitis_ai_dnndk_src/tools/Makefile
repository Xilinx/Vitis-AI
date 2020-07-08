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

ifneq ($(SYSROOT), )
    CC        =   $(CROSS_COMPILE)gcc
    CXX       =   $(CROSS_COMPILE)g++
    CFLAGS    =   --sysroot=$(SYSROOT) -I$(SYSROOT)/usr/include -I$(SYSROOT)/usr/local/include -I. -I./include -fPIC
    LDFLAGS   =   --sysroot=$(SYSROOT) -L$(SYSROOT)/usr/local/lib/ -L./build -Wl,-no-undefined
else
    CFLAGS   +=   -I. -I./include -fPIC
    LDFLAGS  +=   -L./build -Wl,-no-undefined
endif

CFLAGS   +=   -I.
LDFLAGS  +=   -ldl -Wl,-no-undefined

CUR_DIR   =   $(shell pwd)
SRC       =   $(CUR_DIR)/src
BUILD     =   $(CUR_DIR)/build
COMMON_PATH = $(CUR_DIR)/../common
VPATH     =   $(SRC):$(COMMON_DIR)

include ../Makefile.cfg

COMMON_DIR = $(shell find $(COMMON_PATH) -maxdepth 3 -type d)
COMMON_SRCS_C    =  $(foreach dir, $(COMMON_DIR), $(wildcard $(dir)/*.c))
COMMON_OBJS_C = $(patsubst %.c, %.o, $(notdir $(COMMON_SRCS_C)))

LDFLAGS_DEXPLORER = $(LDFLAGS) -L../n2cube/build -ln2cube -lhineon -ldpuaol -lpthread

all: $(BUILD) dexplorer dsight ddump
.PHONY : clean
 
dexplorer: dexplorer.o 
	$(CXX) $(CFLAGS) $(addprefix $(BUILD)/, $^) -o $(BUILD)/$@ $(LDFLAGS_DEXPLORER)

dsight: dsight.o
	$(CC) $(CFLAGS) $(addprefix $(BUILD)/, $^) -o $(BUILD)/$@ $(LDFLAGS)

ddump: ddump.o $(COMMON_OBJS_C)
	$(CC) $(CFLAGS) $(addprefix $(BUILD)/, $^) -o $(BUILD)/$@ $(LDFLAGS)

%.o : %.c
	$(CC) -c $(CFLAGS) $< -o $(BUILD)/$@

%.o : %.cpp
	$(CXX) -c $(CFLAGS) $< -o $(BUILD)/$@

$(COMMON_OBJS_C):%.o : %.c
	$(CC) -c $(CFLAGS) $< -o $(BUILD)/$@

$(BUILD):
	mkdir -p $@

clean:
	$(RM) $(BUILD)/*
