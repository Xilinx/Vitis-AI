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

CROSS_COMPILE := aarch64-linux-gnu-
SYSROOT := /home/sysroots/aarch64-xilinx-linux

CC       =  $(CROSS_COMPILE)gcc
CXX      =  $(CROSS_COMPILE)g++

CUR_DIR  =  $(shell pwd)
HEADER   =  $(CUR_DIR)/include
SRC      =  $(CUR_DIR)/src
BUILD    =  $(CUR_DIR)/build
VPATH    =  $(SRC)

CFLAGS   =  -I. -I${SRC} -I${HEADER} -fPIC -std=c++11 -O2
CFLAGS  += -I${SYSROOT}/usr/include -I${SYSROOT}/usr/local/include -I../../n2cube/include

LDFLAGS  =  -L/lib -L/usr/lib -L/usr/local/lib
LDFLAGS +=  -ln2cube -lhineon -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_core -pthread -lxrt_core


CFLAGS  += -mcpu=cortex-a53 


OBJ      =  main.o
OBJ     +=  dputils.o

all: $(BUILD) resnet50 
.PHONY : clean 
 
resnet50:  $(OBJ)
	$(CXX) $(addprefix $(BUILD)/, $^)  -o $@ ${LDFLAGS} 

%.o : %.cc
	$(CXX)  -c $(CFLAGS) $< -o $(BUILD)/$@ 

$(BUILD):
	mkdir -p $@

clean:
	$(RM) $(BUILD)/* resnet50
