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

# uuid_copy@UUID_1.0 undifne issue due to conflict between the docker and conda libs
TMP_LD_PATH=${LD_LIBRARY_PATH}
LD_LIBRARY_PATH :=

OPENCV_SYM=-lopencv_imgcodecs -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lopencv_video
XBLASOPTS :=-O3 -D MKL -D ZMQ -D XBLAS
#XBLASOPTS :=-O3 -D CBLAS -D ZMQ -D XBLAS
#XBLASOPTS :=-g -O0 -D MKL  -D ZMQ -D XBLAS -D DEBUG
BUILD := build
$(shell mkdir -p $(BUILD))

OUTPUT:= output
$(shell mkdir -p $(OUTPUT))

XILINX_XRT := /opt/xilinx/xrt

###
# Add XRT include
###
INCLUDE += -I$(XILINX_XRT)/include

###
# ADD OPENCV include 
###
INCLUDE += -I$(CONDA_PREFIX)/include

###
# ADD vitis include 
###
INCLUDE += -I$(CONDA_PREFIX)/lib/python3.6/site-packages/vai/dpuv1/rt/vitis/include

###
# Add common include
###
INCLUDE += -I$(CONDA_PREFIX)/include -I$(CONDA_PREFIX)/include/xip/butler 

CXX := g++ #$(GXX)

#####################
#        Linker
#####################
# common
LIB_PATHS := -L $(CONDA_PREFIX)/lib -L $(XILINX_XRT)/lib


LIBS := -lhdf5_cpp -lhdf5 -lxilinxopencl -lxrt_core -lbutler -lglog -ljson-c -lmkl_rt -lpthread
LIBS+=$(OPENCV_SYM)

#-luuid

###
# Enable MKL
###
ifneq (,$(findstring MKL,$(XBLASOPTS)))
LIBS += -lmkl_rt
endif

###
# Enable CBLAS
###
ifneq (,$(findstring CBLAS,$(XBLASOPTS)))
LIBS += -lopenblas
endif

###
# Enable ZMQ
###
ifneq (,$(findstring ZMQ,$(XBLASOPTS)))
LIBS += -lzmq
endif

#####################
#      Rules/Targets
#####################

SRC := \
        ./src/main.cc 

adas_detection: 
	@echo -e "\n[BUILD] Building example \n\n"
	$(CXX)  -std=c++11 ${SRC} $(INCLUDE) -lxfdnn $(LIB_PATHS) $(LIBS) -o ${BUILD}/adas_detection.exe
	
	@echo -e "\n[BUILD] Done\n\n"
LD_LIBRARY_PATH :=${TMP_LD_PATH}
clean :
	rm -rf build/*exe 
