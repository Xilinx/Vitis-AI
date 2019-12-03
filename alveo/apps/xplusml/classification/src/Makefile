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



# Source Code
COMMON_REPO := .
# Check for XRT
ifeq (,/opt/xilinx/xrt/include/version.h)
$(error XILINX_XRT is not present on system)
endif

XILINX_XRT := /opt/xilinx/xrt/

#Output DIR
OUT_DIR= ../../libs/classification

# ----------------------------------------------------------
#  Tool Flags
# ----------------------------------------------------------

   CXXFLAGS += -I$(CONDA_PREFIX)/include -O3 -Wno-unused -Wno-maybe-uninitialized -Wno-unknown-pragmas -w

  
# ----------------------------------------------------------
# OpenCL, OpenCV
# ----------------------------------------------------------
  opencv_LDFLAGS += -L$(CONDA_PREFIX)/lib
  opencv_LDFLAGS += -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
  opencl_LDFLAGS += -lOpenCL -lpthread


# ----------------------------------------------------------
# Host Application(s)
# ----------------------------------------------------------
# xcl
xcl2_SRCS := $(COMMON_REPO)/xcl2.cpp
xcl2_HDRS := $(COMMON_REPO)/xcl2.hpp

# Application1:
APP_NAME := pp_pipeline
$(APP_NAME)_SRCS := $(COMMON_REPO)/xf_$(APP_NAME)_tb.cpp $(xcl2_SRCS)
$(APP_NAME)_HDRS := $(xcl2_HDRS)
$(APP_NAME)_CXXFLAGS := 
#-I. -I$(XLNX_SRC_PATH)/include -I${COMMON_REPO}
$(APP_NAME)_LDFLAGS  := $(opencv_LDFLAGS) $(opencl_LDFLAGS) -Wall -lrt


$(APP_NAME)_CXXFLAGS += -I$(XILINX_XRT)/include


EXES += $(APP_NAME)



# ----------------------------------------------------------
#  Actual Body
# ----------------------------------------------------------


define mk_so
$(1): $($(1)_SRCS) $($(1)_HDRS)
	@echo "Creating \"$$@.so\""
	mkdir -p $(OUT_DIR)
	$(CXX) $(CXXFLAGS) $($(1)_CXXFLAGS) $($(1)_SRCS) $($(1)_LDFLAGS) $(LDFLAGS) -fPIC -shared -DHLS_NO_XIL_FPO_LIB -o $$@.so
	cp $$@.so $(OUT_DIR)/.
EXE_GOALS+= $(1)
	
endef

$(foreach exe,$(EXES),$(eval $(call mk_so,$(exe))))


.PHONY: all
all: $(EXE_GOALS)

.PHONY: so
so: $(EXE_GOALS)


.PHONY: clean
clean:
	rm -rf *.so

