#
# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

CXX_FLAGS += -I=/install/Release/include
CXX_FLAGS += -I=/usr/include/vitis/ai
CXX_FLAGS += -std=c++17
#CXX_FLAGS += -g
CXX_FLAGS += -O3
CXX_FLAGS += -Wall
#CXX_FLAGS += -Werror
CXX_FLAGS += $(shell pkg-config opencv4 --cflags)
OUTPUT_DIR = ./build

LD_FLAGS += -L=/install/Release/lib
LD_FLAGS += -L/opt/petalinux/2021.1/sysroots/cortexa72-cortexa53-xilinx-linux/install/Release/lib/
LD_FLAGS += -lglog
LD_FLAGS += -lprotobuf
LD_FLAGS += -lvitis_ai_library-math
LD_FLAGS += -lvitis_ai_library-dpu_task
LD_FLAGS += $(shell pkg-config opencv4 --libs)
LD_FLAGS += -pthread

all: $(OUTPUT_DIR) demo_hfnet test_performance_hfnet

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

$(OUTPUT_DIR)/demo_hfnet.o : demo_hfnet.cpp 
	$(CXX)  $(CXX_FLAGS) -c -o $@ -I.  $<

$(OUTPUT_DIR)/hfnet.o:  hfnet.cpp hfnet.hpp
	$(CXX)  $(CXX_FLAGS) -c -o $@  -I. $<

demo_hfnet: $(OUTPUT_DIR)/demo_hfnet.o $(OUTPUT_DIR)/hfnet.o
	$(CXX) -std=c++17 $(CXX_FLAGS) -I.  -o $@ $+ $(LD_FLAGS)

$(OUTPUT_DIR)/test_performance_hfnet.o: test_performance_hfnet.cpp
	$(CXX)  $(CXX_FLAGS) -c -o $@  -I. $<

test_performance_hfnet: $(OUTPUT_DIR)/test_performance_hfnet.o $(OUTPUT_DIR)/hfnet.o
	$(CXX)  $(CXX_FLAGS) -std=c++17 -I. -o $@ $+ $(LD_FLAGS) -lvart-util

clean:
	rm -rf $(OUTPUT_DIR)/*
	rm -rf demo_hfnet
	rm -rf test_performance_hfnet
