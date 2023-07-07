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
CXX_FLAGS += -O3
CXX_FLAGS += -Wall
CXX_FLAGS += -DENABLE_NEON
#CXX_FLAGS += -Werror
CXX_FLAGS += $(shell pkg-config opencv4 --cflags)
OUTPUT_DIR = ./build

LD_FLAGS += -L=/install/Release/lib
LD_FLAGS += -L/opt/petalinux/2021.1/sysroots/cortexa72-cortexa53-xilinx-linux/install/Release/lib/
LD_FLAGS +=-lglog
LD_FLAGS +=-lvitis_ai_library-model_config
LD_FLAGS +=-lprotobuf
LD_FLAGS +=-lvitis_ai_library-dpu_task
LD_FLAGS += $(shell pkg-config opencv4 --libs)
LD_FLAGS += -lxrt_coreutil
LD_FLAGS += -lxrt_core
LD_FLAGS += -pthread

all: $(OUTPUT_DIR) demo_fadnet test_performance_fadnet demo_fadnet_v2 test_performance_fadnet_v2

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

$(OUTPUT_DIR)/demo_fadnet.o : demo_fadnet.cpp 
	$(CXX)  $(CXX_FLAGS) -c -o $@ -I.  $<

$(OUTPUT_DIR)/demo_fadnet_v2.o : demo_fadnet_v2.cpp 
	$(CXX)  $(CXX_FLAGS) -c -o $@ -I.  $<

$(OUTPUT_DIR)/fadnet.o:  fadnet.cpp fadnet.hpp
	$(CXX)  $(CXX_FLAGS) -c -o $@  -I. $<

demo_fadnet: $(OUTPUT_DIR)/demo_fadnet.o $(OUTPUT_DIR)/fadnet.o
	$(CXX) -std=c++17 $(CXX_FLAGS) -I.  -o $@ $+ $(LD_FLAGS)

$(OUTPUT_DIR)/test_performance_fadnet.o: test_performance_fadnet.cpp
	$(CXX)  $(CXX_FLAGS) -c -o $@  -I. $<

test_performance_fadnet: $(OUTPUT_DIR)/test_performance_fadnet.o $(OUTPUT_DIR)/fadnet.o
	$(CXX)  $(CXX_FLAGS) -std=c++17 -I. -o $@ $+ $(LD_FLAGS) -lvart-util

$(OUTPUT_DIR)/fadnet_v2.o:  fadnet_v2.cpp fadnet_v2.hpp
	$(CXX)  $(CXX_FLAGS) -c -o $@  -I. $<

demo_fadnet_v2: $(OUTPUT_DIR)/demo_fadnet_v2.o $(OUTPUT_DIR)/fadnet_v2.o
	$(CXX) -std=c++17 $(CXX_FLAGS) -I.  -o $@ $+ $(LD_FLAGS)

$(OUTPUT_DIR)/test_performance_fadnet_v2.o: test_performance_fadnet_v2.cpp
	$(CXX)  $(CXX_FLAGS) -c -o $@  -I. $<

test_performance_fadnet_v2: $(OUTPUT_DIR)/test_performance_fadnet_v2.o $(OUTPUT_DIR)/fadnet_v2.o
	$(CXX)  $(CXX_FLAGS) -std=c++17 -I. -o $@ $+ $(LD_FLAGS) -lvart-util
 
clean:
	rm -rf $(OUTPUT_DIR)/*
	rm -rf demo_fadnet demo_fadnet_v2
	rm -rf test_performance_fadnet test_performance_fadnet_v2
