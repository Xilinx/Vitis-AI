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

CXX_FLAGS += -I=/usr/include/xrt
CXX_FLAGS += -I/usr/include/xrt
CXX_FLAGS += -I=/install/Release/include
CXX_FLAGS += -DENABLE_NEON=1
#CXX_FLAGS += -DENABLE_AIE=1
#CXX_FLAGS += -DPSMNET_OLD
CXX_FLAGS += -std=c++17
CXX_FLAGS += -O3
CXX_FLAGS += -Wall
#CXX_FLAGS += -Werror
CXX_FLAGS += $(shell pkg-config opencv4 --cflags)
#CXX_FLAGS += -I/usr/include/opencv4
OUTPUT_DIR = ./build

LD_FLAGS += -L=/install/Release/lib
LD_FLAGS += -L/opt/petalinux/2021.1/sysroots/cortexa72-cortexa53-xilinx-linux/install/Release/lib/
LD_FLAGS +=-lglog
LD_FLAGS +=-lvitis_ai_library-runner_helper
LD_FLAGS +=-lprotobuf
LD_FLAGS +=-lvitis_ai_library-dpu_task
LD_FLAGS += $(shell pkg-config opencv4 --libs)
#LD_FLAGS += -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio -lopencv_video
LD_FLAGS += -lxrt_coreutil
LD_FLAGS += -lxrt_core
LD_FLAGS += -pthread

all: $(OUTPUT_DIR) demo_psmnet test_performance_psmnet

$(OUTPUT_DIR):
	mkdir -p $@

$(OUTPUT_DIR)/dpu_sfm.o : dpu_sfm.cpp my_xrt_bo.hpp vai_graph.hpp
	$(CXX)  $(CXX_FLAGS) -c -o $@  -I.  $<

$(OUTPUT_DIR)/cpu_op.o:  cpu_op.cpp
	$(CXX)  $(CXX_FLAGS) -c -o $@  -I.  $<

$(OUTPUT_DIR)/demo_psmnet.o : demo_psmnet.cpp
	$(CXX)  $(CXX_FLAGS) -c -o $@ -I.  $<

$(OUTPUT_DIR)/psmnet.o:  psmnet.cpp psmnet.hpp
	$(CXX)  $(CXX_FLAGS) -c -o $@  -I. $<

$(OUTPUT_DIR)/psmnet_aie.o:  psmnet.cpp psmnet.hpp
	$(CXX)  $(CXX_FLAGS) -DENABLE_AIE=1 -c -o $@  -I. $<

$(OUTPUT_DIR)/dpu_resize.o:  dpu_resize.cpp dpu_resize.hpp vai_aie_task_handler.hpp my_xrt_bo.hpp vai_graph.hpp
	$(CXX)  $(CXX_FLAGS) -c -o $@  -I.  $<

$(OUTPUT_DIR)/my_xrt_bo.o:  my_xrt_bo.cpp
	$(CXX)  $(CXX_FLAGS) -c -o $@  -I.  $<

demo_psmnet_aie: $(OUTPUT_DIR)/demo_psmnet.o $(OUTPUT_DIR)/dpu_sfm.o $(OUTPUT_DIR)/psmnet_aie.o $(OUTPUT_DIR)/dpu_resize.o $(OUTPUT_DIR)/my_xrt_bo.o 
	$(CXX) -std=c++17 $(CXX_FLAGS) -DENABLE_AIE=1 -I.  -o $@ $+ $(LD_FLAGS)

demo_psmnet: $(OUTPUT_DIR)/demo_psmnet.o $(OUTPUT_DIR)/psmnet.o $(OUTPUT_DIR)/cpu_op.o 
	$(CXX) -std=c++17 $(CXX_FLAGS) -I.  -o $@ $+ $(LD_FLAGS) -lvitis_ai_library-math

$(OUTPUT_DIR)/test_performance_psmnet.o: test_performance_psmnet.cpp  psmnet_benchmark.hpp
	$(CXX)  $(CXX_FLAGS) -c -o $@  -I. $<

test_performance_psmnet: $(OUTPUT_DIR)/test_performance_psmnet.o $(OUTPUT_DIR)/psmnet.o $(OUTPUT_DIR)/cpu_op.o
	$(CXX)  $(CXX_FLAGS) -std=c++17 -I. -o $@ $+ $(LD_FLAGS) -lvart-util -lvitis_ai_library-math

test_performance_psmnet_aie: $(OUTPUT_DIR)/test_performance_psmnet.o $(OUTPUT_DIR)/psmnet_aie.o $(OUTPUT_DIR)/dpu_sfm.o $(OUTPUT_DIR)/dpu_resize.o $(OUTPUT_DIR)/my_xrt_bo.o
	$(CXX)  $(CXX_FLAGS) -std=c++17 -I. -o $@ $+ $(LD_FLAGS) -lvart-util

clean:
	rm -rf $(OUTPUT_DIR)/*.o
	rm -rf demo_psmnet
	rm -rf test_performance_psmnet
