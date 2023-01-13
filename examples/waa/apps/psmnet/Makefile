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

CXX_FLAGS += -I=/usr/include/
CXX_FLAGS += -I=/usr/include/xrt
CXX_FLAGS += -I=/install/Release/include
CXX_FLAGS += -DENABLE_CV_ACCEL=1
#CXX_FLAGS += -DENABLE_NEON=1
#CXX_FLAGS += -DENABLE_AIE=1
CXX_FLAGS += -std=c++17
CXX_FLAGS += -O3
CXX_FLAGS += -Wall
#CXX_FLAGS += -Werror
#CXX_FLAGS += $(shell pkg-config opencv4 --cflags)
CXX_FLAGS += -I=/usr/include/opencv4
OUTPUT_DIR = ./

LD_FLAGS += -L=/install/Release/lib
LD_FLAGS += -L/opt/petalinux/2021.1/sysroots/cortexa72-cortexa53-xilinx-linux/install/Release/lib/
LD_FLAGS +=-lglog
LD_FLAGS +=-lvitis_ai_library-model_config
LD_FLAGS +=-lprotobuf
LD_FLAGS +=-lvitis_ai_library-dpu_task -lvitis_ai_library-math
#LD_FLAGS += $(shell pkg-config opencv4 --libs)
LD_FLAGS += -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio -lopencv_video
LD_FLAGS += -lxrt_coreutil
LD_FLAGS += -lxrt_core
LD_FLAGS += -pthread

all: $(OUTPUT_DIR)/demo_psmnet $(OUTPUT_DIR)/test_performance_psmnet

$(OUTPUT_DIR):
	mkdir -p $@

$(OUTPUT_DIR)/dpu_sfm.o : dpu_sfm.cpp my_xrt_bo.hpp vai_graph.hpp
	$(CXX)  $(CXX_FLAGS) -c -o $@  -I.  $<

# dpu_sfm.o.cpp : dpu_sfm.cpp vai_aie_task_handler.hpp
# 	$(CXX)  $(CXX_FLAGS) -E -o $@  -I.  $<

$(OUTPUT_DIR)/demo_psmnet.o : demo_psmnet.cpp cpu_op.hpp
	$(CXX)  $(CXX_FLAGS) -c -o $@ -I.  $<

$(OUTPUT_DIR)/psmnet.o:  psmnet.cpp psmnet.hpp cpu_op.hpp cost_volume.hpp
	$(CXX)  $(CXX_FLAGS) -c -o $@  -I. $<

$(OUTPUT_DIR)/dpu_resize.o:  dpu_resize.cpp dpu_resize.hpp vai_aie_task_handler.hpp my_xrt_bo.hpp vai_graph.hpp
	$(CXX)  $(CXX_FLAGS) -c -o $@  -I.  $<

$(OUTPUT_DIR)/cpu_op.o:  cpu_op.cpp
	$(CXX)  $(CXX_FLAGS) -c -o $@  -I.  $<

$(OUTPUT_DIR)/my_xrt_bo.o:  my_xrt_bo.cpp
	$(CXX)  $(CXX_FLAGS) -c -o $@  -I.  $<

$(OUTPUT_DIR)/cost_volume.o:  cost_volume.cpp
	$(CXX)  $(CXX_FLAGS) -c -o $@  -I.  $<


# /tmp/dpu_resize.o.cpp:  dpu_resize.cpp dpu_resize.hpp vai_aie_task_handler.hpp
# 	$(CXX)  $(CXX_FLAGS) -E -o $@  -I.  $<

#$(OUTPUT_DIR)/demo_psmnet: $(OUTPUT_DIR)/dpu_sfm.o $(OUTPUT_DIR)/demo_psmnet.o $(OUTPUT_DIR)/psmnet.o $(OUTPUT_DIR)/dpu_resize.o $(OUTPUT_DIR)/my_xrt_bo.o $(OUTPUT_DIR)/cpu_op.o
$(OUTPUT_DIR)/demo_psmnet: $(OUTPUT_DIR)/demo_psmnet.o $(OUTPUT_DIR)/psmnet.o $(OUTPUT_DIR)/cpu_op.o $(OUTPUT_DIR)/cost_volume.o
	$(CXX) -std=c++17 $(CXX_FLAGS) -I.  -o $@ $+ $(LD_FLAGS)

$(OUTPUT_DIR)/test.o : test.cpp dpu_sfm.hpp dpu_resize.hpp vai_aie_task_handler.hpp
	$(CXX)  $(CXX_FLAGS) -c -o $@ -I.  $<

test: $(OUTPUT_DIR)/dpu_sfm.o $(OUTPUT_DIR)/test.o $(OUTPUT_DIR)/dpu_resize.o $(OUTPUT_DIR)/my_xrt_bo.o
	$(CXX) -std=c++17 $(CXX_FLAGS) -I.  -o $@ $+ $(LD_FLAGS)

deploy: $(OUTPUT_DIR)/demo_psmnet
	ssh b0 mkdir -p /home/wcy && scp $< b0:/home/wcy

run: deploy
	ssh -X b0 "cd ~/psmnet; env DEBUG_BUFFER_OBJECT=0 DEBUG_BUFFER_ALLOCATOR=0 DEBUG_COST_VOLUME=1 DEBUG_DPU_RUNNER=0 LD_LIBRARY_PATH=/home/wcy/lib DEEPHI_PROFILING=1 /home/wcy/demo_psmnet images/fadnet/left/0015.png images/fadnet/right/0015.png images/fadnet/left/0011.png images/fadnet/right/0011.png images/fadnet/left/0006.png images/fadnet/right/0006.png"

check: run
	scp b0:psmnet/*.jpg /workspace/aisw/tmp/


test_dpu_task: test_dpu_task.cpp
	$(CXX) $(CXX_FLAGS) -std=c++17 -I.  -o $@ $+ -lglog -lvitis_ai_library-model_config -lprotobuf -lvitis_ai_library-dpu_task ${NEON_FLAGS} ${OPENCV_FLAGS} -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -I/usr/include/xrt -lxrt_coreutil -lxrt_core

run_test_dpu_task: test_dpu_task
	scp test_dpu_task b0:/home/wcy && ssh b0 /home/wcy/test_dpu_task

sync_debug_lib:
	rsync -avz /opt/petalinux/2020.2/sysroots/aarch64-xilinx-linux/install/Debug/lib b0:max/psmnet/

sync_release_lib:
	rsync -avz /opt/petalinux/2020.2/sysroots/aarch64-xilinx-linux/install/Release/lib b0:/home/wcy


$(OUTPUT_DIR)/test_performance_psmnet.o: test_performance_psmnet.cpp  psmnet_benchmark.hpp
	$(CXX)  $(CXX_FLAGS) -c -o $@  -I. $<

#$(OUTPUT_DIR)/test_performance_psmnet: $(OUTPUT_DIR)/test_performance_psmnet.o $(OUTPUT_DIR)/dpu_sfm.o $(OUTPUT_DIR)/psmnet.o $(OUTPUT_DIR)/dpu_resize.o $(OUTPUT_DIR)/my_xrt_bo.o $(OUTPUT_DIR)/cpu_op.o
$(OUTPUT_DIR)/test_performance_psmnet: $(OUTPUT_DIR)/test_performance_psmnet.o $(OUTPUT_DIR)/psmnet.o $(OUTPUT_DIR)/cpu_op.o $(OUTPUT_DIR)/cost_volume.o
	$(CXX)  $(CXX_FLAGS) -std=c++17 -I. -o $@ $+ $(LD_FLAGS) -lvart-util

deploy_test_performance_psmnet: $(OUTPUT_DIR)/test_performance_psmnet
	scp $< b0:/home/wcy

run_test_performance_psmnet: deploy_test_performance_psmnet
	ssh b0 "cd ~/psmnet; env LD_LIBRARY_PATH=/home/wcy/lib /home/wcy/test_performance_psmnet -t 1 left.list"


$(OUTPUT_DIR)/test_xrt_bo_alloc: test_xrt_bo_alloc.cpp
	$(CXX)  $(CXX_FLAGS) -std=c++17 -I. -o $@ $+ -lglog -lvitis_ai_library-model_config -lprotobuf -lvitis_ai_library-dpu_task ${NEON_FLAGS} ${OPENCV_FLAGS} -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -I/usr/include/xrt -lpthread -lvart-util -lxrt_core -lxrt_coreutil

run_test_xrt_bo_alloc: $(OUTPUT_DIR)/test_xrt_bo_alloc
	scp $< b0:/home/wcy/ && ssh b0 /home/wcy/test_xrt_bo_alloc /media/sd-mmcblk0p1/dpu.xclbin

$(OUTPUT_DIR)/test_resize_xrt: test_resize_xrt.cpp $(OUTPUT_DIR)/dpu_resize.o my_xrt_bo.o
	$(CXX)  $(CXX_FLAGS) -std=c++17 -I. -o $@ $+ -lglog -lvitis_ai_library-model_config -lprotobuf -lvitis_ai_library-dpu_task ${NEON_FLAGS} ${OPENCV_FLAGS} -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -I/usr/include/xrt -lpthread -lvart-util -lxrt_core -lxrt_coreutil 

#.PHONY: test_resize_xrt
#test_resize_xrt: $(OUTPUT_DIR)/test_resize_xrt
#	ssh	b0 "cd /home/root/psmnet;env USE_REAL_SIZE=1 DEEPHI_PROFILING=1 DEBUG_DPU_RUNNER=0 /home/wcy/test_resize_xrt;env DEEPHI_PROFILING=1 DEBUG_DPU_RUNNER=0 USE_REAL_SIZE=0 /home/wcy/test_resize_xrt"

test_cost_volume.o : test_cost_volume.cpp
	$(CXX)  $(CXX_FLAGS) -c -o $@  -I. $<

test_cost_volume: test_cost_volume.o
	$(CXX)  $(CXX_FLAGS) -std=c++17 -I. -o $@ $+ -lvart-util -lglog -lxrt_core -lvitis_ai_library-dpu_task

clean:
	rm -rf *.o
	rm -rf demo_psmnet
	rm -rf test_performance_psmnet
