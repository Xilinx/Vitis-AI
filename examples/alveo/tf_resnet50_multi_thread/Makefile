BUILD_DIR ?= ./build

CXX := g++ 

XILINX_XRT := /opt/xilinx/xrt

install_prefix_default=$(HOME)/.local/Ubuntu.18.04.x86_64

ifndef XILINX_VART
	# setting default path
	XILINX_VART=${install_prefix_default}.Release
endif

XILINX_XRT := /opt/xilinx/xrt
VITIS_INCLUDES := -I $(XILINX_XRT)/include -I$(CONDA_PREFIX)/include/xip/butler -I$(CONDA_PREFIX)/include -I$(XILINX_VART)/include -I/usr/include -I./include
VITIS_LIBS := -L$(BUILD_DIR) -L$(CONDA_PREFIX)/lib -L$(XILINX_XRT)/lib -L$(XILINX_VART)/lib -L/usr/lib

XILINX_XRM := /opt/xilinx/xrm
ifneq "$(wildcard $(XILINX_XRM))" ""
  # XRM installed on system
  ifeq (1, ${XRM})
    VITIS_INCLUDES += -I $(XILINX_XRM)/include -D XRM
  endif
endif

CPPFLAGS := $(VITIS_INCLUDES) $(VITIS_LIBS) -fpermissive -Wall -std=c++17 -O3

.PHONY: all
all: $(BUILD_DIR)/DPUCADF8H.exe

$(BUILD_DIR)/DPUCADF8H.exe: src/*.cpp 
	$(MKDIR_P) build
	$(CXX) $(CPPFLAGS) -o $@ src/*.cpp -lrt-engine -lpthread -lxilinxopencl -lxrt_core -ljson-c -lglog -lgflags -lxrt++ -lxrt_coreutil -lunilog -lprotobuf -lxir -ltarget-factory -lvart-runner -ljsoncpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -ltiff -lboost_system -lboost_filesystem

.PHONY: clean

clean:
	$(RM) $(BUILD_DIR)/DPUCADF8H.exe


MKDIR_P ?= mkdir -p
