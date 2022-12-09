PROJECT   =   app.exe

CXX       :=   g++ -std=c++17

LDFLAGS   :=     -L/opt/xilinx/xrt/lib -L/usr/local/lib64  -L/usr/lib
LDFLAGS   +=     -luuid -lxilinxopencl -lxrt_core -lxrt_coreutil  -lpthread -lopencv_imgcodecs -lopencv_core -lopencv_highgui -lopencv_imgproc   -lpthread -lboost_system -lprotobuf -lglog -lxir -ltarget-factory -lvart-runner



CUR_DIR =   $(shell pwd)
BUILD   =   build
SRC     =   $(CUR_DIR)
SRC     +=   $(CUR_DIR)/tfssd
SRC     +=   $(CUR_DIR)/tfssd/anchor
SRC     +=   $(CUR_DIR)/model_config/src
SRC     +=   $(CUR_DIR)/include/object_detection/protos
SRC     +=   $(CUR_DIR)/model_config/include/vitis/ai/proto
VPATH   =   $(SRC)
C_DIR  :=   $(shell find $(SRC) -name *.c)
OBJ     =   $(patsubst %.c, %.o, $(notdir $(C_DIR)))
CC_DIR :=   $(shell find $(SRC) -name *.cc)
OBJ    +=   $(patsubst %.cc, %.o, $(notdir $(CC_DIR)))
CPP_DIR :=   $(shell find $(SRC) -name *.cpp)
OBJ    +=   $(patsubst %.cpp, %.o, $(notdir $(CPP_DIR)))
PB_DIR :=   $(shell find $(SRC) -name *.pb.cc)
OBJ    +=   $(patsubst %.pb.cc, %.pb.o, $(notdir $(PB_DIR)))

CFLAGS :=  -D ENABLE_INT   -O3
CFLAGS +=   -I$(CUR_DIR)/model_config/include -I$(CUR_DIR)/include -I/usr/include/google/protobuf  -I$(CUR_DIR)/include/object_detection/protos/ -I/usr/include -I/opt/xilinx/xrt/include

CFLAG += -DTGT_DEVICE=${TARGET}

.PHONY: all clean

all: $(BUILD) $(PROJECT)

$(PROJECT) : $(OBJ)
	$(CXX) $(CFLAGS) $(addprefix $(BUILD)/, $^) $(MODEL) -o $@ $(LDFLAGS)

%.o : %.cc
	$(CXX) -c $(CFLAGS) $< -o $(BUILD)/$@
%.o : %.cpp
	$(CXX) -c $(CFLAGS) $< -o $(BUILD)/$@

clean:
	$(RM) -rf $(BUILD)/*.o $(BUILD)
	$(RM) $(PROJECT)

$(BUILD) :	
	-mkdir -p build
	cd include; ./build.sh
	cd model_config/include/vitis/ai/proto; protoc  --cpp_out=./ --proto_path=./ dpu_model_param.proto

