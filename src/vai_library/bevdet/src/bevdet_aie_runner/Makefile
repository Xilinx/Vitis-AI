# Copyright 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
#

TARGET=libbevdet_aie_runner.so
INSTALL=/usr/lib/

CC=$(CROSS_COMPILE)g++

CFLAGS = -O3 -fPIC -std=c++17
LDFLAGS = -lstdc++ -lglog -lxir -lvart-runner -lvart-util -lxrt_coreutil 
INC = -I/usr/include/xrt 

RM=rm -rf $(1)
CP=cp $(1) $(2)

LIB_SRC = ./src/vart_bevdet_aie_runner.cpp
TEST_SRC = ./test/test_bevdet_aie_runner.cpp

all: lib test

lib:
	$(CC) -o $(TARGET) $(LIB_SRC) -shared $(CFLAGS) $(LDFLAGS) $(INC)
	$(call CP, $(TARGET), $(INSTALL))
test: lib
	$(CC) -o test_bevdet_aie_runner $(TEST_SRC) $(CFLAGS) $(LDFLAGS) $(INC)

clean:
	$(call RM,*.o)
	$(call RM,*.so)


