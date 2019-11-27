
CROSS_COMPILE := aarch64-linux-gnu-
SYSROOT :=  /home/yijianlong/platforms/zcu104_20183_PFM_20191_XRT/sdk/sysroots/aarch64-xilinx-linux
#SYSROOT = /home/yijianlong/platforms/zcu104_xrt/sdk/sysroots/aarch64-xilinx-linux
CC       =  $(CROSS_COMPILE)gcc
CXX      =  $(CROSS_COMPILE)g++

CUR_DIR  =  $(shell pwd)
HEADER   =  $(CUR_DIR)/include
SRC      =  $(CUR_DIR)/src
BUILD    =  $(CUR_DIR)/build
VPATH    =  $(SRC)

CFLAGS   =  -I. -I${SRC} -I${HEADER} -fPIC -std=c++11 -O2
CFLAGS  += -I${SYSROOT}/usr/include -I${SYSROOT}/usr/local/include -I../../n2cube/include

LDFLAGS  =  -L../../lib -L${SYSROOT}/usr/lib -L${SYSROOT}/lib -L../../n2cube/build -L$(SYSROOT)/opt/xilinx/xrt/lib
LDFLAGS +=  -ln2cube -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_core -pthread -lxrt_core


CFLAGS  += -mcpu=cortex-a53 


OBJ     =  main.o 

all: $(BUILD) resnet50 
.PHONY : clean 
 
resnet50:  $(OBJ)
	$(CXX) --sysroot=${SYSROOT} $(addprefix $(BUILD)/, $^)  -o $@ ${LDFLAGS} 

%.o : %.cc
	$(CXX)  -c $(CFLAGS) $< -o $(BUILD)/$@ 

$(BUILD):
	mkdir -p $@

clean:
	$(RM) $(BUILD)/* resnet50
