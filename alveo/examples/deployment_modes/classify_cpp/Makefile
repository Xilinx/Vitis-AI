## Mlsuite ROOT folder
ROOT_PATH :=${PWD}/../../..
YOLO_CPP := ${ROOT_PATH}/apps/yolo
XDNN_CPP_PATH := ${ROOT_PATH}/vai/dpuv1/rt/xdnn_cpp
XDNN_CPP_LIB := ${ROOT_PATH}/vai/dpuv1/rt/xdnn_cpp/lib
NMS_PATH=${ROOT_PATH}/apps/yolo/nms
OPENCV_SYM=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

XBLASOPTS :=-O3 -D MKL -D ZMQ -D XBLAS
#XBLASOPTS :=-O3 -D CBLAS -D ZMQ -D XBLAS
#XBLASOPTS :=-g -O0 -D MKL  -D ZMQ -D XBLAS -D DEBUG

###
# Is XRT present
###
ifeq (,/opt/xilinx/xrt/include/version.h)
$(error XILINX_XRT is not present on system)
endif

#auto detect version of XRT_BUILD (2.3.x)
#ERROR if not 2019.2+
INFO:= $(shell cat /opt/xilinx/xrt/include/version.h | grep xrt_build_version\\[ | sed 's/[^0-9.]*//g')
MAJOR:= $(shell echo $(INFO) | cut -f1 -d ".")
MINOR:= $(shell echo $(INFO) | cut -f2 -d ".")
MAJOR_GT:= $(shell expr $(MAJOR) \> 2)
MAJOR_EQ:= $(shell expr $(MAJOR) = 2)
MINOR_:= $(shell expr $(MINOR) \>= 3)

# Validate 2019.2+ XRT
ifeq (1,$(MAJOR_EQ))
ifeq (0,$(MINOR_))
$(error Invalid XRT Version)
endif
else
ifeq (0,$(MAJOR_GT))
$(error Invalid XRT Version)
endif
endif

XILINX_XRT := /opt/xilinx/xrt

###
# Add XRT include
###
INCLUDE += -I$(XILINX_XRT)/include

###
# Add common include
###
INCLUDE += -I$(CONDA_PREFIX)/include -I$(CONDA_PREFIX)/include/xip/butler 

CXX := g++ #$(GXX)

## Runtime
CCOPTS := -D __USE_XOPEN2K8 $(XBLASOPTS) -std=c++11 -fPIC -static-libgcc -static-libstdc++

#####################
#        Linker
#####################
# common
LIB_PATHS := -L $(CONDA_PREFIX)/lib -L $(XILINX_XRT)/lib


# Runtime
LD_OPT := -shared -pthread -fPIC

ifneq (,$(findstring SANITIZE,$(XBLASOPTS)))
#CCOPTS += -fno-omit-frame-pointer -lubsan -lasan -fuse-ld=gold -fsanitize=address,undefined,bounds-strict  -fsanitize-recover=address
#CCOPTS += -fno-omit-frame-pointer -lubsan -lasan -fuse-ld=gold -fsanitize=address,undefined,bounds-strict -fsanitize-recover=address
#CXX := g++-6
CCOPTS += -fsanitize=address -fno-omit-frame-pointer
LD_OPT += -fsanitize=address
endif

XFDNN_OBJS_V3 := $(addprefix objs/v3/,$(addsuffix .o,$(basename $(XBLAS_SRC))))

LIBS := -lhdf5_cpp -lhdf5 -lxilinxopencl -lxrt_core -lbutler 
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

CLASSIFY_SRC := \
        classify.cpp \
	    interface.cpp

classify : 
	@echo -e "\n[BUILD] Building Classification example \n\n"
	g++ -g -std=c++11 ${CLASSIFY_SRC} $(INCLUDE) -I${XDNN_CPP_PATH} -L${XDNN_CPP_LIB} -lxfdnn -lboost_thread -lboost_system $(LIB_PATHS) $(LIBS) ${OPENCV_SYM} -o classify.exe
	
	@echo -e "\n[BUILD] Done\n\n"

clean :
	rm *.exe
