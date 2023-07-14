## Vitis-AI AKS ROOT folder
PWD := $(shell pwd)

ifeq "$(origin AKS_ROOT)" "undefined"
AKS_ROOT := ${PWD}/..
endif

ifeq "$(origin CONDA_PREFIX)" "undefined"
PREFIX := /usr
else
PREFIX := $(CONDA_PREFIX)
endif

###
# Includes
###
# Common
INCLUDE += -I $(PREFIX)/include
# AKS
INCLUDE += -I $(AKS_ROOT)/
INCLUDE += -I $(AKS_ROOT)/src

###
# Libraries
###
LIB_PATHS := -L $(PREFIX)/lib
# Common
LIBS := -lpthread 
# Boost
LIBS += -lboost_filesystem
# Others

###
# Compiler
###
CXX := g++ 
# Flags
CXXOPTS :=-O3
CCOPTS := -D __USE_XOPEN2K8 -std=c++11 -fPIC -static-libgcc -static-libstdc++

###
# Share library flags
###
LD_OPT := -shared -pthread -fPIC

###
# Rules/Targets
###
AKS_LIB := libAks.so
##
# Source Files
##
AKS_SRC := $(wildcard *.cpp)
AKS_SRC := $(filter-out AksPyExt.cpp, $(AKS_SRC))

##
# Objects
##
AKS_OBJS := $(patsubst %.cpp,objs/%.o,$(wildcard $(AKS_SRC)))

#Python Extension
PYEXT := $(shell python3-config --extension-suffix)
PYEXTNAME := _aks$(PYEXT)
PYBINDINC := $(shell python3 -m pybind11 --includes)
PYBINDLIB := $(PREFIX)/lib/libpython3.6m.so
PYBINDRT  := -rdynamic -Wl,-rpath,${PREFIX}/lib

###
# Make Targets
###
engine: $(AKS_LIB)
pyext: $(PYEXTNAME)
all: $(AKS_LIB) $(PYEXTNAME)
	
.PHONY := engine pyext

$(AKS_LIB): $(AKS_OBJS)
	@echo -e "make: $@"
	@mkdir -p libs
	@$(CXX) $(LD_OPT) $(PYBINDRT) -o libs/$@ $(AKS_OBJS) $(LIB_PATHS) $(LIBS) $(OPENCV_SYM) $(PYBINDLIB)
	@cp libs/$@ $(AKS_ROOT)/libs

objs/%.o: %.cpp
	@echo -e "make: $<"
	@mkdir -p objs
	@$(CXX) $(CXXOPTS) $(CCOPTS) -c -o $@ $(INCLUDE) $(PYBINDINC) $(LIB_PATHS) $(LIBS) $< 

$(PYEXTNAME): libs/$(AKS_LIB) AksPyExt.cpp
	@echo -e "make: $@"
	@$(CXX) -rdynamic $(CXXOPTS) $(CCOPTS) $(LD_OPT) $(INCLUDE) $(PYBINDINC) -L$(AKS_ROOT)/libs AksPyExt.cpp -o libs/$@ -lAks
	@cp libs/$@ $(AKS_ROOT)/libs

clean:
	@echo -e "make: src $@"
	@$(RM) -r libs

ultraclean:
	@echo -e "make: src $@"
	@$(RM) -r libs objs

install:
	@echo -e "make: $@"
	@cp libs/libAks.so /usr/lib
	@cp libs/_aks.cpython-36m-x86_64-linux-gnu.so /usr/lib/python${PY_VER}/_aks.cpython-36m-x86_64-linux-gnu.so
	@cp libs/aks.py /usr/lib/python${PY_VER}/aks.py
	@mkdir /usr/include/aks
	@cp aks/*.h /usr/include/aks
