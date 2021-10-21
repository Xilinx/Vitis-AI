DATA_DIR= ./$(BUILD_DIR)/data
GEN_BIN_EXE = ${BUILD_DIR}/gen_bin.exe
API_GEMM_EXE = ${BUILD_DIR}/api_gemm.exe
APP_BIN      = ${DATA_DIR}/app.bin
APP_TXT      = ${DATA_DIR}/app.txt
APP_GOLD_BIN = ${DATA_DIR}/app_gold.bin
APP_GOLD_TXT = ${DATA_DIR}/app_gold.txt

GEN_BIN_PROGRAM=gemm 64 64 64 A B X C

BLAS_pageSizeBytes=4096
BLAS_instrOffsetBytes=0
BLAS_resOffsetBytes=4096
BLAS_dataOffsetBytes=8192
BLAS_ddrMemBits=512
BLAS_maxNumInstrs=64
BLAS_memWordsPerInstr=1 
BLAS_parEntries=16 

BLAS_dataType=float
BLAS_mParWords=4
BLAS_kParWords=4
BLAS_nParWords=4
BLAS_numKernels = 1

BLAS_FLOAT_WIDTH=8
BLAS_deviceId = 0
DEBUG=0


MACROS += -D BLAS_pageSizeBytes=$(BLAS_pageSizeBytes) \
		  -D BLAS_instrOffsetBytes=$(BLAS_instrOffsetBytes) \
		  -D BLAS_resOffsetBytes=$(BLAS_resOffsetBytes) \
	      -D BLAS_dataOffsetBytes=$(BLAS_dataOffsetBytes) \
		  -D BLAS_ddrMemBits=$(BLAS_ddrMemBits) \
	      -D BLAS_maxNumInstrs=$(BLAS_maxNumInstrs) \
		  -D BLAS_memWordsPerInstr=$(BLAS_memWordsPerInstr) \
		  -D BLAS_parEntries=$(BLAS_parEntries) \
		  -D BLAS_dataType=$(BLAS_dataType) \
	      -D BLAS_mParWords=$(BLAS_mParWords) \
		  -D BLAS_kParWords=$(BLAS_kParWords) \
	 	  -D BLAS_nParWords=$(BLAS_nParWords) \
          -D BLAS_numKernels=$(BLAS_numKernels) \
		  -D BLAS_FLOAT_WIDTH=$(BLAS_FLOAT_WIDTH) \
		  -D DEBUG=$(DEBUG) \
		  -D AP_INT_MAX_W=1026
          
CXXFLAGS += ${MACROS}
VPP_FLAGS += ${MACROS}
HOST_ARGS += ${BLAS_deviceId}

CONFIG_INFO = $(shell echo ${MACROS} | sed 's/-D //g; s/ -Wno.*//')

${GEN_BIN_EXE} :$(XFLIB_DIR)/L2/src/streamingKernel/sw/compiler/gen_gemm.cpp
	mkdir -p ${DATA_DIR} 
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

${API_GEMM_EXE} : $(XFLIB_DIR)/L2/src/sw/api_gemm.cpp $(XFLIB_DIR)/L2/src/sw/xcl2/xcl2.cpp
	mkdir -p ${BUILD_DIR} 
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

data_gen: ${GEN_BIN_EXE}
	${GEN_BIN_EXE} -write ${APP_BIN} ${GEN_BIN_PROGRAM}
	${GEN_BIN_EXE} -read ${APP_BIN} > ${APP_TXT}
	${GEN_BIN_EXE} -read ${APP_GOLD_BIN} > ${APP_GOLD_TXT}

run_gemm_api:$(API_GEMM_EXE)
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	$(CP) $(EMCONFIG_DIR)/emconfig.json .
	XCL_EMULATION_MODE=$(TARGET) $(API_GEMM_EXE) $(BINARY_CONTAINERS) 64 64 64 64 64 64 64 1 0
else
	$(API_GEMM_EXE) $(BINARY_CONTAINERS) 64 64 64 64 64 64 64 1 0
endif
	
	
dump_config: 
	@echo ${CONFIG_INFO}  | tr " " "\n" > ${BUILD_DIR}/config_info.dat
check: dump_config
	${GEN_BIN_EXE} -read ${DATA_DIR}/app_out0.bin  > ${DATA_DIR}/app_out0.txt
	cmp -i 8192 ${APP_GOLD_BIN} ${DATA_DIR}/app_out0.bin || ${GEN_BIN_EXE} -compare 1e-3 3e-6 ${APP_GOLD_BIN} ${DATA_DIR}/app_out0.bin
