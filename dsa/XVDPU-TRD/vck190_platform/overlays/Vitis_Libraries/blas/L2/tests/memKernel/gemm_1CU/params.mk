DATA_DIR= ./$(BUILD_DIR)/data
GEN_BIN_EXE = ${BUILD_DIR}/gen_bin.exe
API_GEMM_EXE = ${BUILD_DIR}/api_gemm.exe
APP_BIN      = ${DATA_DIR}/app.bin
APP_TXT      = ${DATA_DIR}/app.txt
APP_GOLD_BIN = ${DATA_DIR}/app_gold.bin
APP_GOLD_TXT = ${DATA_DIR}/app_gold.txt

GEN_BIN_PROGRAM=gemm 64 64 64 64 64 64 64 1 0 A B C X

BLAS_ddrWidth=16 
BLAS_XddrWidth=16 
BLAS_argInstrWidth=1 

BLAS_dataType=float 
BLAS_gemmMBlocks=4
BLAS_gemmKBlocks=4 
BLAS_gemmNBlocks=4 

BLAS_argPipeline        = 2
BLAS_instructionSizeBytes = 64
BLAS_numKernels         = 1

BLAS_dataEqIntType = float
BLAS_XdataType     = float 
BLAS_argInstrWidth =   1
BLAS_numInstr      =  64
TEST_MEMCPY        = 0
BLAS_CACHE         = 0
BLAS_XVEC          = 0

BLAS_deviceId = 0

MACROS += -D TEST_MEMCPY=$(TEST_MEMCPY) \
          -D BLAS_instructionSizeBytes=$(BLAS_instructionSizeBytes) \
          -D BLAS_dataType=$(BLAS_dataType) \
          -D BLAS_dataEqIntType=$(BLAS_dataEqIntType) \
          -D BLAS_ddrWidth=$(BLAS_ddrWidth) \
          -D BLAS_argInstrWidth=$(BLAS_argInstrWidth) \
          -D BLAS_numInstr=$(BLAS_numInstr) \
          -D BLAS_argPipeline=$(BLAS_argPipeline) \
          -D BLAS_runTransp=0 \
          -D BLAS_runGemv=0 \
          -D BLAS_runGemm=1 \
          -D BLAS_runFcn=0 \
          -D BLAS_numKernels=${BLAS_numKernels}\
          -D BLAS_CACHE=${BLAS_CACHE}\
          -D BLAS_XVEC=${BLAS_XVEC} \
          -D BLAS_gemmMBlocks=${BLAS_gemmMBlocks} \
          -D BLAS_gemmKBlocks=${BLAS_gemmKBlocks} \
          -D BLAS_gemmNBlocks=${BLAS_gemmNBlocks} \
          -D BLAS_XdataType=$(BLAS_XdataType) \
          -D BLAS_XddrWidth=$(BLAS_XddrWidth) \
		  -D AP_INT_MAX_W=1026
          
CXXFLAGS += ${MACROS}
VPP_FLAGS += ${MACROS}
#HOST_ARGS+= ${BLAS_deviceId}

CONFIG_INFO = $(shell echo ${MACROS} | sed 's/-D //g; s/ -Wno.*//')

${GEN_BIN_EXE} :$(XFLIB_DIR)/L2/src/memKernel/sw/gen_bin.cpp
	mkdir -p ${DATA_DIR} 
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

${API_GEMM_EXE} : $(XFLIB_DIR)/L2/src/memKernel/sw/api_gemm.cpp $(XFLIB_DIR)/L2/src/xcl2/xcl2.cpp
	mkdir -p ${BUILD_DIR} 
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

data_gen: ${GEN_BIN_EXE} ${API_GEMM_EXE} 
	${GEN_BIN_EXE} -write ${APP_BIN} ${GEN_BIN_PROGRAM}
	${GEN_BIN_EXE} -read ${APP_BIN} > ${APP_TXT}
	${GEN_BIN_EXE} -read ${APP_GOLD_BIN} > ${APP_GOLD_TXT}

run_gemm_api:$(API_GEMM_EXE) emconfig
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	$(CP) $(EMCONFIG_DIR)/emconfig.json .
	XCL_EMULATION_MODE=$(TARGET) $(API_GEMM_EXE) $(BUILD_DIR)/blas.xclbin 64 64 64 64 64 64 64 1 0
else
	$(API_GEMM_EXE) $(BUILD_DIR)/blas.xclbin 64 64 64 64 64 64 64 1 0 ${BLAS_deviceId}
endif
	
check: dump_config
	${GEN_BIN_EXE} -read ${DATA_DIR}/app_out0.bin  > ${DATA_DIR}/app_out0.txt
	cmp -i 8192 ${APP_GOLD_BIN} ${DATA_DIR}/app_out0.bin || ${GEN_BIN_EXE} -compare 1e-3 3e-6 ${APP_GOLD_BIN} ${DATA_DIR}/app_out0.bin
	
dump_config: 
	@echo ${CONFIG_INFO}  | tr " " "\n" > ${BUILD_DIR}/config_info.dat
