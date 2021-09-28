BLAS_ddrWidth=16 
BLAS_XddrWidth=16 
BLAS_argInstrWidth=1 
BLAS_numKernels=1 

BLAS_dataType=float 
BLAS_gemmMBlocks=2
BLAS_gemmKBlocks=2 
BLAS_gemmNBlocks=2 

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
          
CXXFLAGS += -D BLAS_dataType=$(BLAS_dataType)
VPP_FLAGS += ${MACROS}

ifeq ($(TARGET),$(filter $(TARGET),hw_emu))
	CXXFLAGS += -lxrt_hwemu
else ifeq ($(TARGET),$(filter $(TARGET),sw_emu))
	CXXFLAGS += -lxrt_swemu
else
	CXXFLAGS += -lxrt_core
endif

CONFIG_INFO = $(shell echo ${MACROS} | sed 's/-D //g; s/ -Wno.*//')

dump_config: 
	@echo ${CONFIG_INFO}  | tr " " "\n" > ${BUILD_DIR}/config_info.dat
