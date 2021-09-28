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
BLAS_runGemm = 1
BLAS_streamingKernel=1

BLAS_FLOAT_WIDTH=8

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
          -D BLAS_runGemm=$(BLAS_runGemm) \
          -D BLAS_streamingKernel=$(BLAS_streamingKernel) \
          -D AP_INT_MAX_W=1026
          
CXXFLAGS += -D BLAS_streamingKernel=$(BLAS_streamingKernel) \
            -D BLAS_dataType=$(BLAS_dataType)
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
