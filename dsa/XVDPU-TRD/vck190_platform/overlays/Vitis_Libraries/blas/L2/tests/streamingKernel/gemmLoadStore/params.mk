DATA_DIR     = $(BUILD_DIR)/data
APP_BIN      = $(DATA_DIR)/app.bin
APP_TXT      = $(DATA_DIR)/app.txt
APP_GOLD_BIN = $(DATA_DIR)/app_gold.bin
APP_GOLD_TXT = $(DATA_DIR)/app_gold.txt

GEN_BIN_PROGRAM=gemmLdSt 128 128 128 A B X

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

BLAS_FLOAT_WIDTH = 7
DEBUG=yes

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
          
CXXFLAGS += $(MACROS)
VPP_FLAGS += $(MACROS)

GEN_BIN_EXE_NAME = gen_bin
GEN_BIN_SRCS = gen_gemm.cpp
COMPILER_SRC_DIR = $(XF_PROJ_ROOT)/L2/src/streamingKernel/sw/compiler

GEN_BIN_OBJ_FILES = $(foreach s,$(GEN_BIN_SRCS),$(BUILD_DIR)/compiler/$(basename $(s)).o)
GEN_BIN_EXE_FILE ?= $(BUILD_DIR)/$(GEN_BIN_EXE_NAME)$(if $(EXE_EXT),.,)$(EXE_EXT)

$(BUILD_DIR)/compiler/%.o: $(COMPILER_SRC_DIR)/%.cpp
	@echo -e "----\nCompiling object $*..."
	mkdir -p $(BUILD_DIR)/compiler
	$(CXX) -o $@ -c $< $(CXXFLAGS)

$(GEN_BIN_EXE_FILE): $(GEN_BIN_OBJ_FILES)
	@echo -e "----\nCompiling gen_bin $(notdir $@)..."
	mkdir -p $(BUILD_DIR)/data
	$(CXX) -o $@ $^ $(CXXFLAGS) $(GEN_BIN_LDFLAGS)

gen_bin: $(GEN_BIN_EXE_FILE)
data_gen: gen_bin 
	$(GEN_BIN_EXE_FILE) -write $(APP_BIN) $(GEN_BIN_PROGRAM)
	$(GEN_BIN_EXE_FILE) -read $(APP_BIN) > $(APP_TXT)
	$(GEN_BIN_EXE_FILE) -read $(APP_GOLD_BIN) > $(APP_GOLD_TXT)

check:$(GEN_BIN_EXE_FILE)
	$(GEN_BIN_EXE_FILE) -read $(DATA_DIR)/app_out0.bin > $(DATA_DIR)/app_out0.txt
	cmp -i 8192 $(APP_GOLD_BIN) $(DATA_DIR)/app_out0.bin || $(GEN_BIN_EXE_FILE) -compare 1e-3 3e-6 $(APP_GOLD_BIN) $(DATA_DIR)/app_out0.bin
