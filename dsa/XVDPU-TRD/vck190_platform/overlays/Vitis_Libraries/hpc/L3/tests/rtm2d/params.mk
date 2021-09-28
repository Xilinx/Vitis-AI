PY_SCRIPT=${XFLIB_DIR}/L1/tests/sw/python/rtm2d/operation.py
RTM_width  = 128
RTM_depth  = 128
RTM_shots = 1
RTM_time = 10
RTM_verify=1
RTM_deviceId = 0

MACROS += -D RTM_dataType=float\
            -D RTM_numFSMs=10 \
            -D RTM_numBSMs=5 \
            -D RTM_maxDim=1280 \
            -D RTM_order=8 \
            -D RTM_MaxB=40 \
            -D RTM_nPE=2 \
            -D RTM_NXB=40 \
            -D RTM_NZB=40 \
            -D RTM_parEntries=8
			

ifeq ($(TARGET),$(filter $(TARGET),hw_emu))
	CXXFLAGS += -lxrt_hwemu
else ifeq ($(TARGET),$(filter $(TARGET),sw_emu))
	CXXFLAGS += -lxrt_swemu
else
	CXXFLAGS += -lxrt_core
endif

CXXFLAGS += ${MACROS}
VPP_FLAGS += ${MACROS}

DATA_DIR= ./$(BUILD_DIR)/data/
HOST_ARGS += $(RTM_depth) $(RTM_width) $(RTM_time) ${RTM_shots} ${RTM_verify} ${RTM_deviceId}

data_gen:
	mkdir -p ${DATA_DIR} 
	python3 ${PY_SCRIPT} --func testRTM --path ${DATA_DIR} --depth ${RTM_depth} --width ${RTM_width} --time ${RTM_time} --nxb 40 --nzb 40 --order 8 --verify ${RTM_verify} --shot ${RTM_shots} --type float
