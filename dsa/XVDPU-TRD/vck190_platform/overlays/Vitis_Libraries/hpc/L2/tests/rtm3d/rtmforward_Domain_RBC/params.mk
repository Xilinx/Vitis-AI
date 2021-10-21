PY_SCRIPT=${XFLIB_DIR}/L1/tests/sw/python/rtm3d/operation.py

RTM_dataType = float

RTM_x = 80
RTM_y = 60
RTM_z = 90
RTM_maxZ = 280
RTM_maxY = 180

RTM_MaxB = 20
RTM_NXB = $(RTM_MaxB)
RTM_NYB = $(RTM_MaxB)
RTM_NZB = $(RTM_MaxB)

RTM_order = 8
RTM_numFSMs = 2
RTM_nPEZ = 2
RTM_nPEX = 4

RTM_time = 4
RTM_verify=1
RTM_device = 0

MACROS += -D RTM_dataType=$(RTM_dataType) \
		  -D RTM_numFSMs=$(RTM_numFSMs) \
		  -D RTM_maxY=$(RTM_maxY) \
		  -D RTM_maxZ=$(RTM_maxZ) \
		  -D RTM_order=$(RTM_order) \
		  -D RTM_MaxB=$(RTM_MaxB) \
		  -D RTM_nPEZ=$(RTM_nPEZ) \
		  -D RTM_nPEX=$(RTM_nPEX) \
		  -D RTM_NXB=$(RTM_NXB) \
		  -D RTM_NYB=$(RTM_NYB) \
		  -D RTM_NZB=$(RTM_NZB)

CXXFLAGS += ${MACROS}

rtmforward_VPP_FLAGS += ${MACROS}

DATA_DIR = ./$(BUILD_DIR)/dataset_z${RTM_z}_y${RTM_y}_x${RTM_x}_t${RTM_time}/
HOST_ARGS = $(BINARY_CONTAINERS) ${DATA_DIR} $(RTM_z) $(RTM_y) $(RTM_x) $(RTM_time) ${RTM_verify} ${RTM_device}

data_gen:
	mkdir -p ${DATA_DIR} 
	python3 ${PY_SCRIPT} --func testForward --path ${DATA_DIR} --z ${RTM_z} --y ${RTM_y} --x ${RTM_x} --time ${RTM_time} --nxb ${RTM_NXB} --nyb ${RTM_NYB} --nzb ${RTM_NZB} --order ${RTM_order} --verify ${RTM_verify} --rbc

run_hw: data_gen
	$(EXE_FILE) $(HOST_ARGS)

