PY_SCRIPT=${XFLIB_DIR}/L1/tests/sw/python/rtm3d/operation.py

RTM_dataType      = float

RTM_x  = 80
RTM_y  = 60
RTM_z  = 70
RTM_maxZ  = 280
RTM_maxY  = 180

RTM_MaxB = 20
RTM_NXB = $(RTM_MaxB)
RTM_NYB = $(RTM_MaxB)
RTM_NZB = $(RTM_MaxB)

RTM_order  = 8
RTM_numFSMs = 2
RTM_nPEZ = 2
RTM_nPEX = 4

RTM_time = 4
RTM_verify=1
RTM_device = 0

DATA_DIR = ./$(BUILD_DIR)/dataset_z${RTM_z}_y${RTM_y}_x${RTM_x}_t${RTM_time}/
HOST_ARGS = $(BINARY_CONTAINERS) ${DATA_DIR} $(RTM_z) $(RTM_y) $(RTM_x) $(RTM_time) ${RTM_verify} ${RTM_device}

data_gen:
	mkdir -p ${DATA_DIR} 
	python3 ${PY_SCRIPT} --func testForward --path ${DATA_DIR} --z ${RTM_z} --y ${RTM_y} --x ${RTM_x} --time ${RTM_time} --nxb ${RTM_NXB} --nyb ${RTM_NYB} --nzb ${RTM_NZB} --order ${RTM_order} --verify ${RTM_verify}

run_hw: data_gen
	$(EXE_FILE) $(HOST_ARGS)
