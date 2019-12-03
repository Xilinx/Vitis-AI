#!/bin/bash

. /wrk/xhdhdnobkup3/kvraju/MLsuite/overlaybins/setup.sh

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MLSUITE_ROOT}/xfdnn/rt/xdnn_cpp/lib:${CONDA_PREFIX}/lib

gdb --args ./a.out dpu_rundir video/adas.avi
#./a.out dpu_rundir video/adas.avi
