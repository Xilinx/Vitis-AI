#!/usr/bin/env bash
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/usr/bin/env bash

. ${VAI_ALVEO_ROOT}/overlaybins/setup.sh

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib

./build/ssd_video_analysis.exe video/structure.mp4 model 
