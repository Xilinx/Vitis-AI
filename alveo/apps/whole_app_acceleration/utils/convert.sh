#!/usr/bin/env bash
# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Description:
# This script converts an input JPEG file
# to a PPM file, then converts the PPM
# file to a series of output JPEG files
# for various color formats and quality.

# Usage:
# For single mode run
# > ./convert.sh <dir> <jpg> <QP>
# For batch mode run
# > ls <dir> | xargs -n1 -i ./convert.sh <dir> {} <QP>
#!/bin/bash
INDIR=$1
INFILE=$2
QP=$3

mkdir -p input
mkdir -p ppm

OUT_DIR=$INDIR/converted_images
mkdir -p $OUT_DIR


echo "Converting ${INDIR}/${INFILE}"

# Convert Input JPEG to PPM
cp ${INDIR}/${INFILE} input/.
djpeg -outfile ppm/${INFILE}.ppm input/${INFILE}


# Convert PPM to Non-Interleaved JPEG Output
cjpeg -quality ${QP} -qtables ../utils/qtables.txt -dct float -scans ../utils/scans_non_interleaved.txt -sample 1x1,1x1,1x1 -outfile $OUT_DIR/yuv444_n_q${QP}_${INFILE} ppm/${INFILE}.ppm
rm -rf input
rm -rf ppm

