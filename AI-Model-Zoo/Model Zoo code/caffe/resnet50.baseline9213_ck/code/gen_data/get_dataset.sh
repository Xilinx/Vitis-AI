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

# PART OF THIS FILE AT ALL TIMES.

# (c) Copyright 2019 Xilinx, Inc. All rights reserved.
#
# This file contains confidential and proprietary information
# of Xilinx, Inc. and is protected under U.S. and
# international copyright and other intellectual property
# laws.
#
# DISCLAIMER
# This disclaimer is not a license and does not grant any
# rights to the materials distributed herewith. Except as
# otherwise provided in a valid license issued to you by
# Xilinx, and to the maximum extent permitted by applicable
# law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
# WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
# AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
# BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
# INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
# (2) Xilinx shall not be liable (whether in contract or tort,
# including negligence, or under any other theory of
# liability) for any loss or damage of any kind or nature
# related to, arising under or in connection with these
# materials, including for any direct, or any indirect,
# special, incidental, or consequential loss or damage
# (including loss of data, profits, goodwill, or any type of
# loss or damage suffered as a result of any action brought
# by a third party) even if such damage or loss was
# reasonably foreseeable or Xilinx had been advised of the
# possibility of the same.
#
# CRITICAL APPLICATIONS
# Xilinx products are not designed or intended to be fail-
# safe, or for use in any application requiring fail-safe
# performance, such as life-support or safety devices or
# systems, Class III medical devices, nuclear facilities,
# applications related to the deployment of airbags, or any
# other applications that could lead to death, personal
# injury, or severe property or environmental damage
# (individually and collectively, "Critical
# Applications"). Customer assumes the sole risk and
# liability of any use of Xilinx products in Critical
# Applications, subject only to applicable laws and
# regulations governing limitations on product liability.
#
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
# PART OF THIS FILE AT ALL TIMES.

#bash
GEN_DATA="../../data/"
TRAIN_DATA_DIR="Imagenet/train_resize_256"
TRAIN_ANNO_FILE="Imagenet/train.txt" 
VAL_DATA_DIR="Imagenet/val_resize_256"
VAL_ANNO_FILE="Imagenet/val.txt" 
SUB_DIR="Imagenet"
WORK_DIR=$(pwd)"/"
TRAIN_ORI_DATA="train"
VAL_ORI_DATA="validation"

echo $WORK_DIR
if [ -f $GEN_DATA$TRAIN_ANNO_FILE ] && [ -d $GEN_DATA$TRAIN_DATA_DIR ] && [ -f $GEN_DATA$VAL_ANNO_FILE ] && [ -d $GEN_DATA$VAL_DATA_DIR ] 
then

  echo "data link is vaild!"

else
  mkdir $GEN_DATA
  cd $GEN_DATA
  
  rm -r $TRAIN_DATA_DIR
  rm -r $VAL_DATA_DIR
  rm $TRAIN_ANNO_FILE
  rm $VAL_ANNO_FILE
    
  echo "Downloading..."

  #wget -c http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz

  #echo "Unzipping..."

  #tar -xf caffe_ilsvrc12.tar.gz && rm -f caffe_ilsvrc12.tar.gz

  echo "Done."
  
  cd $WORK_DIR

  CUR_GEN_DIR=$GEN_DATA$SUB_DIR
  mkdir $CUR_GEN_DIR

  python gen_data.py --data-dir $GEN_DATA$TRAIN_ORI_DATA --output-dir $GEN_DATA$TRAIN_DATA_DIR --short-size 256 --anno-file $GEN_DATA$TRAIN_ANNO_FILE
 

  python gen_data.py --data-dir $GEN_DATA$VAL_ORI_DATA --output-dir $GEN_DATA$VAL_DATA_DIR --short-size 256 --anno-file $GEN_DATA$VAL_ANNO_FILE

fi

