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

#!/bin/bash

if [ -d quantized_model ] && [ -f quantized_model/quantize_eval_model.pb ]
then
    echo "[INFO] Model existed in 'quantized_model/quantize_eval_model.pb', skip downloading...";
    exit 0;
fi

if [ -d quantized_model ]
then
    rm -r quantized_model
fi

MODEL_URL=https://www.xilinx.com/bin/public/openDownload?filename=tf_yolov3_voc_416_416_65.63G_1.4.zip
DEST_ZIP_FILE=/tmp/tf_yolov3_voc_416_416_65.63G_1.4.zip
UNZIP_FILE=/tmp/tf_yolov3_voc_416_416_65.63G_1.4

if [ -d $UNZIP_FILE ]
then
    rm -r $UNZIP_FILE
fi
wget $MODEL_URL -O $DEST_ZIP_FILE >> /dev/null   \
&& echo "[INFO] Unzip model files..."            \
&& unzip $DEST_ZIP_FILE -d /tmp >> /dev/null

mkdir quantized_model && cp $UNZIP_FILE/quantized/quantize_eval_model.pb quantized_model/

echo "[INFO] Download Yolov3-voc quantized model done, saved in 'quantized_model/quantize_eval_model.pb'."