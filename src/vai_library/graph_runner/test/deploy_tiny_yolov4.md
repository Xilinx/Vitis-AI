#
# Copyright 2022-2023 Advanced Micro Devices Inc.
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
# Analyze the xmodel

``` console
% which xdputil
% xdputil -h
% xdputil xmodel -s /workspace/aisw/debug_models/tensorflow-yolov4-tiny-master/tensorflow-yolov4-tiny-master/compile/yolov4-tiny.svg /workspace/aisw/debug_models/tensorflow-yolov4-tiny-master/tensorflow-yolov4-tiny-master/compile/yolov4-tiny.xmodel
% xdputil xmodel -t /workspace/aisw/debug_models/tensorflow-yolov4-tiny-master/tensorflow-yolov4-tiny-master/compile/yolov4-tiny.txt /workspace/aisw/debug_models/tensorflow-yolov4-tiny-master/tensorflow-yolov4-tiny-master/compile/yolov4-tiny.xmodel
% firefox file:///workspace/aisw/debug_models/tensorflow-yolov4-tiny-master/tensorflow-yolov4-tiny-master/compile/yolov4-tiny.svg
% head /workspace/aisw/debug_models/tensorflow-yolov4-tiny-master/tensorflow-yolov4-tiny-master/compile/yolov4-tiny.txt; echo ...; tail /workspace/aisw/debug_models/tensorflow-yolov4-tiny-master/tensorflow-yolov4-tiny-master/compile/yolov4-tiny.txt
% xdputil xmodel -l /workspace/aisw/debug_models/tensorflow-yolov4-tiny-master/tensorflow-yolov4-tiny-master/compile/yolov4-tiny.xmodel
```

We can see the basic structure of the xmoel.

# generate reference result

##  prepare reference input

``` console
% g++ -o ~/.local/bin/image_to_bin \
         /workspace/aisw/Vitis-AI-Library/usefultools/src/image_to_bin.cpp \
    -lglog -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui
% image_to_bin /workspace/aisw/debug_models/tensorflow-yolov4-tiny-master/tensorflow-yolov4-tiny-master/images/COCO_val2014_000000189868.jpg /workspace/aisw/debug_models/tensorflow-yolov4-tiny-master/tensorflow-yolov4-tiny-master/images/COCO_val2014_000000189868.bin 416 416
% md5sum /workspace/aisw/debug_models/tensorflow-yolov4-tiny-master/tensorflow-yolov4-tiny-master/images/COCO_val2014_000000189868.bin
7432192dbe8b0cacdf99c2112732324b  /workspace/aisw/debug_models/tensorflow-yolov4-tiny-master/tensorflow-yolov4-tiny-master/images/COCO_val2014_000000189868.bin
```

This is an optional step. To follow our own conventions, we usually copy it to the data store.

``` console
% sudo mkdir -p /scratch/models/cache/golden/74
% sudo chmod -R o+rwx /scratch/models/cache/golden
% cp -av /workspace/aisw/debug_models/tensorflow-yolov4-tiny-master/tensorflow-yolov4-tiny-master/images/COCO_val2014_000000189868.bin /scratch/models/cache/golden/74/32192dbe8b0cacdf99c2112732324b
```

## build `vaie`


``` console
% cd /workspace/aisw/
% git clone gits@xcdl190260:aisw/vaie
% cd vaie
% ./cmake.sh
% which vaie-run
```

## generate reference results

``` console
% vaie-run -i /workspace/aisw/debug_models/tensorflow-yolov4-tiny-master/tensorflow-yolov4-tiny-master/compile/./yolov4-tiny.xmodel --init 'inputs/aquant /scratch/models/cache/golden/74/32192dbe8b0cacdf99c2112732324b' --dump 'detector/yolo-v4-tiny/Conv_10/LeakyRelu/aquant /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_Conv_10_LeakyRelu_aquant.bin;detector/yolo-v4-tiny/Conv_17/BiasAdd/aquant /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_Conv_17_BiasAdd_aquant.bin;detector/yolo-v4-tiny/Conv_17/BiasAdd/aquant_ /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_Conv_17_BiasAdd_aquant_.bin;detector/yolo-v4-tiny/Conv_2/LeakyRelu/aquant /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_Conv_2_LeakyRelu_aquant.bin;detector/yolo-v4-tiny/Conv_20/BiasAdd/aquant /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_Conv_20_BiasAdd_aquant.bin;detector/yolo-v4-tiny/Conv_20/BiasAdd/aquant_ /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_Conv_20_BiasAdd_aquant_.bin;detector/yolo-v4-tiny/Conv_6/LeakyRelu/aquant /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_Conv_6_LeakyRelu_aquant.bin;detector/yolo-v4-tiny/ResizeNearestNeighbor/size_const /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_ResizeNearestNeighbor_size_const.bin;detector/yolo-v4-tiny/strided_slice/aquant /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_strided_slice_aquant.bin;detector/yolo-v4-tiny/strided_slice_1/aquant /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_strided_slice_1_aquant.bin;detector/yolo-v4-tiny/strided_slice_2/aquant /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_strided_slice_2_aquant.bin' --target ref --log-path /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0 --deploy release --disable-debug

% md5sum /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_Conv_10_LeakyRelu_aquant.bin \
         /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_Conv_17_BiasAdd_aquant.bin \
         /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_Conv_17_BiasAdd_aquant_.bin \
         /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_Conv_2_LeakyRelu_aquant.bin \
         /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_Conv_20_BiasAdd_aquant.bin \
         /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_Conv_20_BiasAdd_aquant_.bin \
         /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_Conv_6_LeakyRelu_aquant.bin \
         /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_ResizeNearestNeighbor_size_const.bin \
        /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_strided_slice_aquant.bin \
       /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_strided_slice_1_aquant.bin \
       /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_strided_slice_2_aquant.bin
```

## copy the files on the target board.

From the above command, we can see that the xmodel has fingerprint as
below, so that we have to deploy it on ZCU102 with B4096DPU.

```
        "fingerprint":"0x1000020f6014407",
        "DPU Arch":"DPUCZDX8G_ISA0_B4096_MAX_BG2",
```

### copy the libraries

if you have v1.3.1 installed then the libraries are already there. This is an optional step.

``` console
% rsync -avz /opt/petalinux/2020.2/sysroots/aarch64-xilinx-linux/install/Debug/* b0:/usr/
```

### copy the xmodel


``` console
% ssh b0 mkdir -p /workspace/
% rsync -v /workspace/aisw/debug_models/tensorflow-yolov4-tiny-master/tensorflow-yolov4-tiny-master/compile/yolov4-tiny.xmodel b0:/workspace/
```

copy the input

``` console
% ssh b0 mkdir -p /scratch/models/cache/golden/74
% rsync -v /scratch/models/cache/golden/74/32192dbe8b0cacdf99c2112732324b b0:/scratch/models/cache/golden/74/
```

copy the reference results

``` console
% rsync -v /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_Conv_10_LeakyRelu_aquant.bin \
         /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_Conv_17_BiasAdd_aquant.bin \
         /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_Conv_17_BiasAdd_aquant_.bin \
         /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_Conv_2_LeakyRelu_aquant.bin \
         /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_Conv_20_BiasAdd_aquant.bin \
         /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_Conv_20_BiasAdd_aquant_.bin \
         /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_Conv_6_LeakyRelu_aquant.bin \
         /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_ResizeNearestNeighbor_size_const.bin \
        /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_strided_slice_aquant.bin \
       /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_strided_slice_1_aquant.bin \
       /tmp/chunywan/vaie.log/yolov4-tiny-tensorflow/ref/batch_0/detector_yolo-v4-tiny_strided_slice_2_aquant.bin \
       b0:/workspace;
```
