# 1. Preparation

## 1.1 Setup the Running Environment
Please follow [Setup Environment](https://github.com/Xilinx/Vitis-AI/tree/master/demo/WeGO#environment-setup) section to setup the running environment for WeGO.

## 1.2 Download the Quantized Model
Run following script to download the Yolov3-Voc quantized model from Xilinx Vitis-AI modelzoo:
```bash
$ bash code/download_model.sh
```
The downloaded quantized model will locate in ./quantized_model/quantize_eval_model.pb.

## 1.3 Prepare the Dataset

We use the VOC2007_test dataset for testing purpose. Please run following to download and prepare the dataset automatically:

```bash
$ mkdir data && bash code/dataset_tools/prepare_data.sh
```

> Note: if you have downloaded the VOC2007 test set before, you could place them in the data directory manually and choose to skip downloading the dataset when the script asking for a choice.

If everythink is OK, a directory named `voc2007_test` will locate in the `data` directory:

```
data/
├── voc2007_test
│   ├── gt_detection.txt
│   ├── images
│   └── test.txt
├── VOCdevkit
│   └── VOC2007
└── VOCtest_06-Nov-2007.tar
```



# 2. Run Yolov3-Voc Model

In this demo, two different running modes can be selected to enable accuracy and performance test purpose with different running options provided.

## 2.1 Accuracy Test

We use the whole voc2007_test validation dataset for accuracy test purpose, which means **4952** images will be loaded and feed into WeGO for inference:

```bash
$ bash run_eval.sh --accuracy
```

When test is finished without errors, the print accruacy result for Yolov3-Voc should be:

```bash
evaluate 4952 images
dog AP: 0.8360481048959715
person AP: 0.7738129888173254
train AP: 0.8471861767268879
sofa AP: 0.7794771031197054
chair AP: 0.5981153274021767
car AP: 0.8508621068564768
pottedplant AP: 0.478538735165974
diningtable AP: 0.7065950422050893
horse AP: 0.8263323408364206
cat AP: 0.8698703218311403
cow AP: 0.7950756508657904
bus AP: 0.8592832334194384
bicycle AP: 0.8606305376141549
motorbike AP: 0.8383807491087123
bird AP: 0.7515381978395669
tvmonitor AP: 0.7355799326365119
sheep AP: 0.7720042045771628
aeroplane AP: 0.7749846182362002
boat AP: 0.7163596387662483
bottle AP: 0.6143171560872035
mAP: 0.7642496083504077
```

## 2.2 Performance Test

Instead of using 4952 images as in accuracy test mode, only 960 images will be loaded in performance mode and WeGO will repeat running inference 20 times, expecting to reduce image loading time and get more accurate performance numbers:

```bash
$ bash run_eval.sh --perf
```

The FPS numbers will vary depending on the hardware testing environment and workloads in server. For instance, in a x86_64 server with 12 CPU cores(Intel(R) Xeon(R) Bronze 3104 CPU @ 1.70GHz), about 430+ FPS can be achieved. 


# License

Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
