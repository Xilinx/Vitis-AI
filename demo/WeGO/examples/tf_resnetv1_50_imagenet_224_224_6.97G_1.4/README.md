# 1. Preparation

## 1.1 Setup the Running Environment
Please follow [Setup Environment](https://github.com/Xilinx/Vitis-AI/tree/master/demo/WeGO#environment-setup) section to setup the running environment for WeGO.

## 1.2 Download the Quantized Model
Run following script to download the ResNet50 quantized model from Xilinx Vitis-AI modelzoo:
```bash
$ bash code/download_model.sh
```
The downloaded quantized model will locate in ./quantized_model/quantize_eval_model.pb.

## 1.3 Prepare the Dataset

We will use the ImageNet-2012 validation dataset for our testing purpose. To prepare the dataset, please:

- Download the ImageNet-2012 dataset: [[download]( http://www.image-net.org/download-images)]

> Note: The ImageNet-2012 dataset is not public available, you need to register on the download site in order to get the link to download the validation dataset.

- Create a directory named `data` and copy the valiation dataset into it:

```bash
$ cd tf_resnetv1_50_imagenet_224_224_6.97G_1.4 && mkdir data
$ tar -zxf /path/to/download/imagenet_val.tar.gz -C ./data
```

The data directory structure should be as follow:

```bash
data
└── validation
```

- Run following script to preprocess the dataset:

```bash
$  cd code/gen_data && bash get_dataset.sh
```

The final data directory structure after preprocessing will be as follows:

```bash
data
└── Imagenet
    ├── val_dataset 
    └── val.txt   
```

# 2. Run ResNet50 Model

In this demo, two different running modes can be selected to enable accuracy and performance test purpose with different running options provided.

## 2.1 Accuracy Test

We use the whole ImageNet-2.0 validation dataset for accuracy test purpose, which means **50000** images will be loaded and feed into WeGO for inference:

```bash
$ bash run_eval.sh --accuracy
```

> Note: Loading 50000 images is quite time-consuming, so we will cache the images using numpy when loading the images at first time, thus can reduce loading time if you want to repeat the accuracy test multiple times later.

When test is finished without errors, the print accruacy result for ResNet50 should be:

```bash
============ Test Result =============
Total Images: 50000
Recall_1 = [0.74374]
Recall_5 = [0.91872]
```

## 2.2 Performance Test

Instead of using 50000 images as in accuracy test mode, only 2000+ images will be loaded in performance mode and WeGO will repeat running inference 20 times, expecting to reduce image loading time and get more accurate performance numbers:

```bash
$ bash run_eval.sh --perf
```

The FPS numbers will vary depending on the hardware testing environment and workloads in server. For instance, in a x86_64 server with 12 CPU cores(Intel(R) Xeon(R) Bronze 3104 CPU @ 1.70GHz), about 3900+ FPS can be achieved. 

# License

Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
