## Running TVM compiled models on Alveo-U200/Alveo-U250 using AKS

### Table of Contents
- [Introduction](#Introduction)
- [Run Examples on Alveo-U200/Alveo-U250](#run-examples-on-alveo-u200alveo-u250)
- [Run examples on Edge Devices](#run-examples-on-Edge-Devices
)


## Introduction
These examples demonstrates the usage of running TVM compiled graphs using [AKS](../../../../tools/AKS/README.md#introduction). TVM compiled graph is executed with [TVMkernel](kernel_zoo/kernel_tvm.json). This kernel can be used in other graphs too. Below is the list of the sample graphs provided as part of examples 

#### `graph_resnet50_dpucadf8h_tvm.json` contains Tensorflow Resnet50 graph. Below are the kernels used in the graph
* **ClassificationImreadResizeCCrop**: Resize the image to 256x256, centre crops 224x224 and  perfoms mean subtraction. 
* **TvmKernel**: Runs Tensorflow Resnet50 model through tvm runtime with DPUCADF8H and CPU.
* **ClassificationAccuracy**: Measures & reports accuracy of a model (Top-1/Top-5)

#### `graph_yolov3_dpucadf8h_tvm.json` contains darknet yolov3 graph. Below are the kernels used in the graph
* **DetectionImreadPreProcess**: Reads and Preprocesses an image for YOLOV3 network
* **TvmKernel**: Runs Darknet Yolov3 model through tvm runtime with DPUCADF8H and CPU.
* **YoloPostProcessTVM**: Postprocesses data for YOLOv3 network
* **SaveBoxesDarknetFormat**: Saves results of detection network in Darknet format for mAP calculation 

#### `graph_resnet50_dpuczdx8g_zcu104_tvm.json` contains Tensorflow Resnet50 graph. Below are the kernels used in the graph
* **ClassificationImreadResizeCCrop**: Resize the image to 256x256, centre crops 224x224 and  perfoms mean subtraction. 
* **TvmKernel**: Runs Tensorflow Resnet50 model through tvm runtime with DPUCZDX8G-zcu104 and CPU.
* **ClassificationAccuracy**: Measures & reports accuracy of a model (Top-1/Top-5)


### Run Examples on Alveo-U200/Alveo-U250

#### Activate conda env and get Image Dataset 

Download a minimal validation set for [COCO](http://cocodataset.org/#home) using [Collective Knowledge (CK)](https://github.com/ctuning).

:pushpin: **Note:** Skip, if you have already run the below steps.

:pushpin: **Note:** Please make sure you are already inside Vitis-AI docker

:pushpin: **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

```sh
cd ${VAI_HOME}/external/tvm/examples/AKS
# Activate conda env
conda activate vitis-ai-tensorflow
python -m ck pull repo:ck-env

# Download COCO dataset (This may take a while as COCO val dataset is more than 6 GB in size)
python -m ck install package:dataset-coco-2014-val

```

#### Build Common Kernels
```
cd ${VAI_HOME}/src/AKS
./cmake-kernels.sh --clean
```

#### Build TVM Kernels
```sh
cd ${VAI_HOME}/third_party/tvm/examples/AKS
./cmake-kernels.sh --clean
```
#### Build Examples
```sh
cd ${VAI_HOME}/third_party/tvm/examples/AKS
./cmake-examples.sh --clean
```

#### Running the Application

- Running Resnet50 using graph_resnet50_dpucadf8h_tvm.json 
```
./run_resnet50.sh -d /opt/tvm-vai/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/
```
- Running Yolov3 using graph_yolov3_dpucadf8h_tvm.json
```
./run_yolov3.sh -d  ${HOME}/CK-TOOLS/dataset-coco-2014-val/val2014
```

#### Performance

**Note that the overall performance of the application depends on the available system resources.**

By default, TVM uses all availble physical cores in the system with CPU Affinity. To get better performance, user needs to turn off CPU Afinity by setting  `export TVM_BIND_THREADS=0` and tune TVM_NUM_THREADS based on network.

Following table shows the end-to-end application throughput.
| MODEL | TVM_NUM_THREADS |E2E Throughput (FPS) |  
|:-:|:-:|:-:|
| Tensorflow Resnet50 | 1 | 1937|
| Darknet YoloV3 | 8 | 50 |

### Run examples on Edge Devices

#### Setup the Target Device

Please follow the instructions [here](../../../tvm/docs/running_on_zynq.md#petalinux-setup) to setup your target device with correct SD-card image and install tvm runtime. 

### Get Image Dataset

:pushpin: **Note:** If you have active internet connectivity on the target board, you can download the dataset directly on the target. If not, copy the dataset to the SD-Card after downloading it on the host system.

Below steps provide a way to download a minimal version of ImageNet validation dataset on host system using docker.

:pushpin: **Note:** Please make sure you are already inside Vitis-AI docker

:pushpin: **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

Download a minimal validation set for [Imagenet2012](http://www.image-net.org/challenges/LSVRC/2012/) and [COCO](http://cocodataset.org/#home) using [Collective Knowledge (CK)](https://github.com/ctuning) on host with Vitis-AI docker and copy it to SD-card.

```sh
# Activate conda env
conda activate vitis-ai-caffe
python -m ck pull repo:ck-env
python -m ck install package:imagenet-2012-val-min
conda deactivate
```

#### Get AKS library, kernels and examples

Copy the `Vitis-AI` directory to SD-card. Once all copying is finished, boot the device with the SD card.

#### Copy the AKS repo and Image Dataset to home directory
:pushpin: **Note:** Following instructions assume that files which are copied to SD-card are located at `<path-to-copied-files>` after you boot into the board. For example, in our test device, the location is `/mnt/sd-mmcblk0p1/`.


```sh
cd <path-to-copied-files>/src/AKS/
```

#### Build Kernels and Examples on the target device

Use following commands to build these kernels and examples.

#### Build Common Kernels
```
cd <path-to-copied-files>/third_party/tvm/examples/AKS
chmod +x cmake-kernels.sh
./cmake-kernels.sh --clean
```

#### Build TVM Kernels
```sh
cd <path-to-copied-files>/third_party/tvm/examples/AKS
chmod +x cmake-kernels.sh
./cmake-kernels.sh --clean
```
#### Build Examples
```sh
cd <path-to-copied-files>/third_party/tvm/examples/AKS
chmod +x cmake-examples.sh
./cmake-examples.sh --clean
```
#### Running Tensorflow Resnet50 on ZCU104

```
./run_resnet50_dpuczdx8g_zcu104.sh -d <path-to-copied-files>/dataset-imagenet-ilsvrc2012-val-min/ 
```
#### Performance
By default, TVM uses all availble physical cores in the system with CPU Affinity. To get better performance, user needs to turn off CPU Afinity by setting  `export TVM_BIND_THREADS=0` and tune TVM_NUM_THREADS based on network.

Following table shows the end-to-end application throughput.
| MODEL | TVM_NUM_THREADS |E2E Throughput (FPS) |  
|:-:|:-:|:-:|
| Tensorflow Resnet50 | 1 | 143|
