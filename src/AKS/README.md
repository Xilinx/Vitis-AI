<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>

# AI Kernel Scheduler

## Table of Contents

* [Introduction](#introduction)
* [What's New](#whats-new)
* [Installation](#installation)
* [Getting Started](#getting-started)
* [Run Examples on Alveo-U50](#run-examples-on-alveo-u50)
* [Run Examples on Alveo-U200/Alveo-U250 with Batch DPU](#run-examples-on-alveo-u200alveo-u250-with-batch-dpu)
* [Run Examples on Edge Devices](#run-examples-on-edge-devices)
* [Tuning Performance](#tuning-performance)
* [Graphs & Kernels](#graphs--kernels)
* [Integrating AKS in Applications](docs/API.md#Integrating-AI-Kernel-Scheduler-in-Your-Application)
* [Build Custom Graphs](docs/API.md#Creating-Custom-Graphs-for-AI-Kernel-Scheduler)
* [Build Custom Kernels](docs/API.md#Creating-Custom-AKS-Kernel)

## Introduction

Real world deep learning applications involve multi-stage data processing pipelines which include many compute intensive pre-processing operations like data loading from disk, decoding, resizing, color space conversion, scaling, croping etc. and multiple ML networks of different kinds like CNN etc. and various post-processing operations like NMS etc.

**AI Kernel Scheduler** or **AKS** is an application to automatically and efficiently pipeline such **graphs** without much effort from the users. It provides various kinds of **kernels** for every stage of the complex graphs which are plug and play and are highly configurable. For example, pre-processing kernels like image decode and resize, CNN kernel like Vitis AI's DPU Kernel and post processing kernels like SoftMax & NMS. Users can create their graphs using kernels and execute their jobs seamlessly to get the maximum performance.

## What's New?

### Updates (version 1.4)

- New Unified DPU kernel for all supported Xilinx DPUs
- New examples for Alveo-u50/u200/u250
- Removed support for DPUCADX8G
- Kernel for FPGA Accelerated Optical Flow (*Alveo-u200*) with DPUCADF8H

### Updates (version 1.3)

- Kernels for new DPUs
  - DPUCZDZ8G (for ZCU102 & ZCU104 Edge Platforms)
  - DPUCAHX8H (for Alveo-U50, HBM devices)
  - DPUCADF8H (for Alveo-U200/U250, New Batch DPU Engine)
- Kernels for FPGA Accelerated Optical Flow (*Alveo-u200*)

### Updates (version 1.2)

- Multi-FPGA Support in DPUCADX8G kernel
- New Graphs (Face Detect, Yolo-v2)
- Python Kernel
- Example with Video Stream Input
- FPGA Accelerated Pre-Precessing Support (*Alveo-u200*)

## Installation

AKS comes pre-installed in Vitis-AI docker or Vitis-AI edge image.

If a manual installation is required for some reasons, use `cmake.sh` script.

```sh
./cmake.sh --help
```

| Option | Description | Possible Values |
|:-------|:------------|:----------------|
| --type | Set Build Type | release, debug (Default)|
| --clean| Discard previous builds and rebuild | - |
| --pack | Enable packing and set package format | deb, rpm |
| --build-dir | Set customized build directory| Optional |
| --install-prefix | Set customized install prefix | Optional |
| --help | Show help | - |

For example, to generate a DEB package

```sh
./cmake.sh --clean --type=release --pack=deb
```

## Getting Started

Vitis-AI AKS provides shell scripts to build and run various examples provided with this package. Please go through below section to familiarize yourself with the scripts.

### Build Kernels

The shell script [cmake-kernels.sh](./cmake-kernels.sh) is provided to build AKS kernels.

```sh
# Check Usage
./cmake-kernels.sh --help
```

| Option | Description | Possible Values |
|:-------|:------------|:----------------|
| --name | Build a specific kernel | Any kernel from `kernel_src` |
| --type | Set Build Type | release (Default), debug |
| --clean| Discard previous builds and rebuild | - |
| --clean-only | Discard builds/Clean | - |
| --help | Show help | - |

### Build Examples

The shell script [cmake-examples.sh](./cmake-examples.sh) is provided to build AKS examples.

```sh
# Check Usage
./cmake-examples.sh --help
```

| Option | Description | Possible Values |
|:-------|:------------|:----------------|
| --type | Set Build Type | release (Default), debug |
| --clean| Discard previous builds and rebuild | - |
| --clean-only | Discard builds/Clean | - |
| --help | Show help | - |

### Run Examples

The shell script [aks.sh](./aks.sh) is provided to run the AKS examples.

```sh
# Check Usage
./aks.sh --help
```
|Option | Description | Possible Values |
|:-----|:-----|:-----|
|-m, --model | Model Graphs | run `./aks.sh -h` to get possible values |
|-d1, --dir1 | Image Directory for Classification Graphs | Path to directory |
|-d2, --dir2 | Image Directory for Detection Graphs | Path to directory |
|-vf, --video| Video File | Path to video file |
|-v, --verbose| Defines verbosity of log messages | 0 - Only Warnings & Errors, 1 - Important Information, warnings & errors, 2 - All debug, performance metrics, warnings & errors |
|-h, --help  | Print Usage | - |


## **Run examples on Alveo-U50LV**

Below example uses **DPUCAHX8H** IP for CNN Inference Acceleration on Alveo-U50 devices.

### Setup

Follow [Setup Alveo-U50LV](../../setup/alveo/README.md) page to setup your host system with Alveo-U50LV cards (Skip if already done).

### Get Image Dataset

Download a minimal validation set for [Imagenet2012](http://www.image-net.org/challenges/LSVRC/2012/) using [Collective Knowledge (CK)](https://github.com/ctuning).

:pushpin: **Note:** Skip, if you have already run the below steps.

:pushpin: **Note:** Please make sure you are already inside Vitis-AI docker

:pushpin: **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

```sh
cd ${VAI_HOME}/src/AKS

# Activate conda env
conda activate vitis-ai-tensorflow
python -m ck pull repo:ck-env
python -m ck install package:imagenet-2012-val-min

# We don't need conda env for running examples with this DPU
conda deactivate
```

For face detection example, use any face images you have or download [Face Detection Data Set and Benchmark (FDDB)](http://vis-www.cs.umass.edu/fddb/) dataset.

:pushpin: **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

``` sh
mkdir ~/FDDB
wget http://vis-www.cs.umass.edu/fddb/originalPics.tar.gz
tar -xvzf originalPics.tar.gz -C ~/FDDB
```

### Build Kernels and Examples

We have provided a few kernels in the [aks/kernel_src](./kernel_src) directory and examples in the [aks/examples](./examples) directory using C++ AKS APIs.
Use following commands to build these kernels and examples.

```sh
# Buld kernels
./cmake-kernels.sh --clean

# Build examples
./cmake-examples.sh --clean
```

### Download Compiled Models

```sh
# Download models
python3 artifacts.py -d u50lv_v3e
```

### Run Examples

- Resnet50
    ```sh
    # C++
    ./aks.sh -m tf_resnet_v1_50_u50lv_v3e -d1 ${HOME}/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min
    ```

- Face Detect (DenseBox 320x320)
    ```sh
    ./aks.sh -m cf_densebox_320_320_u50lv_v3e -d1 ${HOME}/FDDB/2002/07/19/big
    ```

## **Run Examples on Alveo-U200/Alveo-U250 with Batch DPU**

These examples use **DPUCADF8H** IP for CNN Inference Acceleration on Alveo-U200/Alveo-U250 devices.

### Setup

Follow [Setup Alveo-U200/U250](../../setup/alveo/README.md) cards page to setup your cards on the host system (skip if already done).

:pushpin: **Note:** Skip, if you have already run the below steps.

### Get Image Dataset

Download a minimal validation set for [Imagenet2012](http://www.image-net.org/challenges/LSVRC/2012/) using [Collective Knowledge (CK)](https://github.com/ctuning).

:pushpin: **Note:** Skip, if you have already run the below steps.

:pushpin: **Note:** Please make sure you are already inside Vitis-AI docker

:pushpin: **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

```sh
cd ${VAI_HOME}/src/AKS

# Activate conda env
conda activate vitis-ai-tensorflow
python -m ck pull repo:ck-env
python -m ck install package:imagenet-2012-val-min

# We don't need conda env for running examples with this DPU
conda deactivate
```

For face detection example, use any face images you have or download [Face Detection Data Set and Benchmark (FDDB)](http://vis-www.cs.umass.edu/fddb/) dataset.

:pushpin: **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

``` sh
mkdir ~/FDDB
wget http://vis-www.cs.umass.edu/fddb/originalPics.tar.gz
tar -xvzf originalPics.tar.gz -C ~/FDDB
```

### Build Kernels and Examples

We have provided a few kernels in the [aks/kernel_src](./kernel_src) directory and examples in the [aks/examples](./examples) directory using C++ AKS APIs.
Use following commands to build these kernels and examples.

```sh
# Build kernels
./cmake-kernels.sh --clean

# Build examples
./cmake-examples.sh --clean
```

### Download Compiled Models

```sh
# Download models
python3 artifacts.py -d u200_u250
```

### Run Examples

- TensorFlow Resnet50-v1
    ```sh
    ./aks.sh -m tf_resnet_v1_50_u200_u250 -d1 ${HOME}/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min
    ```

- Face Detect (DenseBox 320x320)
    ```sh
    ./aks.sh -m cf_densebox_320_320_u200_u250 -d1 ${HOME}/FDDB/2002/07/19/big
    ```

## **Run examples on Edge Devices**

Below example uses **DPUCZDX8G** IP for CNN Inference Acceleration on edge devices like ZCU102/ZCU104.

Following packages are required to run example on edge device:
1. SD card system image
2. AKS repo
3. Image Dataset

### Setup the Target Device

Please follow the instructions here to setup your target device with correct SD-card image: [link](../../examples/VART/README.md#setting-up-the-target)

### Get Image Dataset

:pushpin: **Note:** If you have active internet connectivity on the target board, you can download the dataset directly on the target. If not, copy the dataset to the SD-Card after downloading it on the host system.

Below steps provide a way to download a minimal version of ImageNet validation dataset on host system using docker.

:pushpin: **Note:** Please make sure you are already inside Vitis-AI docker

:pushpin: **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

Download a minimal validation set for [Imagenet2012](http://www.image-net.org/challenges/LSVRC/2012/) and [COCO](http://cocodataset.org/#home) using [Collective Knowledge (CK)](https://github.com/ctuning) on host with Vitis-AI docker and copy it to SD-card.

```sh
# Activate conda env
conda activate vitis-ai-tensorflow
python -m ck pull repo:ck-env
python -m ck install package:imagenet-2012-val-min

conda deactivate
```

### Get AKS library, kernels and examples

Copy the `Vitis-AI/src/AKS` directory to SD-card.

Once all copying is finished, boot the device with the SD card.

### Copy the AKS repo and Image Dataset to home directory
:pushpin: **Note:** Following instructions assume that files which are copied to SD-card are located at `<path-to-copied-files>` after you boot into the board. For example, in our test device, the location is `/mnt/sd-mmcblk0p1/` or `/run/media/mmcblk0p1/`.

Now copy the AKS repo and image dataset to home directory.

```sh
cp <path-to-copied-files>/AKS ~/
cp <path-to-copied-files>/dataset-imagenet-ilsvrc2012-val-min ~/
cd ~/AKS
```

### Build Kernels and Examples on the target device

Use following commands to build these kernels and examples.

  ```sh
  # Build kernels
  chmod +x cmake-kernels.sh
  ./cmake-kernels.sh --clean

  # Build examples
  chmod +x cmake-examples.sh
  ./cmake-examples.sh --clean
  ```

### Run Examples

- Resnet50

    ```sh
    chmod +x aks.sh

    # C++
    ./aks.sh -m cf_resnet50_zcu_102_104 -d1 ~/dataset-imagenet-ilsvrc2012-val-min/
    ```

## Tuning Performance

AKS provides a report on various performance metrics of internal worker threads and various kernels. This info can be utilized to understand the bottlenecks in the pipeline and tune the number of CPU workers for each kernel.

This report can be enabled by setting an AKS environment variable, `export AKS_VERBOSE=2`. In above examples, the same can be achieved via appending `-v 2` to every command.

```sh
# C++
./aks.sh -m googlenet -v 2
```

Similarly, number of CPU threads for a kernel can be specified by a field, `num_cu : N`, in corresponding kernel JSON, where N is the number of CPU threads. For example, see [ClassificationImreadPreProcess Kernel JSON](kernel_zoo/kernel_classification_imread_preprocess.json).

Let's take a look at a sample report for googlenet with 2 preprocessing threads (These numbers will vary depending upon your System configuration)

```sh
[INFO] Total Time (s): 55.3752

[DEBUG] Worker: ClassificationAccuracy_0 - Total jobs : 50000
[DEBUG] |--- Blocking Kernel : Exec time (s) : 0.46, Peak FPS possible: 108902.12, Utilization : 0.83%

[DEBUG] Worker: ClassificationImreadPreProcess_1 - Total jobs : 24942
[DEBUG] |--- Blocking Kernel : Exec time (s) : 55.11, Peak FPS possible: 452.57, Utilization : 99.52%

[DEBUG] Worker: ClassificationFCSoftMaxTopK_0 - Total jobs : 50000
[DEBUG] |--- Blocking Kernel : Exec time (s) : 11.30, Peak FPS possible: 4424.70, Utilization : 20.41%

[DEBUG] Worker: ClassificationImreadPreProcess_0 - Total jobs : 25058
[DEBUG] |--- Blocking Kernel : Exec time (s) : 55.12, Peak FPS possible: 454.63, Utilization : 99.53%

[DEBUG] Worker: DPURunner_0 - Total jobs : 50000
[DEBUG] |--- Async Kernel : Submit time (s) : 1.70, Wait time (s) : 0.02, Kernel Active Time (s): 55.25
```

The report shows details on how each worker thread spent its time.

- `Worker : ClassificationAccuracy_0` shows the kernel associated with each worker thread.
- `Total jobs : 50000` tells you how many times a kernel was executed by its worker thread.
    - If there are multiple threads for a kernel, total jobs will be distributed among them.
- `Blocking Kernel / Async Kernel` tells whether kernel was a blocking/non-blocking kernel
    - Blocking Kernel & Non-blocking kernel will have different types of performance metrics.
- `Exec time` is the time spent by a worker thread doing the actual work, i.e. kernel execution.
- `Peak FPS possible` is the theoretical Peak FPS you are supposed to get if this particular blocking kernel is the bottleneck in the pipeline.
- `Utilization` of a blocking kernel is the percentage time it spent on doing the useful work.
    - **Low utilization** denotes that the worker was either waiting for inputs from previous node or waiting to push the output to next node. So either previous/next node could be a bottleneck.
    - **High utilization** denotes that this kernel itself could be the bottleneck and needs more worker threads to distribute this kernel's jobs.
- `Submit time` is the total time spent by the worker to submit a job to an async kernel. It should be ideally very low.
- `Wait time` is the time spent by the worker thread waiting for a background thread to wait for result of async kernel. Again, it should be very low.
- `Kernel Active Time` is the time an async kernel has atleast one job enqueued in it.
    - Compare this with the time your application was running.
    - **Low active time** denotes that the async kernel (mostly a HW IP) has been idle. It means one of the previous node is the bottleneck. So we need to push more jobs to make async kernel always busy by allotting more worker threads for previous nodes.
    - **High active time** denotes that the async kernel is already loaded up and probably this kernel could be the bottleneck.
    - High active time doesn't mean async kernel is running with maximum performance. It only means there are jobs queued up in the async kernel.

In the above example, both the preprocessing threads (ClassificationImreadPreProcess_*) are running at 99.5% utilization. It gives an hint that allotting more worker threads to `ClassificationImreadPreProcess` kernel would give better performance in this case.

Pushing jobs to AKS takes very less time. So to limit the memory usage, AKS limits the maximum number of active jobs in the system manager to **128**. This limit can be controlled with environment variable, **`AKS_MAX_CONCURRENT_JOBS`**. For example: **`export AKS_MAX_CONCURRENT_JOBS = 32`**.

Depending upon the situation, this limit will have to be varied. If a graph's nodes generate large temporary data, this may have to be reduced to a lower value to limit overall memory usage. If the graph has very less execution time and memory usage, then this limit has to be increased to push more jobs to the system to get better performance.

## Graphs & Kernels

As mentioned in the previous sections, AKS pipelines AI graphs specified in the AKS graph JSON format. These graphs make use of the AKS kernels for running various nodes in the graph. The details about the formats of graphs and kernels are captured later in this document. This section lists down the sample graphs and kernels being used in the provided examples.

### Sample Graphs

Below is the list of the sample graphs provided as part of AKS examples. User can [write a new graph](docs/API.md#Creating-Custom-Graphs-for-AI-Kernel-Scheduler) by taking these as reference or can copy and modify a graph which is closest to the target graph.

| Graph | Description |
|:-----|:-----|
| resnet50 | Reads and Pre-Processes images, Runs inference on selected DPU, Post Processes data and Reports accuracy |
| facedetect | Reads and pre-process the images, runs inference on selected DPU, applies post-processing and returns results |

### Sample Kernels

While users can create their own kernels, AKS provides some basic kernels typically used for classification and detection. Users can quickly use these kernels in their graph or build their own kernels as documented [here](docs/API.md#Creating-Custom-AKS-Kernel). Below is the complete list of kernels used in the examples.

<table>
    <thead>
        <tr>
            <th>Category</th>
            <th>Name</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>DPU (Inference Kernels)</td>
            <td>DPURunner</td>
            <td>Runs inference with Xilinx DPUs</td>
        </tr>
        <tr>
            <td rowspan=6>Pre/Post-process for Classification networks </td>
            <td>ClassificationAccuracy</td>
            <td>Measures & reports accuracy of a classification network (Top-1/Top-5)</td>
        </tr>
        <tr>
            <td>ClassificationImreadResizeCCrop</td>
            <td>Reads images, resizes and center crops</td>
        </tr>
        <tr>
            <td>ClassificationImreadPreProcess</td>
            <td>Reads images and preprocess them for classification network</td>
        </tr>
        <tr>
            <td>ClassificationPreProcess</td>
            <td>Preprocesses images for a classification network</td>
        </tr>
        <tr>
            <td>ClassificationPostProcess</td>
            <td>Performs Softmax+TopK for a classification network</td>
        </tr>
        <tr>
            <td>MeanSubtract</td>
            <td>Performs mean subtraction on input data</td>
        </tr>
        <tr>
            <td rowspan=6>Pre/Post-process for Detection networks</td>
            <td>DetectionImreadPreProcess</td>
            <td>Reads and Preprocesses an image for YOLO network </td>
        </tr>
        <tr>
            <td>DetectionPreProcess</td>
            <td>Preprocesses an image for YOLO network </td>
        </tr>
        <tr>
            <td>SaveBoxesDarknetFormat</td>
            <td>Saves results of detection network in Darknet format for mAP calculation</td>
        </tr>
        <tr>
            <td>YoloPostProcess</td>
            <td>Postprocesses data for YOLO v2/v3 network</td>
        </tr>
        <tr>
            <td>FaceDetectPostProcess</td>
            <td>Postprocesses data for Face Detection networks</td>
        </tr>
        <tr>
            <td>FaceDetectImreadPreProcess</td>
            <td>Reads images and pre processes them for face detection networks</td>
        </tr>
        <tr>
            <td>FaceDetectPreProcess</td>
            <td>Pre processes images for face detection networks</td>
        </tr>
        <tr>
            <td rowspan=2>Misc.</td>
            <td>ImageRead</td>
            <td>Reads an image with provided path</td>
        </tr>
        <tr>
            <td>OpticalFlowDenseNonPyrLK</td>
            <td>Run non-pyramidal LK Optical Flow (Available only with DPUCADX8G on Alveo-U200</td>
        </tr>
    </tbody>
</table>
