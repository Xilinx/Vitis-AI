# AI Kernel Scheduler
## Introduction
Real world deep learning applications involve multi-stage data processing pipelines which include many compute intensive pre-processing operations like data loading from disk, decoding, resizing, color space conversion, scaling, croping etc. and multiple ML networks of different kinds like CNN etc. and various post-processing operations like NMS etc. 

**AI Kernel Scheduler** or **AKS** is an application to automatically and efficiently pipeline such **graphs** without much effort from the users. It provides various kinds of **kernels** for every stage of the complex graphs which are plug and play and are highly configurable. For example, pre-processing kernels like image decode and resize, CNN kernel like Vitis AI's DPU Kernel and post processing kernels like SoftMax & NMS. Users can create their graphs using kernels and execute their jobs seamlessly to get the maximum performance.


## Updates

- Multi-FPGA Support in DPUCADX8G kernel
- New Graphs (Face Detect, Yolo-v2)
- Python Kernel
- Example with Video Stream Input
- FPGA Accelerated Pre-Precessing Support (*Alveo-u200*)

## Getting Started

### Run Examples

Try out the examples provided in `/workspace/alveo/apps/aks` directory. The shell script [aks.sh](./aks.sh) runs the corresponding **C++ / Python** executables. 

### Prerequisites

Download a minimal validation set for [Imagenet2012](http://www.image-net.org/challenges/LSVRC/2012/) and [COCO](http://cocodataset.org/#home) using [Collective Knowledge (CK)](https://github.com/ctuning). 

> **Note:** Skip, if you have already run the below steps.

> **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

```sh
# Activate Conda Environment
conda activate vitis-ai-caffe 
```
```sh
# Setup
source /workspace/alveo/overlaybins/setup.sh
```
```sh
cd ${VAI_ALVEO_ROOT}/apps/aks

python -m ck pull repo:ck-env

# Download ImageNet Dataset
python -m ck install package:imagenet-2012-val-min
python -m ck install package:imagenet-2012-aux

head -n 500 ${HOME}/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt > ${HOME}/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val_map.txt

head -n 500 ${HOME}/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt > ${HOME}/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val.txt

python ${VAI_ALVEO_ROOT}/examples/caffe/resize.py ${HOME}/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min 224 224

# To try out examples for detection models like Tiny-YOLO-v3 or Standard-YOLO-v2 
# Download COCO dataset (This may take a while as COCO val dataset is more than 6 GB in size)
python -m ck install package:dataset-coco-2014-val

# To try out face-detect example, download FDDB dataset.
cd ${VAI_ALVEO_ROOT}/apps/face_detect/FDDB
wget http://tamaraberg.com/faceDataset/originalPics.tar.gz
tar -xvf originalPics.tar.gz
cd -
```

Familiarize yourself with the script usage by running below command.
```sh
# Check Usage
./aks.sh -h
```
|Option | Description | Possible Values |
|:-----|:-----|:-----|
|-m, --model | Model Graphs | googlenet, resnet50, googlenet_resnet50, tinyyolov3, tinyyolov3_video, googlenet_tinyyolov3, stdyolov2, facedetect, googlenet_pp_accel |
|-n, --nfpga | Number of FPGAs | Max number of FPGAs connected to System supported |
|-i, --impl  | API Implementation | cpp, py |
|-d1, --dir1 | Image Directory for Classification Graphs | Path to directory |
|-d2, --dir2 | Image Directory for Detection Graphs | Path to directory |
|-vf, --video| Video File | Path to video file |
|-v, --verbose| Defines verbosity of log messages | 0 - Only Warnings & Errors, 1 - Important Information, warnings & errors, 2 - All debug, performance metrics, warnings & errors |
|-h, --help  | Print Usage | - |


### C++ / Python Examples
We have provided few examples in the [aks/examples](./examples) directory using both C++ and Python AKS APIs. 
All of them come with prebuilt executables. Use following commands to run these examples. 

#### Classification 

- Resnet50

    ```sh
    # C++
    ./aks.sh -m resnet50 -d1 ${HOME}/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min
    # Python
    ./aks.sh -i py -m resnet50 -d1 ${HOME}/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min
    ```

- GoogleNet

    ```sh
    # C++
    ./aks.sh -m googlenet -d1 ${HOME}/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min
    # Python
    ./aks.sh -i py -m googlenet -d1 ${HOME}/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min
    ```

- GoogleNet (with FPGA accelerated pre-processing)

    ```sh
    ###
    # Currently supported on Alveo-u200
    ###

    # C++
    ./aks.sh -m googlenet_pp_accel -d1 ${HOME}/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min
    # Python
    ./aks.sh -i py -m googlenet_pp_accel -d1 ${HOME}/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min
    ```

#### Detection

- Tiny YOLOv3

    ```sh
    # C++
    ./aks.sh -m tinyyolov3 -d1 <image-dir>
    # Python
    ./aks.sh -i py -m tinyyolov3 -d1 <image-dir>
    ```

- Tiny YOLOv3 (with video input)

    ```sh
    # C++
    ./aks.sh -m tinyyolov3_video -vf <video-file>
    ```

- Standard YOLOv2

    ```sh
    # C++
    ./aks.sh -m stdyolov2 -d1 <image-dir>
    # Python
    ./aks.sh -i py -m stdyolov2 -d1 <image-dir>
    ```

- Face Detect
    ```sh
    # C++
    ./aks.sh -m facedetect -d1 ../face_detect/FDDB
    # Python
    ./aks.sh -m facedetect -i py -d1 ../face_detect/FDDB
    ```

    >**INFO:** This writes the annotated output images to `face_outputs` directory. A corresponding text file representation is written to `face_results.txt`. This result writing has huge impact on application throughput. If you want to turn-off writing results and improve the performance, please provide empty strings to `save_result_txt` and `save_result_imgdir` fields in `graph_zoo/graph_facedetect.json`.

#### Multi-Net

- Googlenet + Resnet50

    ```sh
    # C++
    ./aks.sh -m googlenet_resnet50 -d1 <image-dir-for-googlenet> -d2 <image-dir-for-resnet50>
    # Python
    ./aks.sh -i py -m googlenet_resnet50 -d1 <image-dir-for-googlenet> -d2 <image-dir-for-resnet50>
    ```

- Googlenet + TinyYolov3

    ```sh
    # C++
    ./aks.sh -m googlenet_tinyyolov3 -d1 <image-dir-for-googlenet> -d2 <image-dir-for-tinyyolov3>
    # Python
    ./aks.sh  -i py -m googlenet_tinyyolov3 -d1 <image-dir-for-googlenet> -d2 <image-dir-for-tinyyolov3>
    ```

## Tuning Performance

AKS provides a report on various performance metrics of internal worker threads and various kernels. This info can be utilized to understand the bottlenecks in the pipeline and tune the number of CPU workers for each kernel. 

This report can be enabled by setting an AKS environment variable, `export AKS_VERBOSE=2`. In above examples, the same can be achieved via appending `-v 2` to every command.

```sh
# C++
./aks.sh -m googlenet -v 2
# Python
./aks.sh  -i py -m googlenet -v 2
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
[DEBUG] Worker: DPUCADX8GRunner_0 - Total jobs : 50000
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

## Performance
These results are collected using a local server with below specs.
- CPU : Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
- Accelerator Card : Alveo-u250

#### Datasets used
- ImageNet2012 (50k images, resized to 224x224)
- COCO val2014 (40k images)

You may get a different resuls based on your system performance.

> **Note:** The performance numbers are in Images / second.

#### Classification 
| Model | Images/Sec | # Pre-Proc Threads | # Post-Proc Threads | 
|:------|:------:|:------:|:------:|
|GoogleNet  | 719  | 1 | 1 |
|           | 1458 | 2 | 1 |
|           | 2888 | 4 | 1 |
|           | 4129 | 8 | 1 |
| ResNet50  | 730  | 1 | 1 |
|           | 1196 | 2 | 1 |
|           | 1204 | 4 | 1 |
|           | 1207 | 8 | 1 |


#### Detection

Input image resolution of the network is 608x608.

| Model | Images/Sec | # Pre-Proc Threads | # Post-Proc Threads | 
|:------|:------:|:------:|:------:|
| Tiny Yolov3       | 259     | 4   | 1 |
|                   | 412     | 8   | 2 |
|                   | 719     | 12  | 4 |
| Standard Yolov2   | 87      | 1   | 1 |
|                   | 140     | 2   | 1 |
|                   | 135     | 4   | 1 |


## Graphs & Kernels

As mentioned in the previous sections, AKS pipelines AI graphs specified in the AKS graph JSON format. These graphs make use of the AKS kernels for running various nodes in the graph. The details about the formats of graphs and kernels are captured later in this document. This section lists down the sample graphs and kernels being used in the provided examples.

### Sample Graphs

Below is the list of the sample graphs provided as part of AKS examples. User can [write a new graph](docs/API.md#Creating-Custom-Graphs-for-AI-Kernel-Scheduler) by taking these as reference or can copy and modify a graph which is closest to the target graph.

| Graph | Description |
|:-----|:-----|
| googlenet | Reads and Pre-Processes images, Runs inference on DPUCADX8G, Post Processes data and Reports accuracy |
| resnet50 | Reads and Pre-Processes images, Runs inference on DPUCADX8G, Post Processes data and Reports accuracy |
| googlenet_pp_accel | Reads images, Pre-Processes image data using FPGA accelerated pre-processing kernel, Runs inference on DPUCADX8G, Post Processes data and Reports accuracy |
| tinyyolov3     | Reads and Pre-Processes images, Runs inference on DPUCADX8G, Post Processes data and Saves detection results in text files in DarkNet format |
| std_yolov2_608 | Reads and Pre-Processes images, Runs inference on DPUCADX8G, Post Processes data and Saves detection results in text files in DarkNet format |
| facedetect | Performs face detection on FDDB Dataset images. It reads and pre-process the images, runs inference on DPUCADX8G, applies post-processing and saves the results as annotated images as well as a text file |

### Sample Kernels

While users can create their own kernels, AKS provides some basic kernels typically used for classification and detection. Users can quickly use these kernels in their graph or build their own kernels as documented [here](docs/API.md#Creating-Custom-AKS-Kernel). Below is the complete list of kernels used in the examples.

| Name | Description |
|:-------|:-------|
| DPUCADX8GRunner | Runs inference on DPUCADX8G with new VAI interface |
| DPUCADX8GNoRunner | Runs inference on DPUCADX8G with pre-VAI interface |
| CaffeKernel | Executes a network using Caffe framework |
| ImageRead | Reads an image with provided path |
| ClassificationAccuracy | Measures accuracy of a classification network (Top-1/Top-5) |
| ClassificationFCSoftMaxTopK | Performs FC+Softmax+TopK for a classification network |
| ClassificationImreadPreProcess | Reads an image and preprocess it for classification network |
| ClassificationPreProcess | Preprocesses an image for a classification network |
| ClassificationPostProcess | Performs Softmax+TopK for a classification network |
| DetectionImreadPreProcess | Reads and Preprocesses an image for YOLO network |
| DetectionPreProcess | Preprocesses an image for YOLO network |
| PythonKernel | Executes kernels written in Python |
| SaveBoxesDarknetFormat | Saves results of detection network in Darknet format for mAP calculation |
| YoloPostProcess | Postprocesses data for YOLO v2/v3 network |
| ClassificationPreProcessAccel | Performs FPGA accelerated pre-processing for classification networks |

## Additional Details

- [Integrating AKS in Applications](docs/API.md#Integrating-AI-Kernel-Scheduler-in-Your-Application)
- [Build Custom Graphs](docs/API.md#Creating-Custom-Graphs-for-AI-Kernel-Scheduler)
- [Build Custom Kernels](docs/API.md#Creating-Custom-AKS-Kernel)
- [Build Python Kernels](docs/API.md#Creating-Python-Kernels)
