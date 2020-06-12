# AI Kernel Scheduler
## Introduction
Real world deep learning applications involve multi-stage data processing pipelines which include many compute intensive pre-processing operations like data loading from disk, decoding, resizing, color space conversion, scaling, croping etc. and multiple ML networks of different kinds like CNN etc. and various post-processing operations like NMS etc. 

**AI Kernel Scheduler** or AKS is an application to automatically and efficiently pipeline such graphs without much effort from the users. It provides various kinds of kernels for every stage of the complex graphs which are plug and play and are highly configurable. For example, pre-processing kernels like image decode and resize, CNN kernel like Vitis AI's FPGA Kernel and post processing kernels like SoftMax & NMS. Users can create their graphs using kernels and execute their jobs seamlessly to get the maximum performance.

## Running Examples
Try out the examples provided in `${VAI_ALVEO_ROOT}/apps/aks` directory. The shell script [aks.sh](./aks.sh) runs the corresponding **C++ / Python** executables. 

### Prerequisites

Download a minimal validation set for [Imagenet2012](http://www.image-net.org/challenges/LSVRC/2012/) using [Collective Knowledge (CK)](https://github.com/ctuning). 

> **NOTE:** Skip if you have already ran the below steps before.

```sh
conda activate vitis-ai-caffe 
```
```sh
cd ${VAI_ALVEO_ROOT}/apps/aks

python -m ck pull repo:ck-env
python -m ck install package:imagenet-2012-val-min
python -m ck install package:imagenet-2012-aux

head -n 500 $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt > $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val_map.txt

head -n 500 $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt > $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val.txt

python ${VAI_ALVEO_ROOT}/examples/caffe/resize.py $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min 224 224
```

Familiarize yourself with the script usage by running below command.
```sh
# Check Usage
./aks.sh -h
```

### C++ / Python APIs
We have provided examples in the [aks/examples](./examples) directory. For **AKS**, we have provided both C++ and Python APIs and examples. 
All of them come with prebuilt executables. Use following commands to run these examples. 

1. Resnet50

    ```sh
    # C++
    ./aks.sh -m resnet50
    # Python
    ./aks.sh  -i py -m resnet50
    ```

1. GoogleNet

    ```sh
    # C++
    ./aks.sh -m googlenet
    # Python
    ./aks.sh  -i py -m googlenet
    ```

1. Tiny YOLOv3

    ```sh
    # C++
    ./aks.sh -m tinyyolov3
    # Python
    ./aks.sh  -i py -m tinyyolov3
    ```

1. Standard YOLOv2

    ```sh
    # C++
    ./aks.sh -m stdyolov2
    # Python
    ./aks.sh  -i py -m stdyolov2
    ```

1. Multi-Net (Googlenet + Resnet50)

    ```sh
    # C++
    ./aks.sh -m googlenet_resnet50
    # Python
    ./aks.sh  -i py -m googlenet_resnet50
    ```

1. Multi-Net (Googlenet + TinyYolov3)

    ```sh
    # C++
    ./aks.sh -m googlenet_tinyyolov3
    # Python
    ./aks.sh  -i py -m googlenet_tinyyolov3
    ```

1. GoogleNet (with FPGA accelerated pre-processing)

    ```sh
    ###
    # Currently supported on Alveo-u200
    ###

    # C++
    ./aks.sh -m googlenet_pp_accel
    # Python
    ./aks.sh  -i py -m googlenet_pp_accel
    ```

## Integrating AI Kernel Scheduler in Your Application

Users can create their own pipelines by defining custom graphs and use AI Kernel Scheduler in their applications through the C++ or Python APIs. Code of the C++ and Python examples mentioned in previous section is available in corresponding directories. This code can be referred to integrate AKS in any application.

Following are the steps for integrating AKS in any user application:

### Include the Header File or Import the Python Module

```cpp
// C++
#include <AksSystemManagerExt.h>
```
```python
# Python
import aks
```

### Get the System Manager Handle

```cpp
// C++
auto sysMan = AKS::SysManagerExt::getGlobal();
```
```python
# Python
sysMan = aks.SysManager()
```

### Load Kernels

Pass the directory containing AKS Kernel Configs to the following API.

```cpp
// C++
sysMan->loadKernels("path-to-kernel-zoo-dir");
```
```python
# Python
sysMan.loadKernels("path-to-kernel-zoo-dir")
```

### Pre-Defined Kernels
Users can create their own kernels but AKS comes with some predefined kernels, as listed below. These are available in kernel_zoo directory.

- DPUv1 (`DPUv1_Runner`) - Runs the network on FPGA

- Image Read (`image_read`) - Reads the image from disk

- Classification Pre-processing (`classificationImreadPre`)- Does the classification preprocessing tasks like image read, resize, crop, mean subtraction

- Classification Post-processing (`classificationFCSoftMaxTopK`)- Computes FC, Softmax & TopK

- Classification Accuracy (`classificationAccuracy`) - Calculates the accuracy for a classification network. Corresponding kernel needs the groundtruth file.

- Detection Pre-processing (`detectionImreadPre`) - Does the detection preprocessing tasks like image read, resize, crop, letterbox, color conversion etc.

- Yolo Post-Processing (`YoloPostProcess`) - Does post processing tasks like NMS, softmax for Yolo varients.

- Caffe (`caffe`) - Executes the network in Caffe. Only a single instance of the same can be run.

### Load Graph(s)
Pass the directory containing AKS Graphs or the individual Graph JSON file to the following API.

```cpp
// C++
sysMan->loadGraphs("path-to-graph-zoo-dir");
```
```python
# Python
sysMan.loadGraphs("path-to-graph-zoo-dir")
```

### Pre-Defined Graphs
Users can create their own graphs but AKS comes with some predefined kernels, as listed below. These are available in graph_zoo directory.

- GoogleNet - This graph has 4 nodes namely
    * Preprocess - Runs with classification preprocess kernel
    * GoogleNet Network - Runs with DPUv1 FPGA kernel
    * Postprocess - Runs with classification post-process kernel which computes FC, Softmax and Computes TopK
    * Accuracy - Runs the accuracy calculation. Needs the ground truth file to be passed to the node

- Resnet50 - This graph also has 4 nodes exactly same as Googlenet graph. The only difference is the `vitis_rundir` parameter for the DPUv1 node which points to a different directory because the network to be run is different.

- Tiny YOLOv3 - This graph has 4 nodes 
    * Preprocess - Runs with detection preprocess kernel
    * TinyYolov3 - Runs with DPUv1 FPGA kernel        
    * Postprocess - Runs with detection postprocess kernel which performs NMS and other tasks
    * Save results - Runs the save result kernel and dumps output results to a directory for each image

- Standard YOLOv2 - This graph has 5 nodes 
    * Preprocess - Runs with detection preprocess kernel
    * Standard YOLOv2 - First part of the network runs with DPUv1 FPGA kernel        
    * Reorg - Reorg and remaining layers runs with Caffe Kernel
    * Postprocess - Runs with detection postprocess kernel which performs NMS and other tasks
    * Save results - Runs the save result kernel and dumps output results to a directory for each image

### Get The Graph To Run

```cpp
// C++
auto graph = sysMan->getGraph("graph-name");
```
```python
# Python
graph = sysMan.getGraph("graph-name")
```

### Enqueue Jobs
```cpp
// C++
// create the data descriptor vector
std::vector<AKS::DataDescriptor> v(3); 

sysMan->enqueueJob (graph, imagePath, std::move(v), nullptr);
```
```python
# Python
sysMan.enqueueJob(graph, imagePath)
```
    

### Wait For Results

```cpp
// C++
sysMan->waitForAllResults();
```
```python
# Python
sysMan.waitForAllResults()
```

### Report Results (Optional)

This is an optional step to report any results anytime during the application. If the kernel has implemented a report function, it gets called on calling the following function. In our example graphs, accuracy kernel has implemented this function to report out the Top-1 and Top-5 accuracies based on request.

```cpp
// C++
sysMan->report(graph);
```
```python
# Python
sysMan.report(graph)
```

More details about the APIs above and their arguments can be found at [AksSysManagerExt.h](./ext/AksSysManagerExt.h) (for C++ interface) & [aks.py](./libs/aks.py) (for Python interface).

## Using Custom Graphs in AI Kernel Scheduler

> Please refer to pre-defined graphs in [graph_zoo](./graph_zoo) directory for reference.

AKS graph is a *Directed Acyclic Graph (DAG)* composed of execution nodes written in JSON format. Each node should be an instance of a kernel defined in [kernel_zoo](./kernel_zoo).

A graph has only two fields:

1. **`"graph_name"`** : This is any name to uniquely identify the graph
1. **`"node_list"`** : This is the list of nodes in the graph. Each node has mainly 3 fields.
    1. **`"node_name"`** : Name of the node
    1. **`"node_params"`** : Parameters of the node. It depends upon the kernel definition.
    1. **`"next_node"`** : It is a list of node names to which this node is connected to. This defines the connectivity in the DAG.
        * For the last node, this field will be empty.

> Currently, AKS expects the graph to have a single input and a single output.


## Creating Custom AKS Kernel

> AKS provides a set of kernels that covers a wide range of use-cases in [kernel_zoo](./kernel_zoo). Please check them out.

Creating a custom kernel requires two components.

1. **Kernel Implementation** : This is the actual code that will be executed for a kernel.
1. **Kernel JSON** : This is the interface to the kernel that contains information like parameters, kernel lib path etc.

We will go in detail with a reference kernel, `Add Kernel`, which simply adds a constant value to the input.

### Kernel Implementation

All kernels should inherit from an abstract class, **`KernelBase`** (*Please see the [header](ext/AksKernelBase.h) for detailed documentation*). Check the [implementation](./examples/kernels/add/add_kernel.cpp) of `Add Kernel` for reference.

There should be a generator function named, **`getKernel()`**, which returns a pointer to Kernel instance. This function should be wrapped in `extern "C"`. If kernel generation requires any specific parameters, it can be passed to `getKernel()` through [**NodeParams**](ext/AksNodeParams.h). This is a generic dict structure filled by AKS with the data provided in the graph json.

**`KernelBase::exec_async()`** is the only pure virtual method. It takes four arguments. 

First two are input and output. They are of type `vector<DataDescriptor*>`. [DataDescriptor](ext/AksDataDescriptor.h) is a N-dimensional array structure to store input/output of each node. Inputs to a node are automatically filled by previous nodes' outputs. `exec_async()` of custom kernel should create as many outputs it wants and keep them in output. AKS will take care of the memory management.

Third argument is `NodeParams* params`. It is populated by AKS with all the node parameters from the graph json. This parameter is unique per node. If graph contains two nodes of same kernel, each of them will have their own `params`.

Fourth argument is `DynamicParamValues* dynParams`. It is passed by the user as an argument to `enqueueJob()`. This includes any input params that is common to all nodes in a graph, but different with each input. For eg: ground truth file for each input image in an object detection task.

Once the implementation is ready, it should be compiled to a shared library and keep it in [libs](./libs) directory. Please refer the [Makefile](./examples/kernels/add/Makefile) for `Add Kernel`.

### Kernel JSON

AKS crawls through all the kernel JSONs in [kernel_zoo](./kernel_zoo) to get information about each kernel present. Please refer [kernel JSON](kernel_zoo/kernel_add.json) for `Add Kernel`. 

Kernel JSON should have following fields :

1. **kernel_name** : Unique identifier for the kernel
1. **kernel_lib** : Path to the shared library of a kernel
1. **param_list** : List of parameters that need to be specified in the graph json. Field name and data type should be mentioned. For eg: For int variable named `adder`, use `"adder": { "type":"int" }`. For array type, use it like this : `"mean": { "type":"float_array" }`.
1. **kernel_type** : Possible values - **"cpp"**
1. **device_type** : Possible values - **"cpu"**, **"fpga"**

There are some optional fields :

1. **xclbin** : Path to the xclbin. Required only if **`"device_type" : "fpga"`**.
1. **num_cu** : Number of CPU threads to be used for a kernel. They are shared among multiple nodes of same kernel, if present, in the graph.

## Performance
These results are collected using a local server with below specs.
- CPU : Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
- Accelerator Card : Alveo-u250

#### Datasets used
- ImageNet2012 (50k images, resized to 224x224)
- COCO val2014 (40k images)

You may get a different resuls based on your system performance.

> **NOTE:** The performance numbers are in Images / second.

#### Classification 
|No. of Pre-Proc Threads | No. of Post-Proc Threads | GoogleNet | ResNet50 |
|:-----:|:-----:|:-----:|:-----:|
|1|1| 719.26  | 730.33  |
|2|1| 1458.86 | 1196.13 |
|4|1| 2888.95 | 1204.29 |
|8|1| 4129.46 | 1207.56 |

#### Detection 
|No. of Pre-Proc Threads | No. of Post-Proc Threads | Tiny Yolov3 | 
|:-----:|:-----:|:-----:|
|4 |1| 249.982 |
|8 |2| 412.533 |
|12|4| 719.824 |

## Future Work
### Additional Kernels

In addition to the predefined kernels mentioned in the section above, we plan to add few more useful kernels in upcoming releases.

- **Classification Pre-Processing FPGA Kernel** - This kernel will run the preprocessing steps (JPEG Decode, Resize, Mean Subtraction) on FPGA which is expected to be faster than the corresponding C++ or Python code running on CPU.

- **Detection Pre-Processing FPGA Kernel** - This kernel will run the preprocessing steps (JPEG Decode, Resize, LetterBox, Mean Subtraction) on FPGA which is expected to be faster than the corresponding C++ or Python code running on CPU.

- **Python Kernels** - These kernel will allow running any custom Python code for execution on any node in the graph.

### Additional Graphs
Though some of the popular graphs are already available ([graph_zoo](./graph_zoo)) in the current release, we plan to add some more popular graphs in upcoming releases.

- Face Detect

- Googlenet X + ML - This will make use of the classification pre-processing kernel for FPGA

- Yolov3 X + ML - This will make use of the detection pre-processing kernel for FPGA
