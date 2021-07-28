<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1>
   </td>
 </tr>
</table>

# Release Notes

## Release 1.4
### New Features/Highlights
1. Support new platforms, including Versal ACAP platforms VCK190, VCK5000 and Kria SoM 
2. Better Pytorch and Tensorflow model support: Pytorch 1.5-1.7.1, improved quantization for Tensorflow 2.x models
3. New models, including 4D Radar detection, Image-Lidar sensor fusion, 3D detection & segmentation, multi-task, depth estimation, super resolution for automotive, smart medical and industrial vision applications
4. New Graph Runner API to deploy models with multiple subgraphs
5. DPUCADX8G (DPUv1)deprecated with DPUCADF8H (DPUv3Int8)
6. DPUCAHX8H (DPUv3E) and DPUCAHX8L (DPUv3ME) release with xo
7. Classification & Detection WAA examples for Versal (VCK190)


### Release Notes
#### Model Zoo
- 17 new models added, 109 total
  - 11 new Pytorch models
  - 5 new Tensorlfow models
  - 1 new Caffe model
- Added support for Pytorch, Tensorflow 2.3 models
- Added new application models
  - Medical and industrial vision: depth estimation, RGB-D segmentation, super resolution 
  - Automotive: 4D Radar detection, Image-Lidar sensor fusion, surround-view 3D detection, upgraded 3D segmentation and multi-task models
- EoU Enhancements:
  - provided automated download script to select models with model name and hardware platform

### Quantizer
- TensorFlow 2.x version
  - Support fast finetune in post-training quantization (PTQ)
  - Improved quantize-aware training (QAT) functions
  - Support more layers: swish/sigmoid, hard-swish, hard-sigmoid, LeakyRelu, nested tf.keras functional and sequential models
  - Support custom layers via subclassing tf.keras.layers and support custom quantization strategies
- Pytorch version
  - Support Pytorch 1.5-1.7.1 
  - Support more operators including hard-swish, hard-sigmoid
  - Support shared parameters in quantization
  - Enhanced quantization profiling and error check functions
  - Improved QAT functions
    - support training from PTQ results
    - support reused modules
    - support resuming training

### Optimizer
- TensorFlow
  - Support tf.keras APIs in TF1
  - Supports single GPU mode for model analysis
- Pytorch version
  - Improved easy-of-use with simplified APIs
  - Support torch.nn.ConvTranspose2d
  - Support reused modules

### Compiler
- Support ALU for DPUCVDX8G (xvDPU)
- Support cross-layer prefetch optimization option
- Support xmodel output nodes assignment
- Enabled features to implement zero-copy for DPUCZDX8G(DPUv2), DPUCAHX8H(DPUv3E) and DPUCAHX8L(DPUv3ME)
- Open-source network visualization tool Netron officially supports XIR

### AI Library
- Added support for 17 new models from AI Model Zoo
- Introduced new deploy APIs graph_runner, especially for models with multiple subgraphs
- Introduced new tool xdputil for DPU and xmodel debug
- Support new KV260 SoM kit
- Support DPUCVDX8G(xvDPU) on VCK190
- Support DPUCVDX8H(DPUv4E) on VCK5000

### AI Profiler
- Support new DPU IPs: DPUCAHX8L(DPUv3ME), DPUCVDX8G(xvDPU) and DPUCVDX8H(DPUv4E)
- Support DPUCZDX8G(DPUv2) and DPUCVDX8G(xvDPU) in Vivado flow
- Add Memory IO statistics

### VART
- Support Petalinux 2021.1 and OpenCV v4
- Update samples to use INT8 as the input instead of FP32

### DPU
- CNN DPU for Zynq SoC / MPSoC, DPUCZDX8G (DPUv2)
  - Upgraded to 2021.1

- CNN DPU for Alveo-HBM, DPUCAHX8H (DPUv3E), DPUCAHX8L (DPUv3ME)
  - Released xo

- CNN DPU for Alveo-DDR, DPUCADF8H (DPUv3Int8)
  - Support latest U250 platform (2020.2) 
  - Support latest U200 platform (2021.1)
  - Support AWS F1

- CNN DPU for Versal, DPUCVDX8G (xvDPU)
  - VCK190 DPU TRD
  - Provide basic unit C32 with 32-aie cores for a single batch
  - Support configurable batch size 1~6
  - Support new OPs: Global Average Pooling up to 256x256, Element Multiply, Hardsigmoid and Hardswish
  
- CNN DPU for Versal, DPUCVDX8H (DPUv4E)
  - Improved the DPU performance of small model inference with weight pre-fetch function

- CNN DPU for Alveo-DDR, DPUCADX8G (DPUv1)
  - Deprecated with DPUCADF8H (DPUv3Int8)

### Whole App Acceleration
- Multi Object Tracking (SORT) example on ZCU102 
- Classification & Detection App examples for Versal (VCK190)  
- Updated existing examples to XRT APIs and zero copy
- Added U200 (DPUv3INT8) based WAA TRD 
- Ported U200/250 examples to DPUCADF8H (DPUv3INT8)
- SSD-MobileNet U280 example accelerates both pre and post-processing on hardware

### AI Kernel Scheduler
- Unified DPU kernels into one and added samples for Alveo U200/250 (DPUv3INT8), U280, U50, U50lv

### TVM
- Support of all DPUs - ZCU102/4, U50, U200, U250, U280
- Using Petalinux for edge devices
- Increased throughput using AKS at the application level
- Yolov3 tutorial as python notebook

### Known Issues
- ZCU104 power patch fail to work in 2021.1. Board will hang or reboot with heavy workload

## Release 1.3
### New Features/Highlights
1. Added support for Pytorch and Tensorflow 2.3 frameworks
2. Added more ready-to-use AI models for a wider range of applications, including 3D point cloud detection and segmentation, COVID-19 chest image segmentation and other reference models
3. Unified XIR-based compilation flow from edge to cloud
4. Vitis AI Runtime (VART) fully open source
5. New RNN overlay for NLP applications
6. New CNN DPUs for the low-latency and higher throughput applications on Alveo cards
7. EoU enhancement with Beta version model partitioning and custom layer/operators plug-in

### Release Notes
#### Model Zoo
- 28 new models added, over 92 total
  - 13 new Pytorch models
  - 17 new Tensorlfow models, including 5 Tensorflow 2 models
  - 6 new Caffe models
- Added support for Pytorch, Tensorflow 2.3 models
- Added new application models
  - Medical: CT segmentation, medical robot instrument segmentation, Covid-19 chest radiograph segmentation and other reference models.
  - Automotive: added 3D point cloud detection, point cloud segmentation models
- EoU Enhancements:
  - Improved accuracy evaluation and quantization scripts for all models
  - Model zoo restructured with clearer model information

### Quantizer
- Added support for Pytorch and Tensorflow 2 frameworks
- Calibration and fine-tune quantization methods upgraded to support TensorFlow 2.3
- Improved quantization performance and added support for fine-tuning for Pytorch



### Optimizer
- Pytorch support added

### Compiler
- Added support for  Tensorflow 2.3
- Added support for all the new CNN DPUs on Alveo and Versal platforms
  1. DPUCAHX8L (DPUv3ME)
  2. DPUCADF8H (DPUv3INT8)
  3. DPUCVDX8G (Versal CNN DPU)
- EoU Enhancement
  1. Added support for model partition & custom layer/operators Plugin (Beta)
  2. AI compilation unified to the XIR-based compilation flow from edge to cloud platforms
  3. Supports hybrid compilation for customer accelerator & DPU for higher e2e performance

### AI Library
- Added support for 36 new models from AI Model Zoo
- Added supports for Xmodel compiled with XIR flow from edge to cloud
- Added support for DPUCAHX8L (DPUv3ME) on Alveo U280/U50
- Added support for supports DPUCVDX8G (Versal DPU) on VCK190



### AI Profiler

- Integrated with Vitis Analyzer 2020.2
  1. Use Vitis Analyzer 2020.2 as default GUI
  2. Added the profiling .csv file to be compatible with Vitis Analyzer
- Vaitrace supports profiling Python applications

### VART
- Fully open source in Vitis AI 1.3
- Added new Python APIs
  1. APIs for TensorBuffer Operation
  2. APIs of RunnerExt
- Added support for Xmodel compiled with XIR flow from edge to cloud
- Added support for all DPU for CNN and RNN
- Added supports for CNN DPU on Versal platforms

### DPU
- CNN DPU for Zynq SoC / MPSoC, DPUCZDX8G (DPUv2)
  - Extended stride from 1-4 to 1-8
  - Extended MaxPooling kernel from 1-8 to 1-256 to support Pointpillar network
  - Addd support for elew_mult feature
  - Optimized save engine to improve efficiency
  - Supported XIR based AI Compiler
  - EoU Enhancement
    - DPU TRD (DPUCZDX8G) upgraded from v3.2 to v3.3
    - Added support for Vitis GUI flow for the integration

- CNN DPU for Alveo-HBM, DPUCAHX8L (v3ME)
  - Released as xclbin
  - Added support for HBM Alveo cards U280, U50, U50LV
  - Optimized with back-to-back Conv & Depthwise Conv engines to increase computing parallelism
  - Designed hierarchical memory system, URAM & HBM, to maximize data movement
  - Added support for low-latency CNN inference for high resolutions images
  - Added support for XIR-based compilation flow

- CNN DPU for Alveo-DDR, DPUCADF8H (DPUv3Int8)
  - Released as xclbin
  - Added support for DDR Alveo cards U200 and U250, Cloud FaaS
  - 2x throughput improvement over DPUv1 in INT8 mode
  - High efficiency engine can reach ~80% efficiency
  - Drop-in replacement for DPUv1 features
    1. Streaming execution
    2. Multi-process support
  - Ready for Whole Application Acceleration workloads
  - Added Support for XIR-based compilation flow

- RNN DPU, DPURAHR16L (xRNN)
  - Released as xclbin
  - Supported platforms:
    1. Alveo U25 for batch 1
    2. Alveo U50lv for batch 7
  - RNN quantizer, INT16 (16bit)
  - RNN compiler
  - Unified XRNN runner in VART;
  - Supports three RNN models
    1. Customer Satisfaction
    2. IMDB Sentiment Detection
    3. Open Information Extraction



### Whole App Acceleration
- DPUv2 TRD flow to build from sources (see [WAA-TRD](https://github.com/xilinx/vitis-ai/dsa/WAA-TRD))
- DFx based TRD flow to build from pre-built IPs
  1. DPUCZDZ8G

- Existing WAA classification and detection examples ported to DPUv3e (earlier only for DPUv2 and DPUv1) (see [Whole-App-Acceleration](https://github.com/xilinx/vitis-ai/demo/Whole-App-Acceleration))
- Fall Detection App using DPUv1 and Accelerated Optical Flow (see [Fall Detection](https://github.com/xilinx/vitis-ai/demo/Whole-App-Acceleration/fall_detection))
- Detection Post Processing (NMS) Acceleration (see [ssd_mobilenet](https://github.com/xilinx/vitis-ai/demo/Whole-App-Acceleration/ssd_mobilenet))

### AI Kernel Scheduler
- Added kernels for new DPUs
  - DPUCZDZ8G (for edge devices - ZCU102, ZCU104)
  - DPUCAHX8H (for HBM devices - Alveo-U50)
- Added kernels for Accelerated Optical Flow

### TVM
- New Flow (BYOC) or running TVM supported models on DPUv1, DPUv2

### Known Issues
- Limitations for DPUCADF8H on U200/U250:  
  - Python API not yet supported   
  - Segmentation networks not yet supported   
  - Possible accuracy issue when accessing DPUCADF8H from multiple threads    
- ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
 vai-q-tensorflow2 1.3.0.dev0 requires dm-tree~=0.1.1, which is not installed.
 aiohttp 3.6.3 requires chardet<4.0,>=2.0, but you have chardet 4.0.0 which is incompatible.
 aiohttp 3.6.3 requires yarl<1.6.0,>=1.0, but you have yarl 1.6.3 which is incompatible.
   - Errors like this can be safely ignored does not affect any Vitis AI functionality
 - Inconsistent accuracies observed in multi-threaded applications using DPUCADF8H
   - View workaround [here](https://github.com/xilinx/vitis-ai/examples/DPUCADF8H/tf_resnet50_multi_thread/scripts).


### Updates
- v1.3.1
  - Bug fixes and improvements for v1.3
  - Updated Compiler to improve performance by 5% in average for most models
  - Added Zero copy support (new APIs in VART / Vitis AI Library)
  - Added Cross-layer equalization support in TensorFlow v1.15 (more benefits for mobilenet models)
  - Added WAA U50 TRD
  - Updated U280 Pre-Processing using Multi-Preprocessing JPEG decode kernels 
- v1.3.2
  - Enable Ubuntu 20.04 on MPSoC (Vitis AI Runtime and Vitis AI Library)
  - Added environment variable for Vitis AI Library’s model search path

------------------
## Release 1.2

### New Features/Highlights
1. Vitis AI Quantizer and DNNDK runtime all open source
2. 14 new Reference Models  AI Model Zoo (Pytorch, Caffe, Tensorflow)
3. VAI Quantizer supports optimized models (pruned)
4. DPU naming scheme has been updated to be consistent across all configurations
5. Introducing Vitis AI profiler for edge and cloud
6. Added  Alveo U50/U50LV support
7. Added  Alveo U280 support
8. Alveo U50/U50LV DPU DPUCAHX8H micro-architecture improvement
9. DPU TRD upgraded to support Vitis 2020.1 and Vivado 2020.1
11. Vitis AI for Pytorch CNN general access (Beta version)



### Release notes
#### Model Zoo
- 8 new Pytorch models added in the AI Model Zoo (Beta version)
  - ENet, SemanticFPN(ResNet18), facerec_pretrain_res20, face_quality, MT-resnet18, face_reid_large, face_reid_small, person_reid
- Added new Caffe models , including license plate detection and recognition, face detection, medical image segmentation, etc.
- Support pruned model quantization
- Caffe_Dev open source for easier integration
- New Models added for DPUCADX8G on Alveo U200/U250
  - Caffe: Refine-Det, U-Net, Pix2Pix (6 models), Re-identification, Face_Detect (360x640)
  - TF: VGG16, VGG19

#### Quantizer
- Vitis AI for Pytorch CNN general access (Beta version)
- Vitis AI Quantizer open source on Github (Caffe, Tensorflow 1.15 and Pytorch)
- Add Caffe binary and pycaffe support in docker environment (python 2.7)
- Integrated quantization finetuning feature for Caffe and Tensorflow
- Option to specify which layer to be 16-bit

#### Optimizer
- Added support for Tensorflow 1.15
- Added Support weight-shared conv pruning
- Optimizer compatible with docker environment

#### Compiler
- Added support for 14 new models from Xilinx AI Model Zoo
- Added support NNDCT quantized pytorch model compilation
- Improved DPUCAHX8H (for Alveo U50) performance by enabling new IP enhancements and complier optimizations
- Reduced compiler times by 10x for DPUCAHX8H (Alveo U50)
- Optimized compiler memory planning to maximize HBM memory reuse for DPUCAHX8H (Alveo U50)

#### AI Library
- Add new Vitis AI examples, including license plate detection & recognition, face detection, medical image segmentation
- Added support forDPUCADX8G (Alveo U200/U250). Users can build and run the documented models on U200/U250 now.

#### Runtime
- Open sourced DNNDK runtime
- VART adds support for Alveo U50LV, U280
- VART updated to use unified APIs, which explicitly uses XIR, as the unified data structure. All samples are updated to use the new APIs.
- Optimizations for single server, Multi-Card deployments
- Added support for TVM
- Added support for ONNXRuntime

#### AI Profiler
- Added support for C++ function level profiling
- Added support for C++ graph level profiling like DPU graph and CPU graph and sub-graph, etc
- Added support for fined-grain level operator profiling (cloud only supports xmodel)

#### AI Kernel Scheduler
- Introduced in Vitis AI 1.1, AKS is an application to automatically and efficiently pipeline complex AI graphs.
- Vitis AI 1.2 adds support for AI graphs with
  - Whole App Acceleration
  - Multiple FPGAs
  - Python based pre/post processing functions

#### DPU
- DPU naming scheme has been updated to be consistant across all configurations
- Alveo U50 DPU DPUCAHX8H (DPUv3) micro-architecture enhanced to support feature-map stationary, instruction fusion and support long weight instructions, which will result in better data movement efficiency
- Edge DPU DPUCZDX8G (DPUv2) in Vitis 2020.1 adds support for Zynq / Zynq Ultrascale devices and supports low power modes
- Edge DPU DPUCZDX8G (DPUv2) TRD upgraded to be compatible with Vitis 2020.1 and Vivado 2020.1

#### Platforms
- Added Alveo U50LV support
- Added Alveo U280 support
- Improved Alveo U50 performance

#### Examples & Demo
- Whole application acceleration
  - Resnet50 and ADAS examples for ZCU102
- 3 AllenNLP demos on U25 (EA)
- 4-bit DPU demo on ZCU102 (EA)

#### Utilities
- Added Darknet to Caffe Conversion Tool (alveo/apps/yolo/darknet_to_caffe)

- Added scripts to convert darknet yolo networks to caffe and test the accuracy



### Known Issues
1. The model "ssd_pedestrain_pruned_0_97" in pre-compiled model packages has a typo, which should be "ssd_pedestrian_pruned_0.97"
2. Force option "--force" should be used when installing updated packages over the default packages in edge board images
3. The cloud demo cannot support drm for display because of the docker limitations
4. VART does not validate model and DPU combinations. This feature will be added in future releases. users should ensure they have loaded the correct models for the target devices. If not, there will be an unexpected runtime error.
5. The "builddrm.sh" under demo directories in Vitis AI Library can only be cross compiled, and cannot be native build on the board directly

### Updates
- v1.2.1
  - Added Zynq Ultrascale Plus Whole App examples
  - Updated U50 XRT and shell to *Xilinx-u50-gen3x4-xdma-2-202010.1-2902115*
  - Updated docker launch instructions  
  - Updated TRD makefile instructions
- v1.3.2
  - Enable Ubuntu 20.04 on MPSoC (Vitis AI Runtime and Vitis AI Library)
  - Added environment variable for Vitis AI Library’s model search path



------



## Release 1.1

### New Features

#### Model Zoo

* Model quantization accuracy update
* Model test and retraining improved
* Caffe_Xilinx updated to version 1.1
* U50, U200, U250 performance added

#### Quantizer

* Add Tensorflow 1.15 support
* Bugfixes

#### Compiler

* Support cross compilation for Zynq and ZU+ based platforms
* Vitis AI Compiler for U50
    -	Based on the new XIR (Xilinx Intermediate Representation)
    -	Support DPUv3E
    -   Tested with 40 models from Vitis AI Model Zoo
* Vitis AI Compiler for Zynq and ZU+
    -   Support DPUv2 1.4.1 instruction set
    -   Support bias rightward-shift computation to improve model accuracy
    -   Support bilinear upsampling operator

#### Runtime

* VART (Vitis AI Runtime)
    -   Unified runtime based on XIR for Zynq, ZU+ and Alveo
    -   Include new APIs for NN performance improvement
    -   7 samples with VART APIs provided


#### DPU

*  DPUv2 for Zynq and ZU+
*  DPUv2
    -  Upgrade to version 1.4.1
    -  DPU TRD update with Vitis 2019.2 and Vitis AI Library 1.1
*  DPUv3E
    -  https://github.com/Xilinx/Vitis-AI/tree/master/alveo-hbm

#### Vitis AI Library

* All source code open source
* Support VART
* Add support for Alveo
* Support batch model for DPUv3E

#### Example & Demo

* Whole Application Acceleration Example
* End-to-end pipeline which includes JPEG decode, Resize, CNN inference on Alveo
* Neptune demo: Use FPGA for multi-stream and multi-model mode
* AKS demo: Building complex application using C++ and threads

#### Others

* TVM (Early access, provide docker upon request)
    -  Supported frontends: TFLite, ONNX, MxNet and Pytorch
    -  Platform support: ZCU102, ZC104, U200 and U250
    -  Tested for 15 models including classification, detection and segmentation from various frameworks
* xButler upgraded to version 3.0 and provides support for docker container.
* Improved support on upsampling, deconvolution and large convolutions for segmentation models including FPN for DPUv1

### Known Issues
* Alveo U50 toolchain doesn't support Conv2DTranspose trained in Keras and converted to TF 1.15, which will be fixed in Vitis AI 1.2 release.

### Updates
* 5/6/20 - Fixed hardware bug which will lead to computation errors in some corner case for Alveo U50 Production shell xclbin.
* 5/6/20 - Added support for Alveo U50 using EA x4 shell for increased performance.

------



## Release 1.0

### New Features

#### Model Zoo

* Release custom Caffe framework distribution caffe_xilinx
* Add accuracy test code and retrain code for all Caffe models
* Increase Tensorflow models to 19 with float/fixed model versions and accuracy test code, including popular models such as SSD, YOLOv3, MLPerf:ssd_resnet34, etc.
* Add multi-task Caffe model for ADAS applications


#### Optimizer (A separate package which requires licensing)

*  Caffe Pruning
    - Support for depthwise convolution layer
    - Remove internal implementation-related parameters in transformed prototxt
*  TensorFlow Pruning
    -   Release pruning tool based on TensorFlow 1.12
    -   Add more validations to user-specified parameters
    -   Bug fixes for supporting more networks
*  Darknet pruning
    -	new interface for pruning tool
    -	support yolov3-spp

#### Quantizer

*  Tensorflow quantization
    -	Support DPU simulation and dumping quantize simulation results.
    -	Improve support for some layers and node patterns, including tf.keras.layers.Conv2DTranspose, tf.keras.Dense, tf.keras.layers.LeakyReLU, tf.conv2d + tf.mul
    -	Move temp quantize info files from /tmp/ to $output_dir/temp folder, to support multi-users on one machine
    -	Bugfixes

*  Caffe quantization
    -	Enhanced activation data dump function
    -	Ubuntu 18 support
    -	Non-unified bit width quantization support
    -	Support HDF5 data layer
    -	Support of scale layers without parameters but with multiple inputs

#### Compiler
* Support cross compilation for Zynq and ZU+ based platforms
* Enhancements and bug fixes for a broader set of Tensorflow models
* New Split IO memory model enablement for performance optimization
* Improved code generation
* Support Caffe/TensorFlow model compilation over cloud DPU V3E (Early Access)


#### Runtime
* Enable edge to cloud deployment over XRT 2019.2
* Offer the unified Vitis AI C++/Python programming APIs
* DPU priority-based scheduling and DPU core affinity
* Introduce adaptive operating layer to unify runtime’s underlying interface for Linux, XRT and QNX
* QNX RTOS enablement to support automotive customers.
* Neptune API for X+ML
* Performance improvements


#### DPU

*  	DPUv2 for Zynq and ZU+
    -	Support Vitis flow with reference design based on ZCU102
    -	The same DPU also supports Vivado flow
    -	All features are configurable
    -	Fixed several bugs

*  DPUv3 for U50/U280 (Early access)

#### Vitis AI Library

* Support of new Vitis AI Runtime - Vitis AI Library is updated to be based on the new Vitis AI Runtime with unified APIs. It also fully supports XRT 2019.2.
* New DPU support - Besides DPUv2 for Zynq and ZU+, a new AI Library will support new DPUv3 IPs for Alveo/Cloud using same codes (Early access).
* New Tensorflow model support - There are up to 19 tensorflow models supported, which are from official tensorflow repository
* New libraries and demos - There are two new libraries “libdpmultitask” and “libdptfssd” which supports multi-task models and Tensorflow SSD models. An updated classification demo is included to shows how to uses unified APIs in Vitis AI runtime.
* New Open Source Library - The “libdpbase” library is open source in this release, which shows how to use unified APIs in Vitis AI runtime to construct high-level libraries.
* New Installation Method - The host side environment adopts uses image installation, which simplifies and unifies the installation process.


#### Others
* Support for TVM which enables support for Pytorch, ONNX and SageMaker NEO
* Partitioning of Tensorflow models and support for xDNNv3 execution in Tensorflow natively
* Automated Tensorflow model partition, compilation and deployment over DPUv3 (Early access)
* Butler API for following:
    -	Automatic resource discovery and management
    -	Multiprocess support – Ability for many containers/processes to access single FPGA
    -	FPGA slicing – Ability to use part of FPGA
    -	Scaleout support for multiple FPGA on same server
* Support for pix2pix models

### Known Issues
