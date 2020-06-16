# DPUv3E for Alveo Accelerator Card with HBM

DPUv3E is a member of Xilinx DPU IP family for convolution nerual network (CNN) inference application. It is designed for latest Xilinx Alveo U50/U50LV/U280 adaptable accelerator cards with HBM support. The U50 version has two DPUv3E kernels, totally six engines; the U50LV version has two DPUv3E kernels, totally nine or ten engines; the U280 version has three DPUv3E kernels, totally fourteen engines.

DPUv3E is released with Vitis AI. Please refer to the relevant parts for usages of DPUv3E on U50 with [VART](../VART/README.md) and [Vitis-AI-Library](../Vitis-AI-Library/README.md) (you could search the keyword "for Cloud").

## Alveo Card Setup
For U280 card, DPUv3E use the standard target platform released in the Xilinx website, please follow the instruction in the [U280 page](https://www.xilinx.com/products/boards-and-kits/alveo/u280) to get the required files. 

For U50 and U50LV card, DPUv3E use the gen3x4 version target platform, instead of the standard gen3x16 platform. Please download required gen3x4 target platform files from below URL:

### For Redhat/CentOS 7.4-7.7 Host
Common Files
* xilinx-cmc-u50-1.0.20-2853996.noarch.rpm
* xilinx-sc-fw-u50-5.0.27-2.e289be9.noarch.rpm

Alveo U50
* xilinx-u50-gen3x4-xdma-validate-2-2889074.noarch.rpm
* xilinx-u50-gen3x4-xdma-base-2-2895184.noarch.rpm

Alveo U50LV
* xilinx-u50lv-gen3x4-xdma-validate-2-2900293.noarch
* xilinx-u50lv-gen3x4-xdma-base-2-2895310.noarch

### For Ubuntu 16.04 Host
Common Files
* xilinx-cmc-u50_1.0.20-2853996_all_16.04
* xilinx-sc-fw-u50-5.0.27-2.e289be9_16.04

Alveo U50
* xilinx-u50-gen3x4-xdma-validate_2-2889074_all_16.04.deb
* xilinx-u50-gen3x4-xdma-base_2-2895184_all_16.04.deb

Alveo U50LV
* xilinx-u50lv-gen3x4-xdma-validate_2-2900293_all_16.04.deb
* xilinx-u50lv-gen3x4-xdma-base_2-2895310_all_16.04.deb


### for Ubuntu 18.04 host
Common Files
* xilinx-cmc-u50_1.0.20-2853996_all_18.04.deb
* xilinx-sc-fw-u50-5.0.27-2.e289be9_18.04.deb

Alveo U50
* xilinx-u50-gen3x4-xdma-validate_2-2889074_all_18.04.deb
* xilinx-u50-gen3x4-xdma-base_2-2895184_all_18.04.deb

Alveo U50LV
* xilinx-u50lv-gen3x4-xdma-validate_2-2900293_all_18.04.deb
* xilinx-u50lv-gen3x4-xdma-base_2-2895310_all_18.04.deb

### Update the card flash
After you have downloaded and installed the platform files above, use following commands and cold reboot your machine to finished the setup.

For Alveo U50:
~~~
sudo /opt/xilinx/xrt/bin/xbmgmt flash --update --shell xilinx_u50_gen3x4_xdma_base_2
~~~

For Alveo U50LV:
~~~
sudo /opt/xilinx/xrt/bin/xbmgmt flash --update --shell xilinx_u50lv_gen3x4_xdma_base_2
~~~

---

## DPUv3E Overlays Setup

Four kinds of DPUv3E overlays are provided for Alveo HBM card:
* U50-6E300M: two kernels, six engines, run at maximal 300MHz
* U50LV-9E275M: two kernels, nine engines, run at maximal 275MHz
* U50LV-10E275M: two kernels, ten engines, run at maximal 275MHz
* U280-14E300M: three kernels, fourteen engines, run at maximal 300MHz

Please the maximal running frequency is the timing sign-off frequency of each overlays. Because of the power limitation of the card, all CNN models on each Alveo card cannot run at all the frequencies listed above. Sometimes frequency scaling-down operation is needed. For the safe working frequency on each card for the CNN models and corresponding performance, please refer to Chapter 7 of *Vitis AI Library User Guide* (ug1354).

### Get and Decompress Overlays Tarball
In the host or docker, get to the shared Vitis AI git repository directory and use following commands to download and decompress the overlays tarball. (the download link is not available yet for BASH stage)
~~~
cd ./Vitis-AI/alveo-hbm
wget https://www.xilinx.com/bin/public/openDownload?filename=alveo_xclbin-1.2.0.tar.gz -O alveo_xclbin-1.2.0.tar.gz
tar xfz alveo_xclbin-1.2.0.tar.gz
~~~

### Settle Down the Overlays
Start the docker, get into the shared Vitis AI git repository directory and use following command to settle down the overlay files for different Alveo card. Please note everytime you start a new docker container, you should do this step.

For Alveo U50, use U50-6E300M overlay:
~~~
cd ./Vitis-AI/alveo-hbm
sudo cp alveo_xclbin-1.2.0/U50/6E300M/* /usr/lib
~~~

For Alveo U50LV, use U50LV-9E275M overlay:
~~~
cd ./Vitis-AI/alveo-hbm
sudo cp alveo_xclbin-1.2.0//U50lv/9E275M/* /usr/lib
~~~

For Alveo U50LV, use U50LV-10E275M overlay:
~~~
cd ./Vitis-AI/alveo-hbm
sudo cp alveo_xclbin-1.2.0//U50lv/10E275M/* /usr/lib
~~~

For Alveo U280, use U280-14E300M overlay:
~~~
cd ./Vitis-AI/alveo-hbm
sudo cp alveo_xclbin-1.2.0//U280/14E300M/* /usr/lib
~~~

### Overlays Frequency Scaling Down
You could use XRT xbutil tools to scale down the running frequency of the DPUv3E overlay before you run any VART/Library examples. 

**Higher overlay frequencies then the recommendation in ug1354 could cause system reboot or other damage to your system because of the power consumption exceeding of Alveo card over the PCIe power supply limitation.**

To scale down the overlay frequency:
~~~
/opt/xilinx/xrt/bin/xbutil clock -dx -g XXX
~~~
dx is the Alveo card number if more than one Alveo card exist in your system. If only one card is installed, you should use *d0*. XXX is the target frequency value, such as 220. For example, following command will set the default U50 card DPUv3E overlay frequency to 200MHz:
~~~
/opt/xilinx/xrt/bin/xbutil clock -d0 -g 200
~~~

---

## Introduction to DPUv3E

DPU V3E is a high performance CNN inference IP optimized for throughput and data center workloads. DPUv3E runs with highly optimized instructions set and supports all mainstream convolutional neural networks, such as VGG, ResNet, GoogLeNet, YOLO, SSD, FPN, etc. 

DPUv3E is one of the fundamental IPs (Overlays) of Xilinx Vitis™ AI development environment, and the user can use Vitis AI toolchain to finish the full stack ML development with DPUv3E. The user can also use standard Vitis flow to finish the integration of DPUv3E with other customized acceleration kernal to realize powerful X+ML solution. DPUv3E is provided as encrypted RTL or XO file format for Vivado or Vitis based integration flow.

The major supported Neural Network operators include:

- Convolution / Deconvolution
- Max pooling / Average pooling
- ReLU, ReLU6, and Leaky ReLU
- Concat
- Elementwise-sum
- Dilation
- Reorg
- Fully connected layer
- Batch Normalization
- Split

DPUv3E is highly configurable, a DPUv3E kernel consists of several Batch Engines, a Instruction Scheduler, a Shared Weights Buffer,  and a Control Register Bank. Following is the block diagram of a DPUv3E kernel including 5 Batch Engines.

<img src = "./images/DPUv3E Kernel Diagram.png" align = "center">

### Batch Engine
Batch Engine is the core computation unit of DPUv3E. A Batch Engine can handle an input image at a time, so multiple Batch Engines in a DPUv3E kenel can process sevel input images simultaneously. The number of Batah Engine in a DPUv3E kernel can be configured based on FPGA resource condition and customer's  performance requirement. For example, in Alveo U280 card, SLR0 (with direct HBM connection) can contain a DPUv3E kernel with maximal four Batch Engines while SLR1 or 2 can contain a DPUv3E kernel with five Batch Engines. In Batch Engine, there is a convolution engine to handle regular convolution/deconvolution compution, and a MISC engine to handle pooling, ReLu, and other miscellaneous operations. MISC engine is also configurable for optional function according specific nerual network requirement. Each Batch Engine use a AXI read/write master interfaces for feature map data exchange between device memory (HBM).

### Instruction Scheduler
Similar to general purpose processor in concept, Instruction Scheduler carries out instruction fetch, decode and dispatch jobs. Since all the Batch Engines in a DPUv3E kernel will run the same nerual network, so Instruction Shceduler serves all the Batch Engines with the same instruction steam. The instruction stream is loaded by host CPU to device memory (HBM) via PCIe interface, and Instruction Scheduler use a AXI read master interface to fetch DPU instruction for Batch Engine.

### Shared Weight Buffer
Shared Weight Buffer includes complex strategy and control logic to manage the loading of nerual network weight from Alveo device memory and transfering them to Batch Engines effeciently. Since all the Batch Engines in a DPUv3E kernel will run the same nerual network, so the weights data are wisely loaded into on-chip buffer and shared by all the Batch Engines to eleminate unnecessary memory access to save bandwidth. Shared Weight Buffer use two AXI read master interfaces to load Weight data from device memory (HBM).

### Control Register Bank
Control Register Bank is the control interface between DPUv3E kernel and host CPU. It implements a set of controler register compliant to Vitis development flow. Control Register Bank has a AXI slave interface.
