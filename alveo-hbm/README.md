# DPUv3E for Alveo Accelerator Card with HBM

DPUv3E is a member of Xilinx DPU IP family for convolution nerual network (CNN) inference application. It is designed for latest Xilinx Alveo U50/U280 adaptable accelerator cards with HBM support. DPU V3E is a high performance CNN inference IP optimized for throughput and data center workloads. DPUv3E runs with highly optimized instructions set and supports all mainstream convolutional neural networks, such as VGG, ResNet, GoogLeNet, YOLO, SSD, FPN, etc. 

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

---
The off-the-shell DPUv3E solution for U50 includes two DPUv3E kernels, which can be used with Vitis AI VART or Vitis AI Library easily. Please refer to the relevant parts for usages of DPUv3E on U50 with [VART](../VART/README.md) and [Vitis-AI-Library](../Vitis-AI-Library/README.md) (you could search the keyword "for Cloud").
