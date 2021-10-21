# Demos of DPU for RNN

The Deep-Learning Processor Unit for Recurrent Neural Network (DPU for RNN) are customized accelerators we built on FPGA and Versal to achieve RNNs acceleration. They support different types of recurrent neural networks, including RNN, Gate Reccurent Unit (GRU), Long-short Term Memory (LSTM), Bi-directional LSTM and lots of their variants. The DPU for RNN have been deployed on Alveo U25, Alveo U50LV and Versal VCK5000 boards. The following table summarizes features of these three accelerators.

|      Feature       | DPURADR16L (U25)           | DPURAHR16L (U50LV)         | DPURVDRML (VCK5000)                           |
|:------------------:|:--------------------------:|:--------------------------:|:---------------------------------------------:|
| Precision          | int16                      | int16                      | Mixed: int8 for GEMM on AIE, int16 for others |
| Operation Type     |Matrix-Vector Multiplication<br>Element-wise Multiplication<br>Addition, Sigmoid, Tanh |Matrix-Vector Multiplication<br>Element-wise Multiplication<br>Addition, Sigmoid, Tanh|Matrix-Matrix Multiplication<br>Element-wise Multiplication<br>Addition, Sigmoid, Tanh, Relu, Max<br>Embedding in RNN-T|
| Multiplication Unit| 1 32x32 Systolic Array |7 16x32 Systolic Array      | 40 AIE Cores                                  |
| Frequency          | Freq\_DSP = 310MHz<br>Freq\_PL = 310MHz     |Freq\_DSP = 540MHz<br>Freq\_PL = 270MHz          | Freq\_AIE = 1.25GHz<br>Freq\_PL = 300MHz                           |
| Example Models     | IMDB\_Sentiment\_Detection<br>Customer\_Satisfaction<br>Open\_Information\_Extraction|IMDB\_Sentiment\_Detection<br>Customer\_Satisfaction<br>Open\_Information\_Extraction |RNN-T                                         |
| Quantization       | NNDCT Tool                  | NNDCT Tool                  | Manually                                |
| Compilation        | Compilver v2                | Compiler v2                 | Manually                                |


#### Alveo U25: DPURADR16L

The Xilinx DPURADR16L IP is a programmable engine optimized for recurrent neural networks, mainly for low latency applications. This IP is implemented on Alveo U25 board with single thread configuration. 

The design is composed of a Scheduler module, Load and Save modules for data movement between the off-chip memory and on-chip caches, a 32x32 systolic array of DSPs to perform Matrix-Vector multiplications and some computation modules for other operations, such as element-wise multiplication and addition, non-linear function, etc. The scheduler is responsible for instructions fetching from the off-chip memory and distributing them to different computation units according to dependency constraints.

The following table shows the resource utilization.

|              | CLB LUTs    | Registers   | Block RAM | URAM    | DSP Slices |
|:------------:|:-----------:|:-----------:|:---------:|:-------:|:----------:|
| Utilization  |187509(35.9%)|303670(29.0%)|659(67.0%) |56(43.8%)|1092(55.5%) |

#### Alveo U50LV: DPURAHR16L

The DPURAHR16L is the design optimized for Alveo U50LV to utilize the high bandwidth of HBMs. The DSP double frequency technique is applied onto the systolic array, thus the 16x32 systolic array on U50LV could achieve similar computation capacity of the U25 version. Total batches of 7 input are supported.

The resource utilization is as the following.

|              | CLB LUTs    | Registers    | Block RAM | URAM     | DSP Slices |
|:------------:|:-----------:|:------------:|:---------:|:--------:|:----------:|
| Utilization  |488679(56.1%)|1045016(60.0%)|796(59.2%) |512(80.0%)|4148(69.7%) |

#### Versal VCK5000: DPURVDRML

The DPURVDRML is a high-performance general RNN processing engine optimized for the Versal AI Core Series. The Versal devices can provide superior performance/watt over conventional FPGAs, CPUs, and GPUs. The DPURVDRML comprises of AI Engines and PL.

In this demo design, the GEMM operations with precision of int8 are deployed onto the 5x8 AIE array. Each AI core performs matrix-matrix multiplication of size 32x64x32. The 40 cores kernel calculates GEMM of size 32x320x256. The output of GEMM is at the precision of int16. The Misc part is composed of different modules to support different types of operations, including element-wise multiplication, addition, sigmoid, tanh, max, etc. Intermediate data should be represented with the precision of int16.

The following table shows the hardware resource utilization.

|           | CLB LUTs | Registers | Block RAM | URAM | DSP Slices | AIE Cores |
|:---------:|:--------:|:---------:|:---------:|:----:|:----------:|:---------:|
| Utilized  |169163(18.8%)|241657(13.43%)|197(20.37%)|332(71.71%)|82(4.17%)|40(10%)|

#### Register Space for DPU-for-RNN.

|      Register     | Address<br>(32-bit) | Description |
|:-------------------|:------------------:|:-------------|
| AP\_CTRL           | 0x0     | AP\_CTRL[0]: ap\_start. 1-busy, 0-idle. <br>AP\_CTRL[1]: ap\_done. 1-done, clear on first read.<br>AP\_CTRL[2]: ap\_idle. 1-idle, 0-busy.<br>AP\_CTRL[3]: ap\_ready. 1-ready, 0-busy. |
| SOFT\_RESET        | 0x14    | Reset the DPU, high level activate |
| FRAME\_LEN         | 0x18    | The sequence length of the input sample |
| INSTR\_BASE\_ADDR\_H | 0x1C    | The higher 32bits of instructions address in DDR. Shared with model parameter address register |
| INSTR\_BASE\_ADDR\_L | 0x20    | The lower 32bits of instructions address in DDR |
| MODEL\_BASE\_ADDR\_L | 0x24    | The lower 32bits of model parameters address in DDR |
| INPUT\_BASE\_ADDR\_H | 0x28    | The higher 32bits of input address in DDR. Shared with output address register |
| INPUT\_BASE\_ADDR\_L | 0x2C    | The lower 32bits of input address in DDR |
| INPUT\_BATCH\_STRIDE| 0x64    | Only used in DPURVDRML. The stride of batches for the input in DDR memory |
| OUTPUT\_BASE\_ADDR\_L| 0x30    | The lower 32bits of output address in DDR |
| OUTPUT\_BATCH\_STRIDE| 0x68   | Only used in DPURVDRML. The stride of batches for the output in DDR memory |

## Install Platform and XRT


Please install the Platform and XRT.

Alveo U25 Platform
```
xilinx-cmc-u25_2.1.3-3021321
xilinx-u25-gen3x8-xdma-validate_1-2954712
xilinx-u25-gen3x8-xdma-base_1-3045185
```

Alveo U50LV Platform
```
xilinx-u50lv-gen3x4-xdma-base-2-2902115
xilinx-u50lv-gen3x4-xdma-validate-2-2902115
xilinx-cmc-u50-1.0.20-2853996
xilinx-sc-fw-u50-5.0.27-2.e289be9
```

[VCK5000 Platform](https://www.xilinx.com/products/boards-and-kits/vck5000.html#getStarted)
```
xilinx-vck5000-es1-gen3x16-base-2-3123623
xilinx-sc-fw-vck5000-4.4.6-2.e1f5e26
xilinx-vck5000-es1-gen3x16-validate-2-3123623
```

XRT
```
xrt-2.11.648-1.x86_64
```
## RNN-T demo dependency setup

### Build RNN Docker from Recipe

The Vitis AI RNN Docker container needs to be built from recipe. Before building the Vitis AI RNN Docker, please follow the instructinos at https://gitenterprise.xilinx.com/Vitis/vitis-ai-staging/tree/1.4.1#building-docker-from-recipe and build the Vitis AI GPU Docker using ./docker_build_gpu.sh

The Vitis AI RNN Docker recipe depends on the xilinx/vitis-ai-gpu:latest Docker image as a base image and will not build without it.

When the  xilinx/vitis-ai-gpu:latest  docker image is available, proceed to build the Vitis AI RNN Docker image
```
cd setup
sh docker_build_rnnt.sh
```

## Run Demos of DPU-for-RNN.

Please refer to the following links to run the demo in the docker image according to instructions.

- [Run RNN demos of Alveo U25 and U50LV](rnn_u25_u50lv/apps)
- [Run the RNN-T demo on Versal VCK5000](rnnt_asr_vck5000)

## License
Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
```
http://www.apache.org/licenses/LICENSE-2.0
```
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.




