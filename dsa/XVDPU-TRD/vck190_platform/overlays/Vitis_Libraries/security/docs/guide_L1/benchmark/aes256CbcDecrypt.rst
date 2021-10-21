.. 
   Copyright 2020 Xilinx, Inc.
  
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
  
       http://www.apache.org/licenses/LICENSE-2.0
  
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.


.. meta::
   :keywords: aes256CbcDecrypt
   :description: The hardware resources and performance for aes256CbcDecrypt
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


.. _guide_l1_benchmark_aes256CbcDecrypt:


=================
aes256CbcDecrypt
=================

To profile performance of aes256CbcDecrypt, we prepare a datapack of 32K messages, each message is 1Kbyte.
We have 1 kernels, each kernel has 4 PUs.
Kernel utilization and throughput is shown in table below.

Executable Usage
================

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in :ref:`l1_vitis_security`. For getting the design,

.. code-block:: bash

   cd L1/benchmarks/aes256CbcDecrypt

* **Build kernel(Step 2)**

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

.. code-block:: bash

   source /opt/xilinx/Vitis/2021.1/settings64.sh
   source /opt/xilinx/xrt/setenv.sh
   export DEVICE=u50_gen3x16
   export TARGET=hw
   make run 

* **Run kernel(Step 3)**

To get the benchmark results, please run the following command.

.. code-block:: bash

   ./BUILD_DIR/host.exe -xclbin ./BUILD_DIR/aes256CbcDecryptKernel.xclbin

Input Arguments:

.. code-block:: bash

   Usage: host.exe -[-xclbin]
          -xclbin     binary;

* **Example output(Step 4)**

.. code-block::  bash

   Found Platform
   Platform Name: Xilinx
   Selected Device xilinx_u250_gen3x16_xdma_3_1_202020_1
   INFO: Importing build_dir.sw_emu.xilinx_u250_gen3x16_xdma_3_1_202020_1/aes256CbcDecryptKernel.xclbin
   Loading: 'build_dir.sw_emu.xilinx_u250_gen3x16_xdma_3_1_202020_1/aes256CbcDecryptKernel.xclbin'
   Kernel has been created.
   allocate to DDR
   DDR buffers have been mapped/copy-and-mapped
   4 channels, 2 tasks, 64 messages verified. No error found!
   Kernel has been run for 2 times.
   Total execution time 1413103us


Profiling 
=========

The aes256CbcDecrypt is validated on Xilinx Alveo U250 board. 
Its resource, frequency and throughput is shown as below.

+-----------+------------+------------+---------+----------+-------+--------------+
|Frequency  |     LUT    |     REG    |   BRAM  |   URAM   |  DSP  |  Throughput  |
+-----------+------------+------------+---------+----------+-------+--------------+
| 286MHz    | 203,595    |  312,900   |  761    |    0     |  29   | 4.7GB/s      |
+-----------+------------+------------+---------+----------+-------+--------------+


.. toctree::
   :maxdepth: 1
