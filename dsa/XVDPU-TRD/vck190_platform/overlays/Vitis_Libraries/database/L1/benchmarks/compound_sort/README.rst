.. 
   Copyright 2021 Xilinx, Inc.
  
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
  
       http://www.apache.org/licenses/LICENSE-2.0
  
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.


.. _l1_componentsort:

=============
Compound Sort
=============

Compound Sort example resides in ``L1/benchmarks/compound_sort`` directory.

This benchmark tests the performance of `compoundSort` primitive with an array of integer keys. This primitive is named as compound sort, as it combines `insertSort` and `mergeSort`, to balance storage and compute resource usage. 

The tutorial provides a step-by-step guide that covers commands for building and running kernel.

Executable Usage
================

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in :ref:`l2_vitis_database`. For getting the design,

.. code-block:: bash

   cd L1/benchmarks/compound_sort

* **Build kernel(Step 2)**

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

.. code-block:: bash

   make run TARGET=hw DEVICE=xilinx_u280_xdma_201920_3

* **Run kernel(Step 3)**

To get the benchmark results, please run the following command.

.. code-block:: bash

   ./build_dir.hw.xilinx_u280_xdma_201920_3/host.exe -xclbin build_dir.hw.xilinx_u280_xdma_201920_3/SortKernel.xclbin 

Compound Sort Input Arguments:

.. code-block:: bash

   Usage: host.exe -xclbin
          -xclbin     compound sort binary

Note: Default arguments are set in Makefile, you can use other platforms to build and run.

* **Example output(Step 4)** 

.. code-block:: bash
   
   -----------Sort Design---------------
   key length is 131072
   [INFO]Running in hw mode
   Found Platform
   Platform Name: Xilinx
   Found Device=xilinx_u280_xdma_201920_3
   INFO: Importing build_dir.hw.xilinx_u280_xdma_201920_3/SortKernel.xclbin
   Loading: 'build_dir.hw.xilinx_u280_xdma_201920_3/SortKernel.xclbin'
   kernel has been created
   kernel start------
   PASS!
   Write DDR Execution time 127.131us
   Kernel Execution time 1129.78us
   Read DDR Execution time 83.459us
   Total Execution time 1340.37us
   ------------------------------------------------------------

Profiling 
=========

The compound sort design is validated on Alveo U280 board at 287 MHz frequency. 
The hardware resource utilizations are listed in the following table.

.. table:: Table 1 Hardware resources for compound sort
    :align: center

    +------------+--------------+-----------+----------+--------+
    |    Name    |      LUT     |    BRAM   |   URAM   |   DSP  |
    +------------+--------------+-----------+----------+--------+
    | Platform   |    142039    |    285    |    0     |    7   |
    +------------+--------------+-----------+----------+--------+
    | SortKernel |    62685     |    18     |    16    |    0   |
    +------------+--------------+-----------+----------+--------+
    | User Budget|   1160681    |   1731    |    960   |   9017 |
    +------------+--------------+-----------+----------+--------+
    | Percentage |    5.40%     |   1.04%   |   1.67%  |    0   |
    +------------+--------------+-----------+----------+--------+


The performance is shown below.
   This design takes 1.130ms to process 0.5MB data, so it achieves 442.56MB/s throughput.


.. toctree::
    :maxdepth: 1
