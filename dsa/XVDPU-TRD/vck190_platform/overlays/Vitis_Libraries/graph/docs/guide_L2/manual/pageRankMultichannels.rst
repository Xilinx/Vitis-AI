.. 
   Copyright 2019 Xilinx, Inc.
  
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
  
       http://www.apache.org/licenses/LICENSE-2.0
  
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. _l2_manual_pagerank_multichannels:

======================
PageRank MultiChannels
======================

PageRank MultiChannels example resides in ``L2/benchmarks/pagerank_multi_channels`` directory. The tutorial provides a step-by-step guide that covers commands for building and running kernel.

Executable Usage
================

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in :ref:`l2_vitis_graph`. For getting the design,

.. code-block:: bash
 
   cd L2/benchmarks/pagerank_multi_channels

* **Build kernel(Step 2)**   

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

.. code-block:: bash

   make run TARGET=hw DEVICE=xilinx_u50_gen3x16_xdma_201920_3

* **Run kernel(Step 3)**

To get the benchmark results, please run the following command.

.. code-block:: bash

   ./build_dir.hw.xilinx_u50_gen3x16_xdma_201920_3/host.exe -xclbin build_dir.hw.xilinx_u50_gen3x16_xdma_201920_3/kernel_pagerank_0.xclbin -dataSetDir data/ -refDir data/ 

PageRank MultiChannels Input Arguments:

.. code-block:: bash

   Usage: host.exe -[-xclbin -dataSetDir -refDir]
         -xclbin:      the kernel name
         -dataSetDir:  the path point to input directory
         -refDir:      the path point to reference directory

Note: Default arguments are set in Makefile, you can use other :ref:`datasets` listed in the table.

* **Example output(Step 4)**

.. code-block:: bash

   Found Platform
   Platform Name: Xilinx
   INFO: Found Device=xilinx_u50_gen3x16_xdma_201920_3
   INFO: Importing build_dir.hw.xilinx_u50_gen3x16_xdma_201920_3/kernel_pagerank_0.xclbin
   Loading: 'build_dir.hw.xilinx_u50_gen3x16_xdma_201920_3/kernel_pagerank_0.xclbin'
   INFO: Kernel has been created
   INFO: Finish kernel setup
   INFO: Finish event_write
   INFO: Finish event_kernel
   ...

   INFO: Finish kernel execution
   INFO: Finish E2E execution
   INFO: Data transfer from host to device: 379 us
   INFO: Data transfer from device to host: 192 us
   INFO: Average kernel execution per run: 174373 us
   INFO: Average execution per run: 174944 us
   ...
   
   INFO: sum_golden = 4.30706
   INFO: sum_pagerank = 4.30706
   INFO: Accurate Rate = 1
   INFO: Err Geomean = 8.74996e-06
   INFO: Result is correct

Profiling
=========

The hardware resource utilizations are listed in the following table.

.. table:: Table 1 Hardware resources for PageRankMultiChannels with 2 working channels
    :align: center

    +-------------------+----------+----------+----------+----------+---------+-----------------+
    |    Kernel         |   BRAM   |   URAM   |    DSP   |    FF    |   LUT   | Frequency(MHz)  |
    +-------------------+----------+----------+----------+----------+---------+-----------------+
    | kernel_pagerank_0 |   303    |   224    |    84    |  352693  |  135193 |      229        |
    +-------------------+----------+----------+----------+----------+---------+-----------------+

Table 2 : Comparison between CPU tigergraph and FPGA VITIS_GRAPH

.. _my-figure-PageRank-5:
.. figure:: /images/PageRankMultiChannels/Performance.png
      :alt: Table 2 : Comparison between CPU tigergraph and FPGA VITIS_GRAPH
      :width: 80%
      :align: center

.. note::
    | 1. Tigergraph time is the execution time of funciton "pageRank" Developer Edition 2.4.1 .
    | 2. Tigergraph running on platform with Intel(R) Xeon(R) CPU E5-2640 v3 @2.600GHz, 32 Threads (16 Core(s)).
    | 3. time unit: second.
    | 4. "-" Indicates that the result could not be obtained due to insufficient memory.
    | 5. FPGA time is the kernel runtime by adding data transfer and executed with pagerank_cache
    | 6. Collected on Alveo u50 platform

.. toctree::
   :maxdepth: 1

