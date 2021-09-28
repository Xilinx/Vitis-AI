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
   :keywords: GTSV, Alveo, Lapack, Jacobi, matrix
   :description: The hardware resources and performance for Tridiagonal Linear Solver (GTSV).
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


.. _guide_l2_benchmark_gtsv:


==========================================================
Tridiagonal Linear Solver (GTSV)
==========================================================

GTSV example resides in ``L2/benchmarks/gtsv`` directory. The tutorial provides a step-by-step guide that covers commands for building and running kernel.

Executable Usage
================

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in :ref:`l2_vitis_solver`. For getting the design,

.. code-block:: bash

   cd L2/benchmarks/gtsv

* **Build kernel(Step 2)**

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

.. code-block:: bash

   source /opt/xilinx/Vitis/2021.1/settings64.sh
   source /opt/xilinx/xrt/setenv.sh
   export DEVICE=/opt/xilinx/platforms/xilinx_u250_xdma_201830_2/xilinx_u250_xdma_201830_2.xpfm
   export TARGET=hw
   make run 

* **Run kernel(Step 3)**

To get the benchmark results, please run the following command.

.. code-block:: bash

   ./build_dir.hw.xilinx_u250_xdma_201830_2/test_gtsv.exe -xclbin build_dir.hw.xilinx_u250_xdma_201830_2/kernel_gtsv.xclbin -runs 1 -M 1024

GTSV Input Arguments:

.. code-block:: bash

   Usage: test_gtsv.exe -[-xclbin -o -c -g]
          -xclbin     gtsv binary;
          -runs       number of runs; 
          -M          size of input Matrix row/cloumn; 

Note: Default arguments are set in Makefile. The default configs are: -runs 1 -M 16.

* **Example output(Step 4)** 

.. code-block:: bash
   
   ---------------------GTSV Test----------------
   Found Platform
   Platform Name: Xilinx
   INFO: Found Device=xilinx_u250_xdma_201830_2
   INFO: Importing build_dir.hw.xilinx_u250_xdma_201830_2/wcc_kernel.xclbin
   Loading: 'build_dir.hw.xilinx_u250_xdma_201830_2/wcc_kernel.xclbin'
   INFO: kernel has been created
   INFO: kernel start------
   INFO: kernel end------
   INFO: Execution time 53.697ms
   INFO: Write DDR Execution time 0.11773ms
   INFO: Kernel Execution time 53.198ms
   INFO: Read DDR Execution time 0.049562ms
   INFO: Total Execution time 53.3653ms
   ============================================================

Profiling 
=========

The GTSV is validated on Xilinx Alveo U250 board. 

The hardware resources and performance for double datatype is listed in :numref:`table_gtsvDouble`.
To describe the resource utilization, we separate the overall utilization into two parts, P stands for the resource usage in platform, that is those instantiated in static region of the FPGA card, K stands for those used in kernels(dynamic region).  
The Unroll factor means how many CUs are configured to calculate Matrix in parell.

.. _tabgtsvDouble:

.. table:: double Type GTSV performance table
    :align: center

    +-------------+--------+----------------+--------------+-------------+-------------+-----------+-----------+----------+
    | Matrix Size | Unroll | Frequency(MHz) | Latency(MHz) |     LUT     |     REG     |  BRAM     |    URAM   |  DSP     |
    +-------------+--------+----------------+--------------+-------------+-------------+-----------+-----------+----------+
    |  1024x1024  |   16   |     300        |              |  146609(P)  |  225383(P)  |   283(P)  |     0(P)  |    7(P)  |
    |             |        |                |              +-------------+-------------+-----------+-----------+----------+
    |             |        |                |              |  112880(K)  |  118118(K)  |   144(K)  |     0(K)  |  112(K)  |
    +-------------+--------+----------------+--------------+-------------+-------------+-----------+-----------+----------+



.. note:: 
    The unroll factor is limited by 2 factors, the matrix size and URAM port. The maximum unroll factor should be less than half of matrix size, and :math:`2 \times {Unroll}^{2}` should also be less than available URAM on board. Besides, unroll factor can only be the factorization of 2.


    +-------------+--------+------+------+-----+----------+--------+------------------+-----------------+
    | Matrix Size | Unroll | URAM | BRAM | DSP | Register |  LUT   | Kernel time (us) | Frequency (MHz) |
    +-------------+--------+------+------+-----+----------+--------+------------------+-----------------+
    |  1024x1024  |   16   |  128 |  16  | 960 |  260297  | 223889 |      16.6        |      291        |
    +-------------+--------+------+------+-----+----------+--------+------------------+-----------------+

.. toctree::
   :maxdepth: 1
