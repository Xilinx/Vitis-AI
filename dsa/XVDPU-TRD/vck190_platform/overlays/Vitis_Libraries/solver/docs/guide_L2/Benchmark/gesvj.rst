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
   :keywords: GESVJ, Alveo, Lapack, Jacobi, matrix
   :description: The hardware resources and performance for double datatype general matrix (GESVJ).
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


.. _guide_l2_benchmark_gesvj:


=========================================================
Singular Value Decomposition for general matrix (GESVJ)
=========================================================

GESVJ example resides in ``L2/benchmarks/gesvj`` directory. The tutorial provides a step-by-step guide that covers commands for building and running kernel.

Executable Usage
================

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in :ref:`l2_vitis_solver`. For getting the design,

.. code-block:: bash

   cd L2/benchmarks/gesvj

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

   ./build_dir.hw.xilinx_u250_xdma_201830_2/test_gesvj.exe -xclbin build_dir.hw.xilinx_u250_xdma_201830_2/kernel_gesvj.xclbin -runs 1 -M 512 -N 512 -seed 12 

GESVDJ Input Arguments:

.. code-block:: bash

   Usage: test_gesvj.exe -[-xclbin -o -c -g]
          -xclbin     gesvj binary;
          -runs       number of runs; 
          -M          size of input Matrix row; 
          -N          size of input Matrix column;
          -seed       seed for generating a random number;

Note: Default arguments are set in Makefile. The default configs are: -runs 1 -M 4 -N 3 -seed 12.

* **Example output(Step 4)** 

.. code-block:: bash
   
   ---------------------GESVJ Test----------------
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

The GESVJ is validated on Xilinx Alveo U250 board. 
The accuracy of GESVDJ implementation has been verified with Lapack dgesvd (QR based SVD) and dgesvj (Jacobi SVD) functions. 
The hardware resources and performance for double datatype gesvj is listed in :numref:`tabgesvjDouble`.
To describe the resource utilization, we separate the overall utilization into two parts, P stands for the resource usage in platform, that is those instantiated in static region of the FPGA card, K stands for those used in kernels(dynamic region).  
The Unroll factor means how many CUs are configured to calculate Matrix in parell.

.. _tabgesvjDouble:

.. table:: double Type GESVJ performance table
    :align: center

    +-------------+--------+----------------+--------------+-------------+-------------+-----------+-----------+----------+
    | Matrix Size | Unroll | Frequency(MHz) | Latency(MHz) |     LUT     |     REG     |  BRAM     |  URAM     |  DSP     |
    +-------------+--------+----------------+--------------+-------------+-------------+-----------+-----------+----------+
    |  512x512    |    16  |     248.4      |              |  146836(P)  |  225255(P)  |   283(P)  |     0(P)  |     7(P) |
    |             |        |                |              +-------------+-------------+-----------+-----------+----------+
    |             |        |                |              |  137271(K)  |  180145(K)  |   123(K)  |   192(K)  |  1801(K) |
    +-------------+--------+----------------+--------------+-------------+-------------+-----------+-----------+----------+


.. note:: 
    The unroll factor is limited by 2 factors, the matrix size and URAM port. The maximum unroll factor should be less than half of matrix size, and :math:`2 \times {Unroll}^{2}` should also be less than available URAM on board. Besides, unroll factor can only be the factorization of 2.

    +-------------+--------+------+------+------+----------+--------+---------------------+-----------------+
    | Matrix Size | Unroll | URAM | BRAM | DSP  | Register |  LUT   | Kernel time (ms)    | Frequency (MHz) |
    +-------------+--------+------+------+------+----------+--------+---------------------+-----------------+
    |    64x64    |    2   |  55  |  27  | 282  |   81753  | 73895  |         27.8        |    300          |
    +-------------+--------+------+------+------+----------+--------+---------------------+-----------------+
    |   512x512   |    4   |  192 |  41  | 500  |   98763  | 92207  |         4827        |    230          |
    +-------------+--------+------+------+------+----------+--------+---------------------+-----------------+
    |   512x512   |   16   |  192 |  125 | 1808 |  203666  | 165800 |        4686.5       |    249          |
    +-------------+--------+------+------+------+----------+--------+---------------------+-----------------+


.. toctree::
   :maxdepth: 1
