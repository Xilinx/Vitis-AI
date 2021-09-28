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

.. meta::
   :keywords: BLAS, Library, Vitis BLAS Library, L2, level 2
   :description: Vitis BLAS library level 2 application programming interface reference. Intel Math Kernel Library provides performance improvement of math functions, e.g. GEMM, when running with Intel processors.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


.. _benchmark_gemv_l2:

***********************
L2 GEMV benchmark
***********************

1. gemvStreamCh16
=====================

This example resides in ``L2/benchmarks/streamingKernel/gemvStreamCh16`` directory. The tutorial provides a step-by-step guide that covers commands for building and running kernel. It performs the matrix-vecotr multiplication, M is number of rows of matrix, N is number of columns of matrix.

1.1 Executable Usage
------------------------

1.1.1 Work Directory(Step 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The steps for library download and environment setup can be found in [here](https://github.com/Xilinx/Vitis_Libraries/tree/master/blas/L2/benchmarks#building). For getting the design,

.. code-block:: bash 

    cd L2/benchmarks/streamingKernel/gemvStreamCh16


1.1.2 Build kernel(Step 2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

.. code-block:: bash 

    make run TARGET=hw PLATFORM_REPO_PATHS=/opt/xilinx/platforms DEVICE=xilinx_u280_xdma_201920_1


1.1.3 Run kernel(Step 3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To get the benchmark results, please run the following command.

gemvStreamCh16 Input Arguments:

.. code-block:: bash 

    <host application> <xclbin> <m> <n> <path_to_data> device_id
    

For example:

.. code-block:: bash 

    build_dir.hw.xilinx_u280_xdma_201920_1/host.exe build_dir.hw.xilinx_u280_xdma_201920_1/gemv.xclbin 512 256 build_dir.hw.xilinx_u280_xdma_201920_1/data/ 0


1.1.4 Example output(Step 4)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash 

    Found Platform
    Platform Name: Xilinx
    INFO: Importing gemv.xclbin
    Loading: 'gemv.xclbin'
    Software-measured execution time 0.000292705s.
    Software-measured HW efficiency 2.09904%.
    Execution clock cycles is: 4759
    Efficiency is: 43.0343%.
    Results verified.


1.2 Profiling for u280
-------------------------

The xclbin could be built in 319 MHz
The hardware resource utilization and benchmark results are shown in the two tables below.

*Table 1 Hardware resources*

+---------------------+-------------------+------------------+-------------------+----------------+---------------+----------------+
| Name                | LUT               | LUTAsMem         | REG               | BRAM           | URAM          | DSP            |
+=====================+===================+==================+===================+================+===============+================+
| krnl_gemv           |  122248 [ 10.48%] |  11010 [  1.90%] |  215381 [  9.02%] |   72 [  3.97%] |   0 [  0.00%] |  966 [ 10.71%] |
| streamTimer         |     195 [  0.02%] |      0 [  0.00%] |     291 [  0.01%] |    0 [  0.00%] |   0 [  0.00%] |    0 [  0.00%] |
+---------------------+-------------------+------------------+-------------------+----------------+---------------+----------------+

*Table 2 Benchmark results* 

+-------+-------+---------------------------+-------------------------+-----------------+
|  M    |  N    | Kernel execution time [s] | api execution time [s]  |  efficiency [%] |
+=======+=======+===========================+=========================+=================+
| 512   | 256   | 1.4316e-05                | 0.00330468              | 42.9173         |
+-------+-------+---------------------------+-------------------------+-----------------+
| 512   | 512   | 1.9998e-05                | 0.00337302              | 61.4461         |
+-------+-------+---------------------------+-------------------------+-----------------+
| 1024  | 1024  | 6.5904e-05                | 0.0035207               | 74.5812         |
+-------+-------+---------------------------+-------------------------+-----------------+
| 2048  | 2048  | 0.000235251               | 0.00365028              | 83.5737         |
+-------+-------+---------------------------+-------------------------+-----------------+
| 4096  | 4096  | 0.000939699               | 0.00452506              | 83.6898         |
+-------+-------+---------------------------+-------------------------+-----------------+
| 8192  | 8192  | 0.00332612                | 0.0105467               | 94.5764         |
+-------+-------+---------------------------+-------------------------+-----------------+


1.3 Profiling for u50
-----------------------

The xclbin could be built in 333 MHz
The hardware resource utilization and benchmark results are shown in the two tables below.

*Table 1 Hardware resources*

+---------------------+------------------+------------------+-------------------+----------------+---------------+----------------+
| Name                | LUT              | LUTAsMem         | REG               | BRAM           | URAM          | DSP            |
+=====================+==================+==================+===================+================+===============+================+
| krnl_gemv           | 121535 [ 16.26%] |  11002 [  2.85%] |  215897 [ 13.72%] |   72 [  6.19%] |   0 [  0.00%] |  966 [ 16.27%] |
+---------------------+------------------+------------------+-------------------+----------------+---------------+----------------+
| streamTimer         |    195 [  0.03%] |      0 [  0.00%] |     291 [  0.02%] |    0 [  0.00%] |   0 [  0.00%] |    0 [  0.00%] |
+---------------------+------------------+------------------+-------------------+----------------+---------------+----------------+

*Table 2 Benchmark results* 

+-------+-------+-----------------------+------------------------------+----------------------------+--------------------------+--------------+
|  M    |  N    | hw execution time (s) | cold api execution time (s)  | hot api execution time (s) |  execution clock cycles  |  efficiency  |
+=======+=======+=======================+==============================+============================+==========================+==============+
| 512   | 256   | 1.4481e-05            | 0.000241345                  | 0.00014245                 | 4827                     | 42.428%      |
+-------+-------+-----------------------+------------------------------+----------------------------+--------------------------+--------------+
| 512   | 512   | 2.0853e-05            | 0.000428344                  | 0.000136975                | 6951                     | 58.9268%     |
+-------+-------+-----------------------+------------------------------+----------------------------+--------------------------+--------------+
| 1024  | 1024  | 6.6462e-05            | 0.000439357                  | 0.00017869                 | 22154                    | 73.955%      |
+-------+-------+-----------------------+------------------------------+----------------------------+--------------------------+--------------+
| 2048  | 2048  | 0.000248076           | 0.000637851                  | 0.000367888                | 82692                    | 79.2531%     |
+-------+-------+-----------------------+------------------------------+----------------------------+--------------------------+--------------+
| 4096  | 4096  | 0.000898929           | 0.00156095                   | 0.00101729                 | 299643                   | 87.4854%     |
+-------+-------+-----------------------+------------------------------+----------------------------+--------------------------+--------------+
| 8192  | 8192  | 0.00332855            | 0.00478017                   | 0.00365307                 | 1109516                  | 94.5075%     |
+-------+-------+-----------------------+------------------------------+----------------------------+--------------------------+--------------+

