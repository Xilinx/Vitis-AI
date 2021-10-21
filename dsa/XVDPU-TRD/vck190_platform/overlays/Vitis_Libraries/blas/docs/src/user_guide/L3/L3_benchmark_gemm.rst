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
   :keywords: BLAS, Library, Vitis BLAS Library, L3, level 3
   :description: Vitis BLAS library level 3 application programming interface reference. Intel Math Kernel Library provides performance improvement of math functions, e.g. GEMM, when running with Intel processors.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


.. _benchmark_gemm_l3:

***********************
L3 API GEMM benchmark
***********************

The benchmark performs the matrix-matrix multiplication (A * B = C), M is number of rows of matrix A/C, K is number of columns of matrix A/number of rows of matrix B, N is number of columns of matrix B/C

1. memKernel
===============
This example resides in ``L3/benchmarks/gemm/memKernel`` directory. The tutorial provides a step-by-step guide that covers commands for building and running kernel.

1.1 Executable Usage
------------------------

1.1.1 Work Directory(Step 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The steps for library download and environment setup can be found in [here](https://github.com/Xilinx/Vitis_Libraries/tree/master/blas/L2/benchmarks#building). For getting the design,

.. code-block:: bash 

   cd L3/benchmarks/gemm/memKernel


1.1.2 Build kernel(Step 2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

.. code-block:: bash 

    make run TARGET=hw PLATFORM_REPO_PATHS=/opt/xilinx/platforms DEVICE=xilinx_u250_xdma_201830_2

1.1.3 Run kernel(Step 3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To get the benchmark results, please run the following command.

Input Arguments:

.. code-block:: bash 

    <host application> <xclbin> <config_info.dat>


For example:

.. code-block:: bash 

    build_dir.hw.xilinx_u250_xdma_201830_2/gemm_bench.exe build_dir.hw.xilinx_u250_xdma_201830_2/blas.xclbin build_dir.hw.xilinx_u250_xdma_201830_2/config_info.dat


1.1.4 Example output(Step 4)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 

.. code-block:: bash 

    xfblasCreate  276.965961 msec
    copyToFpga  0.237744 msec
    copyFromFpga  0.753792 msec
    Api time is 0.991536 msec
    DATA_CSV:,Freq,M,K,N,TimeApiMs,EffApiPct,PerfApiTops
    DATA_CSV:,242.000000,64,64,64,0.991536,0.426753,0.000541
    >> Kernel #0 << Test passed!



1.1.5 Use script to run benchmark
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use mkl to generate dataset, usage of this script is: ./run_gemm_mkl.sh number_of_thread datatype g(generate)/b(benchmark)
Then use run_gemm_bench.sh to run benchmark

.. code-block:: bash 

    cd ../gemm_mkl
    ./run_gemm_mkl.sh 16 float g
    ./run_gemm_bench.sh build_dir.hw.xilinx_u250_xdma_201830_2/blas.xclbin build_dir.hw.xilinx_u250_xdma_201830_2/config_info.dat


1.2 Profiling
-----------------

The xclbin could be built in 242 MHz
The hardware resource utilization and benchmark results are shown in the two tables below.

*Table 1 Hardware resources*

+------------+----------+--------+-------+--------+---------+
|    Name    |   LUT    |  BRAM  |  URAM |   DSP  |    FF   |
+============+==========+========+=======+========+=========+
| blasKernel | 250679   | 94     | 24    | 1224   | 430512  |
+------------+----------+--------+-------+--------+---------+



*Table 2 Benchmark results*

+------+------+------+----------------------------+--------------+---------------+
|  M   |  N   |  K   |  api execution time [ms]   | api Eff [%]  |  PerfApiTops  |
+======+======+======+============================+==============+===============+
| 256  | 256  | 256  | 2.295277                   | 11.798572    | 0.058818      |
+------+------+------+----------------------------+--------------+---------------+
| 512  | 512  | 512  | 7.185994                   | 30.148638    | 0.149859      |
+------+------+------+----------------------------+--------------+---------------+
| 1024 | 1024 | 1024 | 33.357721                  | 51.957490    | 0.257887      |
+------+------+------+----------------------------+--------------+---------------+
| 2048 | 2048 | 2048 | 218.662946                 | 63.410230    | 0.314501      |
+------+------+------+----------------------------+--------------+---------------+
| 4096 | 4096 | 4096 | 1594.648667                | 69.559988    | 0.344877      |
+------+------+------+----------------------------+--------------+---------------+
| 8192 | 8192 | 8192 | 12695.637510               | 69.897233    | 0.346485      |
+------+------+------+----------------------------+--------------+---------------+

2. streamingKernel
======================

This example resides in ``L3/benchmarks/gemm/streamingKernel`` directory. The tutorial provides a step-by-step guide that covers commands for building and running kernel.

2.1 Executable Usage
---------------------

2.1.1 Work Directory(Step 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The steps for library download and environment setup can be found in [here](https://github.com/Xilinx/Vitis_Libraries/tree/master/blas/L2/benchmarks#building). For getting the design,

.. code-block:: bash 

   cd L3/benchmarks/gemm/streamingKernel


2.1.2 Build kernel(Step 2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

.. code-block:: bash 

    make run TARGET=hw PLATFORM_REPO_PATHS=/opt/xilinx/platforms DEVICE=xilinx_u250_gen3x16_xdma_3_1_202020_1


2.1.3 Run kernel(Step 3)
^^^^^^^^^^^^^^^^^^^^^^^^^^

To get the benchmark results, please run the following command.

Input Arguments:

.. code-block:: bash 

    <host application> <xclbin> <config_info.dat>


For example:

.. code-block:: bash 

    build_dir.hw.xilinx_u250_gen3x16_xdma_3_1_202020_1/gemm_bench.exe build_dir.hw.xilinx_u250_gen3x16_xdma_3_1_202020_1/blas.xclbin build_dir.hw.xilinx_u250_gen3x16_xdma_3_1_202020_1/config_info.dat


2.1.4 Example output(Step 4)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 

.. code-block:: bash 

    xfblasCreate  249.914832 msec
    copyToFpga  0.243765 msec
    copyFromFpga  0.437556 msec
    Api time is 0.681321 msec
    DATA_CSV:,Freq,M,K,N,TimeApiMs,EffApiPct,PerfApiTops
    DATA_CSV:,250.000000,64,64,64,0.681321,0.601185,0.000788
    >> Kernel #0 << Test passed!


2.1.5 Use script to run benchmark
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use mkl to generate dataset, usage of this script is: ./run_gemm_mkl.sh number_of_thread datatype g(generate)/b(benchmark)
Then use run_gemm_bench.sh to run benchmark

.. code-block:: bash 

    cd ../gemm_mkl
    ./run_gemm_mkl.sh 16 float g
    ./run_gemm_bench.sh build_dir.hw.xilinx_u250_gen3x16_xdma_3_1_202020_1/blas.xclbin build_dir.hw.xilinx_u250_gen3x16_xdma_3_1_202020_1/config_info.dat


2.2 Profiling
--------------

The xclbin could be built in 250 MHz
The hardware resource utilization and benchmark results are shown in the two tables below.

*Table 1 Hardware resources*

+-------------------------+--------------+-----------+----------+--------+------------+
|    Name                 |      LUT     |    BRAM   |   URAM   |   DSP  |      REG   |
+=========================+==============+===========+==========+========+============+
| gemmAddsKernel          | 101988       | 0         | 0        | 384    | 192516     |
+-------------------------+--------------+-----------+----------+--------+------------+
| gemmCPlusXKernel        | 8529         | 24        | 0        | 66     | 20358      |
+-------------------------+--------------+-----------+----------+--------+------------+
| gemmLoadStoreKernel     | 7126         | 23        | 0        | 16     | 19457      |
+-------------------------+--------------+-----------+----------+--------+------------+
| gemmMergeKernel         | 8342         | 0         | 0        | 0      | 25219      |
+-------------------------+--------------+-----------+----------+--------+------------+
| gemmMulsKernel          | 50640        | 0         | 0        | 768    | 98013      |
+-------------------------+--------------+-----------+----------+--------+------------+
| gemmSystolicArrayKernel | 2541         | 0         | 0        | 0      | 240        |
+-------------------------+--------------+-----------+----------+--------+------------+
| gemmTagsKernel          | 20203        | 15        | 0        | 8      | 34678      |
+-------------------------+--------------+-----------+----------+--------+------------+
| gemmTimerKernel         | 32           | 0         | 0        | 0      | 115        |
+-------------------------+--------------+-----------+----------+--------+------------+



*Table 2 Benchmark results*

+------+------+------+----------------------------+--------------+---------------+
|  M   |  N   |  K   |  api execution time [ms]   | api Eff [%]  |  PerfApiTops  |
+======+======+======+============================+==============+===============+
| 256  | 256  | 256  | 1.370527                   | 19.127241    | 0.024626      |
+------+------+------+----------------------------+--------------+---------------+
| 512  | 512  | 512  | 4.517989                   | 46.417820    | 0.059589      |
+------+------+------+----------------------------+--------------+---------------+
| 1024 | 1024 | 1024 | 29.500145                  | 56.871639    | 0.072902      |
+------+------+------+----------------------------+--------------+---------------+
| 2048 | 2048 | 2048 | 217.555482                 | 61.693563    | 0.079026      |
+------+------+------+----------------------------+--------------+---------------+
| 4096 | 4096 | 4096 | 1685.337895                | 63.710774    | 0.081580      |
+------+------+------+----------------------------+--------------+---------------+

