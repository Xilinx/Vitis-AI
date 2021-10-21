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


.. _module_benchmark:

=========================
Benchmark
=========================

1. Performance
=========================
- Kernel execution time only includes kernel running in fpga device time
- api execution time include Kernel execution time + memory copy between host and kernel time 

1.1 gemv
----------------------
This benchmark performs the matrix-vecotr multiplication, M is number of rows of matrix, N is number of columns of matrix

*gemv with OpenCL in u280*

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

For more details on this benchmark, see:

.. toctree::
   :maxdepth: 1
   
   user_guide/L2/L2_benchmark_gemv.rst


1.2 gemm
---------------
This benchmark performs the matrix-matrix multiplication (A * B = C), M is number of rows of matrix A/C, K is number of columns of matrix A/number of rows of matrix B, N is number of columns of matrix B/C

*gemm with OpenCL in u250*

+------+------+------+------------------------------+--------------------------+-----------------+
|  M   |  N   |  K   |  Kernel execution time [ms]  |  api execution time [ms] | Kernel Eff [%]  |  
+======+======+======+==============================+==========================+=================+
| 64   | 64   | 64   | 0.010905                     | 1.750123                 | 38.802577       | 
+------+------+------+------------------------------+--------------------------+-----------------+
| 128  | 128  | 128  | 0.048517                     | 13.802416                | 69.772592       | 
+------+------+------+------------------------------+--------------------------+-----------------+
| 256  | 256  | 256  | 0.328314                     | 14.645931                | 82.485022       | 
+------+------+------+------------------------------+--------------------------+-----------------+
| 512  | 512  | 512  | 3.213388                     | 18.199255                | 67.420400       | 
+------+------+------+------------------------------+--------------------------+-----------------+
| 1024 | 1024 | 1024 | 24.113855                    | 45.519852                | 71.875005       | 
+------+------+------+------------------------------+--------------------------+-----------------+
| 2048 | 2048 | 2048 | 186.688153                   | 264.195138               | 74.270743       | 
+------+------+------+------------------------------+--------------------------+-----------------+
| 4096 | 4096 | 4096 | 1469.773731                  | 1708.938204              | 75.469945       | 
+------+------+------+------------------------------+--------------------------+-----------------+

For more details on this benchmark, see:

.. toctree::
   :maxdepth: 1
   
   user_guide/L2/L2_benchmark_gemm.rst

*gemm with XRT in u250*

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

*gemm with XRT (one CU, streaming Kernel) in u250*

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

For more details on the benchmarks, see:

.. toctree::
   :maxdepth: 1
   
   user_guide/L3/L3_benchmark_gemm.rst


2. Benchmark Test Overview
============================

Here are benchmarks of the Vitis BLAS library using the Vitis environment. It supports software and hardware emulation as well as running hardware accelerators on the Alveo U250.

2.1 Prerequisites
----------------------

2.1.1 Vitis BLAS Library
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Alveo U250 installed and configured as per https://www.xilinx.com/products/boards-and-kits/alveo/u250.html#gettingStarted (when running hardware)
- Xilinx runtime (XRT) installed
- Xilinx Vitis 2021.1 installed and configured

2.2 Building
----------------

2.2.1 Download code
^^^^^^^^^^^^^^^^^^^^^

These blas benchmarks can be downloaded from [vitis libraries](https://github.com/Xilinx/Vitis_Libraries.git) ``master`` branch.

.. code-block:: bash 

   git clone https://github.com/Xilinx/Vitis_Libraries.git
   cd Vitis_Libraries
   git checkout master
   cd blas

   
2.2.2 Setup environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

Setup and build envrionment using the Vitis and XRT scripts:

.. code-block:: bash 

    source <install path>/Vitis/2021.1/settings64.sh
    source /opt/xilinx/xrt/setup.sh
