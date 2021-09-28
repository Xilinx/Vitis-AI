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

.. _l2_manual_labelpropagation:

=================
Label Propagation
=================

Label Propagation example resides in ``L2/benchmarks/label_propagation`` directory. The tutorial provides a step-by-step guide that covers commands for building and running kernel.

Executable Usage
================

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in :ref:`l2_vitis_graph`. For getting the design,

.. code-block:: bash

   cd L2/benchmarks/label_propagation

* **Build kernel(Step 2)**

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

.. code-block:: bash

   make run TARGET=hw DEVICE=xilinx_u250_xdma_201830_2

* **Run kernel(Step 3)**

To get the benchmark results, please run the following command.

.. code-block:: bash

   ./build_dir.hw.xilinx_u250_xdma_201830_2/host.exe -xclbin build_dir.hw.xilinx_u250_xdma_201830_2/LPKernel.xclbin -o data/csr_offsets.txt -i data/csr_columns.txt -label data/label.txt

Label Propagation Input Arguments:

.. code-block:: bash

   Usage: host.exe -[-xclbin -o -i -label]
          -xclbin         label propagation binary
          -o              offset file of input graph in CSR format
          -i              edge file of input graph in CSR format
          -label          golden reference file for validatation

Note: Default arguments are set in Makefile, you can use other :ref:`datasets` listed in the table.  

* **Example output(Step 4)**

.. code-block:: bash

   ---------------------Label Propagation----------------
   Found Platform
   Platform Name: Xilinx
   Found Device=xilinx_u250_xdma_201830_2
   INFO: Importing build_dir.hw.xilinx_u250_xdma_201830_2/LPKernel.xclbin
   Loading: 'build_dir.hw.xilinx_u250_xdma_201830_2/LPKernel.xclbin'
   kernel has been created
   kernel start------
   vertexNum=11   edgeNum=20
   burstReadSplit2Strm ing
   offsetCtrlIndex ing
   mergeSortWrapper ing
   labelProcess ing
   combineStrm ing
   kernel end------
   Execution time 218.327ms
   Write DDR Execution time 38.692 ms
   Kernel Execution time 13.2172 ms
   Read DDR Execution time 93.9891 ms
   Total Execution time 218.276 ms
   INFO: case pass!

Profiling
=========

The hardware resource utilizations are listed in the following table.

.. table:: Table 1 Hardware resources
    :align: center

    +------------------+----------+----------+----------+---------+-----------------+
    |  Kernel          |   BRAM   |   URAM   |    DSP   |   LUT   | Frequency(MHz)  |
    +------------------+----------+----------+----------+---------+-----------------+
    |  Platform        |    375   |    0     |    7     |  162080 |                 |
    +------------------+----------+----------+----------+---------+-----------------+
    |  LP_Kernel       |    100   |    0     |    0     |   72777 |      292        |
    +------------------+----------+----------+----------+---------+-----------------+


The performance is shown in the table below.

.. table:: Table 2 Comparison between CPU and FPGA (iteration=30) 
    :align: center

    +------------------+----------+----------+-----------+-----------------------+-----------------------+-----------------------+-----------------------+
    |                  |          |          |           |     Spark (4 threads) |     Spark (8 threads) |    Spark (16 threads) |    Spark (32 threads) |
    | Datasets         | Vertex   | Edges    | FPGA time +------------+----------+------------+----------+------------+----------+------------+----------+
    |                  |          |          | (u250)    | Spark time |  speedup | Spark time |  speedup | Spark time |  speedup | Spark time |  speedup |
    +------------------+----------+----------+-----------+------------+----------+------------+----------+------------+----------+------------+----------+
    | as-Skitter       | 1694616  | 11094209 |  27.85    |  1336.85   |  48.01   |   524.35   |  18.83   |   348.45   |  12.51   |  314.62    |  11.30   |
    +------------------+----------+----------+-----------+------------+----------+------------+----------+------------+----------+------------+----------+
    | coPapersDBLP     | 540486   | 15245729 |  31.00    |   619.02   |  19.97   |   342.48   |  11.05   |   314.44   |  10.14   |  346.20    |  11.17   |
    +------------------+----------+----------+-----------+------------+----------+------------+----------+------------+----------+------------+----------+
    | coPapersCiteseer | 434102   | 16036720 |  31.16    |   566.42   |  18.18   |   335.87   |  10.78   |   319.40   |  10.25   |  350.42    |  11.25   |
    +------------------+----------+----------+-----------+------------+----------+------------+----------+------------+----------+------------+----------+
    | cit-Patents      | 3774768  | 16518948 |  40.51    |   976.52   |  24.10   |   588.92   |  14.54   |   529.59   |  13.07   |  501.36    |  12.37   |
    +------------------+----------+----------+-----------+------------+----------+------------+----------+------------+----------+------------+----------+
    | europe_osm       | 50912018 | 54054660 | 250.56    |  3095.14   |  12.35   |  2567.74   |  10.25   |  2047.45   |   8.17   | 1679.05    |   6.70   |
    +------------------+----------+----------+-----------+------------+----------+------------+----------+------------+----------+------------+----------+
    | hollywood        | 1139905  | 57515616 | 107.39    | 48523.23   | 451.83   | 15495.58   | 144.29   |  8589.30   |  79.98   | 9118.71    |  84.91   |
    +------------------+----------+----------+-----------+------------+----------+------------+----------+------------+----------+------------+----------+
    | soc-LiveJournal1 | 4847571  | 68993773 | 143.20    |  4017.49   |  28.05   |  2018.39   |  14.09   |  1529.69   |  10.68   | 1577.56    |  11.02   |
    +------------------+----------+----------+-----------+------------+----------+------------+----------+------------+----------+------------+----------+
    | ljournal-2008    | 5363260  | 79023142 | 162.31    |  5027.63   |  30.98   |  2216.32   |  13.65   |  1846.45   |  11.38   | 1735.08    |  10.69   |
    +------------------+----------+----------+-----------+------------+----------+------------+----------+------------+----------+------------+----------+
    | GEOMEAN          |          |          |  71.48    |  2470.70   |  34.56X  |  1259.24   |  17.62X  |   989.71   |  13.85X  |  972.79    |  13.61X  |
    +------------------+----------+----------+-----------+------------+----------+------------+----------+------------+----------+------------+----------+

.. note::
    | 1. Spark running on platform with Intel(R) Xeon(R) CPU E5-2690 v4 @2.600GHz, 56 Threads (2 Sockets, 14 Core(s) per socket, 2 Thread(s) per core).
    | 2. time unit: second.

.. toctree::
   :maxdepth: 1


