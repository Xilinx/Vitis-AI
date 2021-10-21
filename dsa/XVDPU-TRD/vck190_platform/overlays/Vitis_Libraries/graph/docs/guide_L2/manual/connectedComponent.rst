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


.. _l2_manual_connectedcomponent:

===================
Connected Component
===================

Connected Component example resides in ``L2/benchmarks/connected_component`` directory. The tutorial provides a step-by-step guide that covers commands for building and running kernel.

Executable Usage
================

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in :ref:`l2_vitis_graph`. For getting the design,

.. code-block:: bash

   cd L2/benchmarks/connected_component

* **Build kernel(Step 2)**

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

.. code-block:: bash

   make run TARGET=hw DEVICE=xilinx_u250_xdma_201830_2

* **Run kernel(Step 3)**

To get the benchmark results, please run the following command.

.. code-block:: bash

   ./build_dir.hw.xilinx_u250_xdma_201830_2/host.exe -xclbin build_dir.hw.xilinx_u250_xdma_201830_2/wcc_kernel.xclbin -o data/test_offset.csr -c data/test_column.csr -g data/test_golden.mtx 

Connected Component Input Arguments:

.. code-block:: bash

   Usage: host.exe -[-xclbin -o -c -g]
          -xclbin     connected component binary
          -o          offset file of input graph in CSR format
          -c          edge file of input graph in CSR format
          -g          golden reference file for validatation

Note: Default arguments are set in Makefile, you can use other :ref:`datasets` listed in the table.  

* **Example output(Step 4)** 

.. code-block:: bash
   
   ---------------------WCC Test----------------
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

The connected component is validated on Alveo U250 board at 280MHz frequency. 
The hardware resource utilization and benchmark results are shown in the two tables below.

.. table:: Table 1 Hardware resources
    :align: center

    +------------+--------------+-----------+----------+--------+
    |    Name    |      LUT     |    BRAM   |   URAM   |   DSP  |
    +------------+--------------+-----------+----------+--------+
    |  Platform  |    104112    |    165    |     0    |    4   |
    +------------+--------------+-----------+----------+--------+
    | wcc_kernel |    103923    |    387    |    112   |    3   |
    +------------+--------------+-----------+----------+--------+
    |    Total   | 208035 (12%) | 552 (21%) | 112 (9%) | 7 (0%) |
    +------------+--------------+-----------+----------+--------+


.. table:: Table 2 Comparison between spark on CPU and FPGA
    :align: center
    
    +------------------+---------+----------+---------+-----------------------+-----------------------+-----------------------+-----------------------+
    |     Datasets     |  Vertex |   Edges  |  FPGA   |   Spark (4 threads)   |   Spark (8 threads)   |   Spark (16 threads)  |   Spark (32 threads)  |
    |                  |         |          |  Time   +------------+----------+------------+----------+------------+----------+------------+----------+
    |                  |         |          |  (u250) | Spark Time | Speed up | Spark Time | Speed up | Spark Time | Speed up | Spark Time | Speed up |
    +------------------+---------+----------+---------+------------+----------+------------+----------+------------+----------+------------+----------+
    |    as-Skitter    | 1696415 | 11095298 |   3401  |    27063   |   7.96   |    18195   |   5.35   |    16382   |   4.82   |    20490   |   6.02   |
    +------------------+---------+----------+---------+------------+----------+------------+----------+------------+----------+------------+----------+
    |   coPapersDBLP   |  540486 | 15245729 |   1958  |    24109   |   12.31  |    17997   |   9.19   |    13723   |   7.01   |    17136   |   8.75   |
    +------------------+---------+----------+---------+------------+----------+------------+----------+------------+----------+------------+----------+
    | coPapersCiteseer |  434102 | 16036720 |   1811  |    24020   |   13.26  |    20516   |   11.33  |    14546   |   8.03   |    18863   |   10.42  |
    +------------------+---------+----------+---------+------------+----------+------------+----------+------------+----------+------------+----------+
    |    cit-Patents   | 3774768 | 16518948 |  16365  |    58366   |   3.57   |    42697   |   2.61   |    34405   |   2.10   |    34862   |   2.13   |
    +------------------+---------+----------+---------+------------+----------+------------+----------+------------+----------+------------+----------+
    |     hollywood    | 1139905 | 57515616 |   7887  |    60888   |   7.72   |    41505   |   5.26   |    34689   |   4.40   |    31272   |   3.97   |
    +------------------+---------+----------+---------+------------+----------+------------+----------+------------+----------+------------+----------+
    | soc-LiveJournal1 | 4847571 | 68993773 |  30519  |   116193   |   3.81   |    91749   |   3.01   |    59977   |   1.97   |    67258   |   2.20   |
    +------------------+---------+----------+---------+------------+----------+------------+----------+------------+----------+------------+----------+
    |   ljournal-2008  | 5363260 | 79023142 |  24334  |   144183   |   5.93   |   102186   |   4.20   |    74971   |   3.08   |    87338   |   3.59   |
    +------------------+---------+----------+---------+------------+----------+------------+----------+------------+----------+------------+----------+
    |      GEOMEAN     |         |          | 7347.43 |  51284.68  |   6.98X  |  37865.87  |   5.15X  |  29071.30  |   3.96X  |  32977.43  |   4.49X  |
    +------------------+---------+----------+---------+------------+----------+------------+----------+------------+----------+------------+----------+

.. note::
    | 1. Spark running on platform with Intel(R) Xeon(R) CPU E5-2690 v4 @2.600GHz, 56 Threads (2 Sockets, 14 Core(s) per socket, 2 Thread(s) per core)
    | 2. Time unit: ms.

.. toctree::
    :maxdepth: 1
