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

.. _l2_manual_twoHop:   

===========================
Two hop path count
===========================

Two hop path count (twoHop) example resides in ``L2/benchmarks/twoHop`` directory. The tutorial provides a step-by-step guide that covers commands for building and running kernel.

Executable Usage
================

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in :ref:`l2_vitis_graph`. For getting the design,

.. code-block:: bash

   cd L2/benchmarks/twoHop

* **Build kernel(Step 2)**

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

.. code-block:: bash

   make run TARGET=hw DEVICE=xilinx_u50_gen3x16_xdma_201920_3

* **Run kernel(Step 3)**

To get the benchmark results, please run the following command.

.. code-block:: bash

   ./build_dir.hw.xilinx_u50_gen3x16_xdma_201920_3/host.exe -xclbin build_dir.hw.xilinx_u50_gen3x16_xdma_201920_3/twoHop_kernel.xclbin --offset ./data/data-csr-offset.mtx --index ./data/data-csr-indicesweights.mtx --pair ./data/data-pair.mtx --golden ./data/data-golden.twoHop.mtx 

Two hop path count Input Arguments:

.. code-block:: bash

   Usage: host.exe -[-xclbin --offset --index --pair --golden]
          -xclbin      Xclbin File Name
          --offset     Offset File Name
          --index      Indices File Name
          --pair       Pair File Name
          --golden     Golden File Name

Note: Default arguments are set in Makefile, you can use other :ref:`datasets` listed in the table.  

* **Example output(Step 4)**

.. code-block:: bash

    ---------------------Two Hop-------------------
    Found Platform
    Platform Name: Xilinx
    Found Device=xilinx_u50_gen3x16_xdma_201920_3
    INFO: Importing ./twoHop_kernel.xclbin
    Loading: './twoHop_kernel.xclbin'
    kernel has been created
    kernel start------
    kernel end------
    Execution time 3.392ms
    Write DDR Execution time 0.243388ms
    kernel Execution time 2.85988ms
    Read DDR Execution time 0.116564ms

Profiling
=========

The hardware resource utilizations are listed in the following table.

.. table:: Table 1 Hardware resources
    :align: center

    +----------------+----------+----------+----------+---------+-----------------+
    |  Kernel        |   BRAM   |   URAM   |    DSP   |   LUT   | Frequency(MHz)  |
    +----------------+----------+----------+----------+---------+-----------------+
    | twoHop_Kernel  |    42    |     0    |    0     |  6825   |      300        |
    +----------------+----------+----------+----------+---------+-----------------+

The performance is shown below.

.. table:: Table 2 Performance
    :align: center

    +------------------+----------+----------+-----------+
    |                  |          |          |           |
    | Datasets         | Vertex   | Edges    | u50 time  | 
    |                  |          |          | (s)       |
    +------------------+----------+----------+-----------+
    | as-Skitter       | 1694616  | 11094209 |     10.16 |
    +------------------+----------+----------+-----------+
    | coPapersDBLP     | 540486   | 15245729 |     50.25 |
    +------------------+----------+----------+-----------+
    | coPapersCiteseer | 434102   | 16036720 |     80.52 |
    +------------------+----------+----------+-----------+
    | cit-Patents      | 3774768  | 16518948 |      7.41 |
    +------------------+----------+----------+-----------+
    | europe_osm       | 50912018 | 54054660 |      1.91 |
    +------------------+----------+----------+-----------+
    | hollywood        | 1139905  | 57515616 |    289.24 |
    +------------------+----------+----------+-----------+
    | soc-LiveJournal1 | 4847571  | 68993773 |     34.72 |
    +------------------+----------+----------+-----------+
    | ljournal-2008    | 5363260  | 79023142 |     38.90 |
    +------------------+----------+----------+-----------+
