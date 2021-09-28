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
   :keywords: Adler32
   :description: The hardware resources and performance for Adler32
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


.. _guide_l1_benchmark_adler32:


========
Adler32
========

To profile performance of adler32, we prepare a datapack of 268,435,456 byte messages as kernel input.
Base on U50, We have 1 kernel, each kernel has 1 PU.
Kernel utilization and throughput is shown in table below.

Executable Usage
================

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in :ref:`l1_vitis_security`. For getting the design,

.. code-block:: bash

   cd L1/benchmarks/adler32

* **Build kernel(Step 2)**

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

.. code-block:: bash

   source /opt/xilinx/Vitis/2021.1/settings64.sh
   source /opt/xilinx/xrt/setenv.sh
   export DEVICE=u50_gen3x16
   export TARGET=hw
   make run 

* **Run kernel(Step 3)**

To get the benchmark results, please run the following command.

.. code-block:: bash

   ./BUILD_DIR/host.exe -xclbin ./BUILD_DIR/Adler32Kernel.xclbin -data PROJECT/data/test.dat -num 16

Input Arguments:

.. code-block:: bash

   Usage: host.exe -[-xclbin]
          -xclbin     binary;

* **Example output(Step 4)**

.. code-block:: bash

   kernel has been created
   kernel start------
   kernel end------
   Execution time 724.018ms
   Write DDR Execution time 1.19501 ms
   Kernel Execution time 721.203 ms
   Read DDR Execution time 0.07055 ms
   Total Execution time 723.504 ms


Profiling 
=========

The Adler32 is validated on Xilinx Alveo U50 board. 
Its resource, frequency and throughput is shown as below.

+-----------+------------+------------+----------+--------+--------+-------------+
| Frequency |    LUT     |     REG    |   BRAM   |  URAM  |   DSP  | Throughput  |
+-----------+------------+------------+----------+--------+--------+-------------+
| 262 MHz   |   6,348    |   12,232   |   16     |   0    |   0    |   4.1 GB/s  |
+-----------+------------+------------+----------+--------+--------+-------------+


.. toctree::
   :maxdepth: 1
