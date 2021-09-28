.. 
   Copyright 2021 Xilinx, Inc.
  
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
  
       http://www.apache.org/licenses/LICENSE-2.0
  
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. _l2_log_analyzer:

============
Log Analyzer
============

Log Analyzer resides in ``L2/demos/text/log_analyzer`` directory.
It is an integration frame included 3 part: Grok, GeoIP and JsonWriter. 


Dataset
=======

- Input log: http://www.almhuette-raith.at/apache-log/access.log (1.2GB)
- logAnalyzer Demo execute time: 0.99 s, throughput: 1.2 GB/s
- Baseline `ref_result/ref_result.cpp` execute time: 53.1 s, throughput: 22.6 MB/s
- Accelaration Ratio: 53X

.. note::
    | 1. The baseline version run on Intel(R) Xeon(R) CPU E5-2690 v4, clocked at 2.60GHz.
    | 2. The baseline version is a single thread program.



Executable Usage
===============

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in :ref:`l2_vitis_data_analytics`. For getting the design,

.. code-block:: bash

   cd L2/demos/text/log_analyzer

* **Build kernel(Step 2)**

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

.. code-block:: bash

   make run TARGET=hw DEVICE=xilinx_u200_xdma_201830_2 HOST_ARCH=x86

* **Run kernel(Step 3)**

To get the benchmark results, please run the following command.

.. code-block:: bash

   ./build_dir.hw.xilinx_u200_xdma_201830_2/test.exe -xclbin ./build_dir.hw.xilinx_u200_xdma_201830_2/logAnalyzer.xclbin -log ./data/access.log -dat ./data/geo.dat -ref ./data/golden.json

Log Analyzer Input Arguments:

.. code-block:: bash

   Usage: test.exe -xclbin <xclbin_name> -log <input log> -data <input geo path> -ref <golden data>
          -xclbin:     the kernel name
          -log   :     input log
          -data  :     input geo path
          -ref   :     golden data


* **Example output(Step 4)** 

.. code-block:: bash

   ----------------------log analyzer----------------
   DEBUG: found device 0: xilinx_u200_xdma_201830_2
   INFO: initilized context.
   INFO: initilized command queue.
   INFO: created program with binary build_dir.hw.xilinx_u200_xdma_201830_2/logAnalyzer.xclbin
   INFO: built program.
   load log from disk to in-memory buffer
   load geoip database disk to in-memory buffer
   execute log analyzer
   geoIPConvert
   netsLow21 actual use buffer size is 333
   required geo buffer size 1454733
   The log file is partition into 1 slice with max_slice_lnm 102 and  takes 0.006000 ms.
   DEBUG: reEngineKernel has 4 CU(s)
   DEBUG: GeoIP_kernel has 1 CU(s)
   DEBUG: WJ_kernel has 1 CU(s)
   logAnalyzer pipelined, time: 5.401 ms, size: 0 MB, throughput: 0 GB/s
   -----------------------------Finished logAnalyzer pipelined test----------------------------------------------
   

Profiling
=========

The log analyzer design is validated on Alveo U200 board at 251 MHz frequency. 
The hardware resource utilizations are listed in the following table.

.. table:: Table 1 Hardware resources for log analyzer
    :align: center
 
    +---------------------+---------+--------+--------+-------+
    | Name                | LUT     | BRAM   | URAM   |  DSP  |
    +---------------------+---------+--------+--------+-------+
    | Platform            | 282591  |  835   |   0    |   16  |
    +---------------------+---------+--------+--------+-------+
    | GeoIP_kernel        |  28802  |   24   |  16    |    8  |
    +---------------------+---------+--------+--------+-------+
    | WJ_kernel           |  32028  |   44   |   0    |    2  |
    +---------------------+---------+--------+--------+-------+
    | reEngineKernel      | 165934  |  264   | 192    |   12  |
    +---------------------+---------+--------+--------+-------+
    |    reEngineKernel_1 |  41412  |   66   |  48    |    3  |
    +---------------------+---------+--------+--------+-------+
    |    reEngineKernel_2 |  41496  |   66   |  48    |    3  |
    +---------------------+---------+--------+--------+-------+
    |    reEngineKernel_3 |  41514  |   66   |  48    |    3  |
    +---------------------+---------+--------+--------+-------+
    |    reEngineKernel_4 |  41512  |   66   |  48    |    3  |
    +---------------------+---------+--------+--------+-------+
    | User Budget         | 899649  | 1325   | 960    | 6824  |
    +---------------------+---------+--------+--------+-------+
    | Used Resources      | 226764  |  332   | 208    |   22  |
    +---------------------+---------+--------+--------+-------+
    | Percentage          | 25.21%  | 25.06% | 21.67% | 0.32% |
    +---------------------+---------+--------+--------+-------+

The performance is shown below.
   This benchmark takes 0.99s to process 1.2GB data, so its throughput is 1.2GB/s.


.. toctree::
   :maxdepth: 1

