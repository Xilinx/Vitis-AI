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

.. _l1_hash_anti_join:

==============
Hash Anti-join
==============

Hash Anti-join resides in ``L1/benchmarks/hash_anti_join`` directory.
This benchmark tests the performance of `anti_join_build_probe` from `hash_anti_join.h`
with the following query.

.. code-block:: bash

   SELECT
           SUM(l_extendedprice * (1 - l_discount))
   FROM
           Lineitem
   WHERE
           l_orderkey NOT IN (SELECT o_orderkey FROM Orders)
   ;



Here ``Orders`` is a self-made table filled with random data, which contains a column named ``o_orderkey``; 
     ``Lineitem`` is also a table, which contains 3 columns named ``l_orderkey``, ``l_extendedprice`` and ``l_discount``.

Dataset
=======

This project uses 32-bit data for numeric fields.
To benchmark 64-bit performance, edit `host/table_dt.h` and make `TPCH_INT` an `int64_t`.

Executable Usage
===============

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in :ref:`l2_vitis_database`. For getting the design,

.. code-block:: bash

   cd L1/benchmarks/hash_anti_join

* **Build kernel(Step 2)**

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

.. code-block:: bash

   make run TARGET=hw DEVICE=xilinx_u280_xdma_201920_3 HOST_ARCH=x86

* **Run kernel(Step 3)**

To get the benchmark results, please run the following command.

.. code-block:: bash

   ./build_dir.hw.xilinx_u280_xdma_201920_3/test_join.exe -xclbin build_dir.hw.xilinx_u280_xdma_201920_3/hash_anti_join.xclbin

Hash anti-join Input Arguments:

.. code-block:: bash

   Usage: test_join.exe -xclbin
          -xclbin:      the kernel name


* **Example output(Step 4)** 

.. code-block:: bash

   ------------- Hash-Join Test ----------------
   Data integer width is 32.
   Host map buffer has been allocated.
   Lineitem 6001215 rows
   Orders 227597rows
   Lineitem table has been read from disk
   Orders table has been read from disk
   INFO: CPU ref matched 611326 rows, sum = 288203573816672
   Found Platform
   Platform Name: Xilinx
   Selected Device xilinx_u280_xdma_201920_3
   INFO: Importing build_dir.hw.xilinx_u280_xdma_201920_3/hash_anti_join.xclbin
   Loading: 'build_dir.hw.xilinx_u280_xdma_201920_3/hash_anti_join.xclbin'
   Kernel has been created
   DDR buffers have been mapped/copy-and-mapped
   Test Pass
   FPGA result 0: 28820357381.6672
   Golden result 0: 28820357381.6672
   FPGA execution time of 1 runs: 352393 usec
   Average execution per run: 352393 usec
   INFO: kernel 0: execution time 342568 usec
   ---------------------------------------------


Profiling
=========

The hash anti-join desgin is validated on Alveo U280 board at 250 MHz frequency. 
The hardware resource utilizations are listed in the following table.


.. table:: Table 1 Hardware resources for hash anti-join
    :align: center

    +----------------+---------------+-----------+------------+----------+
    |      Name      |       LUT     |    BRAM   |    URAM    |    DSP   |
    +----------------+---------------+-----------+------------+----------+
    |    Platform    |     130442    |    204    |     0      |    4     |
    +----------------+---------------+-----------+------------+----------+
    |   join_kernel  |     134647    |    291    |    192     |    99    |
    +----------------+---------------+-----------+------------+----------+
    |   User Budget  |     1172278   |   1812    |    960     |   9020   |
    +----------------+---------------+-----------+------------+----------+
    |   Percentage   |     11.49%    |  16.06%   |   20.00%   |   1.10%  |
    +----------------+---------------+-----------+------------+----------+

The performance is shown below.
   In above test, table ``Lineitem`` has 3 columns and 6001215 rows and ``Orders`` does 1 column and 227597 rows.
   This means that the design takes 342.568ms to process 69.55MB, so it achieves 203.02MB/s throughput.


.. toctree::
   :maxdepth: 1

