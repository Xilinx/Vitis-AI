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

.. _l1_hash_join_v2:

============
Hash Join V2
============

Hash Join V2 resides in ``L1/benchmarks/hash_join_v2`` directory.
This benchmark tests the performance of `hashJoinMPU` from `hash_join_v2.hpp`
with the following query.

.. code-block:: bash

   SELECT
          SUM(l_extendedprice * (1 - l_discount)) as revenue
   FROM
          Orders,
          Lineitem
   WHERE
          l_orderkey = o_orderkey
   ;

Here, ``Orders`` is a self-made table filled with random data,which contains a column named ``o_orderkey``.
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

   cd L1/benchmarks/hash_join_v2

* **Build kernel(Step 2)**

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

.. code-block:: bash

   make run TARGET=hw DEVICE=xilinx_u280_xdma_201920_3 HOST_ARCH=x86

* **Run kernel(Step 3)**

To get the benchmark results, please run the following command.

.. code-block:: bash

   ./build_dir.hw.xilinx_u280_xdma_201920_3/test_join.exe -xclbin build_dir.hw.xilinx_u280_xdma_201920_3/hash_join.xclbin

Hash Join V2 Input Arguments:

.. code-block:: bash

   Usage: test_join.exe -xclbin
          -xclbin:      the kernel name

Note: Default arguments are set in Makefile, you can use other platforms to build and run.

* **Example output(Step 4)** 

.. code-block:: bash

   ------------- Hash-Join Test ----------------
   Data integer width is 32.
   Host map buffer has been allocated.
   Lineitem 6001215 rows
   Orders 227597rows
   Lineitem table has been read from disk
   Orders table has been read from disk
   INFO: CPU ref matched 13659906 rows, sum = 6446182945969752
   Found Platform
   Platform Name: Xilinx
   Selected Device xilinx_u280_xdma_201920_3
   INFO: Importing build_dir.hw.xilinx_u280_xdma_201920_3/hash_join_v2.xclbin
   Loading: 'build_dir.hw.xilinx_u280_xdma_201920_3/hash_join_v2.xclbin'
   Kernel has been created
   DDR buffers have been mapped/copy-and-mapped
   FPGA result 0: 644618294596.9752
   Golden result 0: 644618294596.9752
   FPGA execution time of 1 runs: 63985 usec
   Average execution per run: 63985 usec
   INFO: kernel 0: execution time 55945 usec
   ---------------------------------------------   


Profiling
=========

The hash join v2 design is validated on Alveo U280 board at 282 MHz frequency. 
The hardware resource utilizations are listed in the following table.

.. table:: Table 1 Hardware resources for hash join v2
    :align: center

    +----------------+---------------+-----------+------------+----------+
    |      Name      |       LUT     |    BRAM   |    URAM    |    DSP   |
    +----------------+---------------+-----------+------------+----------+
    |   Platform     |     122083    |    202    |     0      |    4     |
    +----------------+---------------+-----------+------------+----------+
    |   join_kernel  |     63697     |    98     |     64     |    3     |
    +----------------+---------------+-----------+------------+----------+
    |   User Budget  |     1180637   |   1814    |     960    |   9020   |
    +----------------+---------------+-----------+------------+----------+
    |   Percentage   |     5.40%     |   5.40%   |    6.67%   |  0.03%   |
    +----------------+---------------+-----------+------------+----------+

The performance is shown below.
   In above test, table ``Lineitem`` has 3 columns and 6001215 rows and ``Orders`` does 1 column and 227597 rows.
   This means that the design takes 55.5945ms to process 69.55MB data, so it achieves 1.21GB/s throughput.


.. toctree::
   :maxdepth: 1

