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

.. _l1_hash_group_aggregate:

====================
Hash Group Aggregate
====================

Hash Group Aggregate resides in ``L1/benchmarks/hash_group_aggregate`` directory.

.. code-block:: bash

   SELECT
           max(l_extendedprice), min(l_extendedprice), count_non_zero(l_extendedprice) as revenue
   FROM
           Lineitem
   GROUP BY
           l_orderkey
   ;

Here, ``Lineitem`` is a table filled with random data, which contains 2 columns named ``l_orderkey`` and ``l_extendedprice``.

Dataset
=======

This project uses 32-bit data for numeric fields.
To benchmark 64-bit performance, edit `host/table_dt.h` and make `TPCH_INT` an `int64_t`.

Executable Usage
===============

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in :ref:`l2_vitis_database`. For getting the design,

.. code-block:: bash

   cd L1/benchmarks/hash_group_aggregate

* **Build kernel(Step 2)**

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

.. code-block:: bash

   make run TARGET=hw DEVICE=xilinx_u280_xdma_201920_3 HOST_ARCH=x86

* **Run kernel(Step 3)**

To get the benchmark results, please run the following command.

.. code-block:: bash

   ./build_dir.hw.xilinx_u280_xdma_201920_3/test_aggr.exe -xclbin build_dir.hw.xilinx_u280_xdma_201920_3/hash_aggr_kernel.xclbin

Hash Group Aggregate Input Arguments:

.. code-block:: bash

   Usage: test_aggr.exe -xclbin
          -xclbin:      the kernel name

Note: Default arguments are set in Makefile, you can use other platforms to build and run.

* **Example output(Step 4)** 

.. code-block:: bash

   ---------- Query with TPC-H 1G Data ----------
   
    select max(l_extendedprice), min(l_extendedprice),
           sum(l_extendedprice), count(l_extendedprice)
    from lineitem
    group by l_orderkey
    ---------------------------------------------
   Host map Buffer has been allocated.
   Lineitem 6000000 rows
   Lineitem table has been read from disk
   insert: idx=0 key=180 i_pld=30bca4
   insert: idx=1 key=377 i_pld=6137c
   ...
   Checking: idx=3e5 key:192 pld:d20c5351
   Checking: idx=3e6 key:385 pld:c074aacf
   Checking: idx=3e7 key:104 pld:e6713746
   No error found!
   kernel done!
   kernel_result_num=0x3e8
   FPGA execution time of 3 runs: 104107 usec
   Average execution per run: 34702 usec
   INFO: kernel 0: execution time 31273 usec
   INFO: kernel 1: execution time 56554 usec
   INFO: kernel 2: execution time 43182 usec
   read_config: pu_end_status_a[0]=0x22222222
   read_config: pu_end_status_b[0]=0x22222222
   read_config: pu_end_status_a[1]=0x08
   read_config: pu_end_status_b[1]=0x08
   read_config: pu_end_status_a[2]=0x08
   read_config: pu_end_status_b[2]=0x08
   read_config: pu_end_status_a[3]=0x3e8
   read_config: pu_end_status_b[3]=0x3e8
   ref_result_num=3e8
   ---------------------------------------------
   PASS!
   
   ---------------------------------------------   


Profiling
=========

The hash group aggregate design is validated on Alveo U280 board at 200 MHz frequency. 
The hardware resource utilizations are listed in the following table.

.. table:: Table 1 Hardware resources for hash group aggregate
    :align: center

    +------------------+---------------+-----------+------------+----------+
    |      Name        |       LUT     |    BRAM   |    URAM    |    DSP   |
    +------------------+---------------+-----------+------------+----------+
    |    Platform      |     202971    |    427    |     0      |    10    |
    +------------------+---------------+-----------+------------+----------+
    | hash_aggr_kernel |     184064    |    207    |    256     |    0     |
    +------------------+---------------+-----------+------------+----------+
    |   User Budget    |     1099749   |    1589   |    960     |   9014   |
    +------------------+---------------+-----------+------------+----------+
    |   Percentage     |     16.74%    |   13.03%  |   26.67%   |    0     |
    +------------------+---------------+-----------+------------+----------+

The performance is shown below:
   In above test, table ``Lineitem`` has 2 columns and 6000000 rows.
   This means that the design takes 34.702ms to process 45.78MB data, so it achieves 1.29GB/s throughput.


.. toctree::
   :maxdepth: 1

