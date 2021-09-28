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

.. _l2_spmv_double:

=======================
SPMV (Double precision)
=======================

SPMV (Double precision) resides in ``L2/benchmarks/spmv_double`` directory.


Dataset
=======

There are 22 sparse matrices used in the benchmark. These sparse matrices can be downloaded from https://sparse.tamu.edu.

+---------------+------+------+--------+
| matrix        | rows | cols | NNZs   | 
+===============+======+======+========+
| nasa2910      | 2910 | 2910 | 174296 | 
+---------------+------+------+--------+
| ex9           | 3363 | 3363 | 99471  | 
+---------------+------+------+--------+
| bcsstk24      | 3562 | 3562 | 159910 | 
+---------------+------+------+--------+
| bcsstk15      | 3948 | 3948 | 117816 | 
+---------------+------+------+--------+
| bcsstk28      | 4410 | 4410 | 219024 | 
+---------------+------+------+--------+
| s3rmt3m3      | 5357 | 5357 | 207695 | 
+---------------+------+------+--------+
| s2rmq4m1      | 5489 | 5489 | 281111 | 
+---------------+------+------+--------+
| nd3k          | 9000 | 9000 | 3279690| 
+---------------+------+------+--------+
| ted_B_unscaled| 10605| 10605| 144579 | 
+---------------+------+------+--------+
| ted_B         | 10605| 10605| 144579 | 
+---------------+------+------+--------+
| msc10848      | 10848| 10848| 1229778| 
+---------------+------+------+--------+
| cbuckle       | 13681| 13681| 676515 | 
+---------------+------+------+--------+
| olafu         | 16146| 16146| 1015156| 
+---------------+------+------+--------+
| gyro_k        | 17361| 17361| 1021159| 
+---------------+------+------+--------+
| bodyy4        | 17546| 17546| 121938 | 
+---------------+------+------+--------+
| nd6k          | 18000| 18000| 6897316| 
+---------------+------+------+--------+
| raefsky4      | 19779| 19779| 1328611| 
+---------------+------+------+--------+
| bcsstk36      | 23052| 23052| 1143140| 
+---------------+------+------+--------+
| msc23052      | 23052| 23052| 1154814| 
+---------------+------+------+--------+
| ct20stif      | 52329| 52329| 2698463| 
+---------------+------+------+--------+
| nasasrb       | 54870| 54870| 2677324| 
+---------------+------+------+--------+
| bodyy6        | 19366| 19366| 134748 | 
+---------------+------+------+--------+

Executable Usage
===============

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in :ref:`l2_vitis_sparse`. For getting the design,

.. code-block:: bash

   cd L2/benchmarks/spmv_double

* **Build hw and host (Step 2)**

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

.. code-block:: bash

   make build TARGET=hw PLATFORM_REPO_PATHS=/opt/xilinx/platforms DEVICE=xilinx_u280_xdma_291020_3
   make host TARGET=hw PLATFORM_REPO_PATHS=/opt/xilinx/platforms DEVICE=xilinx_u280_xdma_291020_3

* **Generate inputs(Step 3)**

.. code-block:: bash

    conda activate xf_blas
    source ./gen_test.sh

The gen_test.sh triggers a set of python scripts to download the .mtx files listed in test.txt under current directory and partitions them evenly across 16 HBM channels. Each paritioned data set, including the value and indices of each NNZ entry, is stored in one HBM channel. Each row of the partitioned data set is padded to multiple of 32 to accommodate the double precision accumulation latency. The padding overhead for each matrix is summarized in the benchmark result as well. This overhead will be reduced with the improvement of floating point support on FPGA platforms.

* **Run benchmark(Step 4)**

To get the benchmark results, please run the following command.

.. code-block:: bash

    python ./run_test.py

The run_test.py launches the host executable with each partitioned data set and offloads the double precision SpMV operation to U280 card. The SpMV operation is run numerous time (2000 in this benchmark) to mask out the host code overhead. The total run time in the benchmark results includs the OpenCl function call time to trigger the CUs and the hardware run time. The run time [ms] / iteration field gives single SpMV run time on the U280 card.

* **Example output(Step 5)** 

.. code-block:: bash

    All tests pass!
    Please find the benchmark results in spmv_perf.csv.

Profiling
=========

The SPMV double precision design is validated on Alveo U280 board at 256 MHz frequency. 
The hardware resource utilizations are listed in the following table.

.. table:: Table 1 Hardware resources for SPMV double precision design
    :align: center
    
    +--------------------------+---------------+-----------+---------+----------+
    |           Name           |       LUT     |    BRAM   |   URAM  |    DSP   |
    +--------------------------+---------------+-----------+---------+----------+
    |         Platform         |  165475       | 323       | 64      |  4       |
    +--------------------------+---------------+-----------+---------+----------+
    |        SPMV design       |  220980       | 211       | 64      | 900      |
    +--------------------------+---------------+-----------+---------+----------+
    |        User Budget       |  1137245      | 1693      | 896     | 9020     |
    +--------------------------+---------------+-----------+---------+----------+
    |        Percentage        |     19.43%    |   12.46%  |   7.14% |   9.98%  |
    +--------------------------+---------------+-----------+---------+----------+

The performance result is shown below.

    +---------------+------+-----------------+----------------+
    | matrix        | runs | total time[sec] | time[ms]/run   | 
    +===============+======+=================+================+
    | nasa2910      | 2000 | 0.102513        | 0.0512565      | 
    +---------------+------+-----------------+----------------+
    | ex9           | 2000 | 0.0759525       | 0.0379762      | 
    +---------------+------+-----------------+----------------+
    | bcsstk24      | 2000 | 0.0747713       | 0.0373857      | 
    +---------------+------+-----------------+----------------+
    | bcsstk15      | 2000 | 0.0872443       | 0.0436221      | 
    +---------------+------+-----------------+----------------+
    | bcsstk28      | 2000 | 0.116322        | 0.0581609      | 
    +---------------+------+-----------------+----------------+
    | s3rmt3m3      | 2000 | 0.106942        | 0.0534711      | 
    +---------------+------+-----------------+----------------+
    | s2rmq4m1      | 2000 | 0.126217        | 0.0631087      | 
    +---------------+------+-----------------+----------------+
    | nd3k          | 2000 | 0.677946        | 0.338973       | 
    +---------------+------+-----------------+----------------+
    | ted_B_unscaled| 2000 | 0.136411        | 0.0682054      | 
    +---------------+------+-----------------+----------------+
    | ted_B         | 2000 | 0.149135        | 0.0745673      | 
    +---------------+------+-----------------+----------------+
    | msc10848      | 2000 | 0.391394        | 0.195697       | 
    +---------------+------+-----------------+----------------+
    | cbuckle       | 2000 | 0.216792        | 0.108396       | 
    +---------------+------+-----------------+----------------+
    | olafu         | 2000 | 0.263899        | 0.131949       | 
    +---------------+------+-----------------+----------------+
    | gyro_k        | 2000 | 0.412774        | 0.206387       | 
    +---------------+------+-----------------+----------------+
    | bodyy4        | 2000 | 0.269815        | 0.134907       | 
    +---------------+------+-----------------+----------------+
    | nd6k          | 2000 | 1.50509         | 0.752544       | 
    +---------------+------+-----------------+----------------+
    | raefsky4      | 2000 | 0.446744        | 0.223372       | 
    +---------------+------+-----------------+----------------+
    | bcsstk36      | 2000 | 0.374293        | 0.187146       | 
    +---------------+------+-----------------+----------------+
    | msc23052      | 2000 | 0.723612        | 0.361806       | 
    +---------------+------+-----------------+----------------+
    | ct20stif      | 2000 | 1.01894         | 0.509468       | 
    +---------------+------+-----------------+----------------+
    | nasasrb       | 2000 | 0.780656        | 0.390328       | 
    +---------------+------+-----------------+----------------+
    | bodyy6        | 2000 | 0.247517        | 0.123759       | 
    +---------------+------+-----------------+----------------+

.. toctree::
   :maxdepth: 1

