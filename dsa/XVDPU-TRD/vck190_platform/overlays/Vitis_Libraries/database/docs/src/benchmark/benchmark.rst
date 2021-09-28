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


.. Project documentation master file, created by
   sphinx-quickstart on Thu Jun 20 14:04:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==========
Benchmark 
==========
    
.. _datasets:

Datasets
-----------

There are two tables, named ``Lineitem`` and ``Orders``, which are filled with random data.

Performance
-----------

For representing the resource utilization in each benchmark, we separate the overall utilization into 2 parts, where P stands for the resource usage in
platform, that is those instantiated in static region of the FPGA card, as well as K represents those used in kernels (dynamic region). The target device is set to Alveo U280.

.. table:: Table 1 Performance on FPGA
    :align: center

    +-------------------------------+----------------------------+--------------+----------+-----------------+------------+------------+------------+
    |     Architecture              |     Dataset                |  Latency(ms) |  Timing  |   LUT(P/K)      |  BRAM(P/K) |  URAM(P/K) |  DSP(P/K)  |
    +===============================+============================+==============+==========+=================+============+============+============+
    |  Compound Sort (U280)         |  Orders 131072 rows        |    1.130     |  287MHz  |  142.0K/62.7K   |   285/18   |    0/16    |    7/0     |
    +-------------------------------+----------------------------+--------------+----------+-----------------+------------+------------+------------+
    |  Hash Anti-Join (U280)        |  Lineitem 6001215 rows     |    342.568   |  250MHz  |  130.4K/134.6K  |   204/291  |    0/192   |    4/99    |
    |                               |  Orders 227597 rows        |              |          |                 |            |            |            |
    +-------------------------------+----------------------------+--------------+----------+-----------------+------------+------------+------------+
    |  Hash Group Aggregate (U280)  |  Lineitem 6000000 rows     |    34.702    |  200MHz  |  203.0K/184.1K  |   427/207  |    0/256   |    10/0    |
    +-------------------------------+----------------------------+--------------+----------+-----------------+------------+------------+------------+
    |  Hash Join V2 (U280)          |  Lineitem 6001215 rows     |    55.95     |  282MHz  |  122.1K/63.7K   |   202/98   |    0/64    |    4/3     |
    |                               |  Orders 227597 rows        |              |          |                 |            |            |            |
    +-------------------------------+----------------------------+--------------+----------+-----------------+------------+------------+------------+
    |  Hash Join V3 (U280)          |  Lineitem 6001215 rows     |    65.26     |  240MHz  |  197.0K/128.2K  |   359/239  |    0/192   |    10/99   |
    |                               |  Orders 227597 rows        |              |          |                 |            |            |            |
    +-------------------------------+----------------------------+--------------+----------+-----------------+------------+------------+------------+
    |  Hash Join V4 (U280)          |  Lineitem 6001215 rows     |    1354.795  |  240MHz  |  201.5/110.1K   |   359/187  |    0/256   |    10/19   |
    |                               |  Orders 227597 rows        |              |          |                 |            |            |            |
    +-------------------------------+----------------------------+--------------+----------+-----------------+------------+------------+------------+
    |  Hash Multi-Join (U280)       |  Lineitem 6001215 rows     |    76.899    |  200MHz  |  130.6K/133.4K  |   204/271  |    0/192   |    4/99    |
    |                               |  Orders 1500000 rows       |              |          |                 |            |            |            |
    +-------------------------------+----------------------------+--------------+----------+-----------------+------------+------------+------------+
    |  Hash Semi-Join (U280)        |  Lineitem 6001215 rows     |    18.914    |  274MHz  |  124.0K/67.6K   |   202/120  |     0/64   |    4/3     |
    |                               |  Orders 1500000 rows       |              |          |                 |            |            |            |
    +-------------------------------+----------------------------+--------------+----------+-----------------+------------+------------+------------+

These are details for benchmark result and usage steps.

.. toctree::
   :maxdepth: 1

   compound_sort.rst
   hash_anti_join.rst
   hash_group_aggregate.rst
   hash_join_v2.rst
   hash_join_v3.rst
   hash_join_v4.rst
   hash_multi_join.rst
   hash_semi_join.rst



Test Overview
--------------

Here are benchmarks of the Vitis Database Library using the Vitis environment. 

.. _l2_vitis_database:

Vitis Database Library
~~~~~~~~~~~~~~~~~~~

* **Download code**

These database benchmarks can be downloaded from `vitis libraries <https://github.com/Xilinx/Vitis_Libraries.git>`_ ``master`` branch.

.. code-block:: bash

   git clone https://github.com/Xilinx/Vitis_Libraries.git 
   cd Vitis_Libraries
   git checkout master
   cd database

* **Setup environment**

Specifying the corresponding Vitis, XRT, and path to the platform repository by running following commands.

.. code-block:: bash

   source <intstall_path>/installs/lin64/Vitis/2021.1_released/settings64.sh
   source /opt/xilinx/xrt/setup.sh
   export PLATFORM_REPO_PATHS=/opt/xilinx/platforms
