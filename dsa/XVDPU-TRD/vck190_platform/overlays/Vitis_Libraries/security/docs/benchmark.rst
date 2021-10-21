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

.. meta::
   :keywords: Vitis, Security, Library, Vitis Security design, benchmark, result
   :description: Vitis Security Library benchmark results.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


.. result:

*********
Benchmark
*********


+------------------+-----------+----------------+----------------+--------------+-------+----------+-------------+
|  API             | Frequency |     LUT        |     REG        |    BRAM      |  URAM |   DSP    | Throughput  |
+------------------+-----------+----------------+----------------+--------------+-------+----------+-------------+
| aes256CbcDecrypt |  286MHz   |    203,595     |    312,900     |     761      |   0   |    29    |   4.7GB/s   |
+------------------+-----------+----------------+----------------+--------------+-------+----------+-------------+
| aes256CbcEncrypt |  224MHz   |   1,059,093    |    1,010,145   |     654      |   0   |    152   |   5.5GB/s   |
+------------------+-----------+----------------+----------------+--------------+-------+----------+-------------+
| rc4              |  147MHz   |   1,126,259    |    1,120,505   |     640      |   0   |    216   |   3.0GB/s   |
+------------------+-----------+----------------+----------------+--------------+-------+----------+-------------+
| hmacSha1         |  227MHz   |    959,078     |    1,794,522   |     777      |   0   |    56    |   8.0 GB/s  |
+------------------+-----------+----------------+----------------+--------------+-------+----------+-------------+
| crc32            |  300MHz   |    5,322       |    10,547      |     16       |   0   |    0     |   4.7 GB/s  |
+------------------+-----------+----------------+----------------+--------------+-------+----------+-------------+
| adler32          |  262MHz   |    6,348       |    12,232      |     16       |   0   |    0     |   4.1 GB/s  |
+------------------+-----------+----------------+----------------+--------------+-------+----------+-------------+


These are details for benchmark result and usage steps.

.. toctree::
   :maxdepth: 1

   ../guide_L1/benchmark/aes256CbcDecrypt.rst
   ../guide_L1/benchmark/aes256CbcEncrypt.rst
   ../guide_L1/benchmark/hmacSha1.rst
   ../guide_L1/benchmark/rc4.rst
   ../guide_L1/benchmark/crc32.rst
   ../guide_L1/benchmark/adler32.rst

Test Overview
--------------

Here are benchmarks of the Vitis Security Library using the Vitis environment,


.. _l1_vitis_security: 

Vitis Security Library
~~~~~~~~~~~~~~~~~~~~~~~

* **Download code**

These solver benchmarks can be downloaded from `vitis libraries <https://github.com/Xilinx/Vitis_Libraries.git>`_ ``master`` branch.

.. code-block:: bash

   git clone https://github.com/Xilinx/Vitis_Libraries.git
   cd Vitis_Libraries
   git checkout master
   cd security

* **Setup environment**

Specifying the corresponding Vitis, XRT, and path to the platform repository by running following commands.

.. code-block:: bash

   source /opt/xilinx/Vitis/2021.1/settings64.sh
   source /opt/xilinx/xrt/setup.sh
   export PLATFORM_REPO_PATHS=/opt/xilinx/platforms
