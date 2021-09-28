Vitis Data Compression Library
==============================

Vitis Data Compression library is an open-sourced Vitis library written
in C++ for accelerating data compression applications in a variety of
use cases. The library covers two levels of acceleration: the module level
and the pre-defined kernel level, and will evolve to offer the third
level as pure software APIs working with pre-defined hardware overlays.

-  L1: Module level, it provides optimized hardware implementation of
   the core LZ based and data compression specific modules like lz4
   compress and snappy compress.
-  L2: Kernel level, a demo on lz4, snappy and zlib data compression
   algorithms are shown via kernel which internally uses the optimized
   hardware modules.
-  L3: The software API level will wrap the details of offloading
   acceleration with prebuilt binary (overlay) and allow users to
   accelerate data compression tasks on Alveo cards without hardware
   development.

Advanced users can easily tailor, optimize or
combine with property logic at any levels as all the kernel code is developed in HLS C++ with the permissive
Apache 2.0 license. Demos of different data
compression acceleration are also provided with the library for easy
on-boarding.

Requirements
------------

Software Platform
~~~~~~~~~~~~~~~~~

This library is designed to work with Vitis 2021.1 and later, and
therefore inherits the system requirements of Vitis and XRT.

Supported operating systems are RHEL/CentOS 7.4, 7.5 and Ubuntu 16.04.4
LTS, 18.04.1 LTS. With CentOS/RHEL 7.4 and 7.5, C++11/C++14 should be
enabled via
`devtoolset-6 <https://www.softwarecollections.org/en/scls/rhscl/devtoolset-6/>`__.

PCIE Accelerator Card
~~~~~~~~~~~~~~~~~~~~~

Hardware modules and kernels are designed to work with 16nm Alveo cards.
\* `Alveo
U280 <https://www.xilinx.com/products/boards-and-kits/alveo/u280.html#gettingStarted>`__
\* `Alveo
U250 <https://www.xilinx.com/products/boards-and-kits/alveo/u250.html#gettingStarted>`__
\* `Alveo
U200 <https://www.xilinx.com/products/boards-and-kits/alveo/u200.html#gettingStarted>`__

Shell Environment
~~~~~~~~~~~~~~~~~

Setup the build environment using the Vitis and XRT scripts:

::

       $ source <install path>/Vitis/2021.1/settings64.sh
       $ source /opt/xilinx/xrt/setup.sh
       $ export PLATFORM_REPO_PATHS=/opt/xilinx/platforms

Setting the PLATFORM_REPO_PATHS to installation folder of platform files
can enable makefiles in this library to use DEVICE variable as a
pattern. Otherwise, full path to .xpfm file needs to be provided via
DEVICE variable.

Benchmark Result
----------------

By offloading compression to FPGA, we have achieved 19.3x speedup using single GZIP
compress kernel against single core CPU Zlib fast (1.2.11, -1) and a 2x speedup
achieved using single GZIP decompress kernel against single core CPU Zlib fast
(1.2.11, -1).

Dataset
~~~~~~~
Benchmark evaluation of compression performance is of reference `Silesia Corpus.
<http://sun.aei.polsl.pl/~sdeor/index.php?page=silesia>`__


Compression
~~~~~~~~~~~

Tables below showcases throughput details of compression for various Alveo accelerated data compression algorithms.

+------------------------------------------------------------------------+----------------------+-------------------+----------+---------+-------+-------+
| Architecture                                                           |  Compression Ratio   |     Throughput    |  FMax    |  LUT    |  BRAM |  URAM |
+========================================================================+======================+===================+==========+=========+=======+=======+
| `LZ4 Streaming <L2/tests/lz4_compress_streaming>`_                     |        2.13          |      290 MB/s     |  300MHz  |  3K     |  5    |  6    |
+------------------------------------------------------------------------+----------------------+-------------------+----------+---------+-------+-------+
| `Snappy Streaming <L2/demos/snappy_streaming>`_                        |        2.13          |      290 MB/s     |  300MHz  |  3K     |  4    |  6    |
+------------------------------------------------------------------------+----------------------+-------------------+----------+---------+-------+-------+
| `GZip/Zlib Memory Mapped <L2/tests/gzipc_block_mm>`_                   |        2.67          |      2 GB/s       |  285MHz  |  52K    |  67   |  72   |
+------------------------------------------------------------------------+----------------------+-------------------+----------+---------+-------+-------+
| `GZip/Zlib Compress Stream <L2/tests/gzipc>`_                          |        2.67          |      2 GB/s       |  290MHz  |  48.5K  |  61   |  72   |
+------------------------------------------------------------------------+----------------------+-------------------+----------+---------+-------+-------+
| `GZip/Zlib Fixed Compress Stream <L2/tests/gzipc_static>`_             |        2.25          |      2 GB/s       |  300MHz  |  33.2K  |  45   |  64   |
+------------------------------------------------------------------------+----------------------+-------------------+----------+---------+-------+-------+

.. [*] LZ4 Streaming and Snappy Streaming: Uses Single Engine and Datawidth 8-bit
.. [*] LZ4 Memory Mapped and Snappy Memory Mapped: Uses 8-Engines with Data Movers
.. [*] GZip/Zlib Memory Mapped and GZip/Zlib Compress Stream: Uses 8-Engines with Data Movers and supports Dynamic Huffman


Decompression
~~~~~~~~~~~~~

Tables below showcases throughput details of decompression for various Alveo accelerated data compression algorithms.

+----------------------------------------------------------------------+-------------------+----------+---------+-------+------+
| Architecture                                                         |    Throughput     |  FMax    |  LUT    |  BRAM | URAM |           
+======================================================================+===================+==========+=========+=======+======+
| `LZ4 Streaming <L2/tests/lz4_dec_streaming_parallelByte8>`_          |     1.8  GB/s     |  284MHz  |  5.5K   |  0    |  4   |
+----------------------------------------------------------------------+-------------------+----------+---------+-------+------+
| `Snappy Streaming <L2/tests/snappy_dec_streaming_parallelByte8>`_    |     1.97 GB/s     |  290MHz  |  6.4K   |  0    |  4   |
+----------------------------------------------------------------------+-------------------+----------+---------+-------+------+
| `GZip/Zlib Streaming <L2/demos/gzip>`_                               |     518  MB/s     |  273MHz  |  6.9K   |  0    |  2   |
+----------------------------------------------------------------------+-------------------+----------+---------+-------+------+
| `ZStd Streaming <L2/demos/zstd_decompress>`_                         |     463  MB/s     |  252MHz  |  21K    |  33   |  6   |
+----------------------------------------------------------------------+-------------------+----------+---------+-------+------+
| `ZStd Full File Streaming <L2/demos/zstd_decompress>`_               |     463  MB/s     |  252MHz  |  21K    |  33   |  6   |
+----------------------------------------------------------------------+-------------------+----------+---------+-------+------+

.. [*] The amount of resources used indicate that we still have room on Alveo U200 to go for more compute units which can further improve the throughput.
.. [*] LZ4 Streaming and Snappy Streaming: Uses Single Engine and Datawidth 64-bit
.. [*] LZ4 Memory Mapped and Snappy Memory Mapped: Uses 8-Engines with Data Movers
.. [*] GZip/Zlib Streaming: Full standard support (Dynamic Huffman, Fixed Huffman and Stored Blocks supported) and data width 64-bit.
.. [*] ZStd Streaming: Uses single engine with datawidth 32-bit and full Standard support with limited Window Size upto 128KB.


LICENSE
-------

Licensed using the `Apache 2.0
license. <https://www.apache.org/licenses/LICENSE-2.0>`__

::

   Copyright 2019-2021 Xilinx, Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   Copyright 2019-2021 Xilinx, Inc.

Contribution/Feedback
---------------------

Welcome! Guidelines to be published soon.
