====================
Zlib HBM Application
====================

This section presents brief introduction about Zlib Compress Streaming HBM application and step by step
guidelines to build and deployment.

Overview
--------

ZLIB is an Open Source data compression library* which provides
high compression ratio compared to Limpel Ziev based data compression algorithms
(Byte Compression). It applies two levels of compression,

*  Byte Level (Limpel Ziev  LZ Based Compression Scheme)
*  Bit Level (Huffman Entropy)

Due to its high compression ratio it takes higher precedence over LZ based
compression schemes. Traditionally the CPU based solutions are limited to MB/s
speed but there is a high demand for accelerated ZLIB which provides throughput
in terms of GB/s. 

This demo is aimed at showcasing Xilinx Alveo U50 (HBM Platform) acceleration of ZLIB for
compression.

.. code-block:: bash

   Tested Tool: 2021.1 
   Tested XRT: 2021.1
   Tested XSA: xilinx_u50_gen3x16_xdma_201920_3 


Executable Usage
----------------

This application is present under ``L3/tests/zlib_compress_hbm/`` directory. Follow build instructions to generate executable and binary.

The host executable generated is named as "**xil_zlibc**" and it is generated in ``./build`` directory.

Following is the usage of the executable:

1. To execute single file for compression 	          : ``./<build_directory>/xil_zlibc -xbin ./<build_directory>/xclbin_<xsa_name>_<TARGET mode>/compress.xclbin -c <input file_name>``
2. To validate multiple files for compression              : ``./<build_directory>/xil_zlibc -xbin ./<build_directory>/xclbin_<xsa_name>_<TARGET mode>/compress.xclbin -cfl <files.list>``

	- ``<files.list>``: Contains various file names with current path

The usage of the generated executable is as follows:

.. code-block:: bash
 
   Usage: application.exe -[-h-c-xbin-l-k-id-mcr]
          --help,                -h        Print Help Options
          --compress,            -c        Compress
          --compress_list,       -cfl      Compress List of Input Files
          --max_cr,              -mcr      Maximum CR                                      Default: [10]
          --xclbin,              -xbin     XCLBIN
          --device_id,           -id       Device ID                                       Default: [0]
          --zlib,                -zlib     [0:GZip, 1:Zlib]                                Default: [0]
===========================================================

