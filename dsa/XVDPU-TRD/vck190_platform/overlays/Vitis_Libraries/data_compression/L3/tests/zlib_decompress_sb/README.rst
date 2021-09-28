======================
Zlib Slave Bridge
======================

This section presents brief introduction about GZip/Zlib slave bridge
decompression application and step by step
guidelines to build and deployment.

Overview
--------

GZip is an Open Source data compression library* which provides
high compression ratio compared to Limpel Ziev based data compression algorithms
(Byte Compression). It applies two levels of compression,

*  Byte Level (Limpel Ziev  LZ Based Compression Scheme)
*  Bit Level (Huffman Entropy)

Due to its high compression ratio it takes higher precedence over LZ based
compression schemes. Traditionally the CPU based solutions are limited to MB/s
speed but there is a high demand for accelerated GZip which provides throughput
in terms of GB/s. 

This demo is aimed at showcasing Xilinx Alveo U250 acceleration of GZip for both
compression and decompression, it also supports Zlib with a host argument switch. 

.. code-block:: bash

   Tested Tool: 2021.1
   Tested XRT:  2021.1
   Tested XSA: xilinx_u50_gen3x16_xdma_201920_3 
   Tested XSA: xilinx_u250_gen3x16_xdma_3_1_202020_1 


Executable Usage
----------------

This application is present under ``L3/tests/zlib_decompress_sb/`` directory. Follow build instructions to generate executable and binary.

The host executable generated is named as "**xil_zlibc**" and it is generated in ``./build`` directory.

Following is the usage of the executable:

1. To execute single file for compression 	          : ``./<build_directory>/xil_zlibc -dx ./<build_directory>/xclbin_<xsa_name>_<TARGET mode>/compress.xclbin -d <input file_name>``
2. To validate multiple files (compress)              : ``./<build_directory>/xil_zlibc -cx ./<build_directory>/xclbin_<xsa_name>_<TARGET mode>/compress.xclbin -l <files.list>``

The usage of the generated executable is as follows:

.. code-block:: bash

   Usage: application.exe -[-h-d-dx-k-id-mcr]

          --help,                    -h        Print Help Options
          --decompress,              -d        DeCompress
          --single_xclbin,           -dx       Single XCLBIN           Default: [single]
          --file_list,               -l        List of Input Files
          --cu,                      -k        CU                      Default: [0]
          --id,                      -id       Device ID               Default: [0]
          --max_cr,                  -mcr      Maximum CR              Default: [10]
===========================================================

