====================
GZip Application
====================

This section presents brief introduction about GZip application and step by step
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
   Tested XSA:  xilinx_u250_gen3x16_xdma_3_1_202020_1


Executable Usage
----------------

This application is present under ``L3/demos/gzip_app/`` directory. Follow build instructions to generate executable and binary.

The host executable generated is named as "**xil_gzip**" and it is generated in ``./build`` directory.

Following is the usage of the executable:

1. To execute single file for compression 	          : ``./build/xil_gzip -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/compress_decompress.xclbin -c <input file_name>``
2. To execute single file for decompression           : ``./build/xil_gzip -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/compress_decompress.xclbin -d <compressed file_name>``
3. To validate single file (compress & decompress)    : ``./build/xil_gzip -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/compress_decompress.xclbin -t <input file_name>``
4. To execute multiple files for compression          : ``./build/xil_gzip -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/compress_decompress.xclbin -cfl <files.list>``
5. To execute multiple files for decompression        : ``./build/xil_gzip -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/compress_decompress.xclbin -dfl <compressed files.list>``
6. To validate multiple files (compress & decompress) : ``./build/xil_gzip -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/compress_decompress.xclbin -l <files.list>``

	- ``<files.list>``: Contains various file names with current path

The default design flow is GZIP design to run the ZLIB, enable the switch ``-zlib`` in the command line, as mentioned below:
``./build/xil_gzip -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/compress_decompress.xclbin -c <input file_name> -zlib 1``

The -xbin option mentioned above is optional, you can provide path to your binary file using -xbin option otherwise it will by default map to ``./build/xclbin_<xsa_name>_<TARGET mode>/compress_decompress.xclbin`` 


The usage of the generated executable is as follows:

.. code-block:: bash

   Usage: application.exe -[-h-c-d-xbin-t-l-id-mcr]
          --help,                -h        Print Help Options
          --compress,            -c        Compress
          --decompress,          -d        Decompress
          --test,                -t        Xilinx compress & Decompress
          --compress_list,       -cfl      Compress List of Input Files
          --decompress_list,     -dfl      Decompress List of compressed Input Files
          --test_list,           -l        Xilinx Compress & Decompress on Input Files
          --max_cr,              -mcr      Maximum CR                                      Default: [10]
          --xclbin,              -xbin     XCLBIN
          --device_id,           -id       Device ID                                       Default: [0]
          --zlib,                -zlib     [0:GZip, 1:Zlib]                                Default: [0]
===========================================================

