====================
GZip HBM Application
====================

This section presents brief introduction about GZip HBM Bandwidth application and step by step
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

This demo is aimed at showcasing Xilinx Alveo U50 (HBM Platform) acceleration of GZip for both
        compression and decompression, it also supports Zlib using a host argument switch. 

        .. code-block:: bash

           Tested Tool: 2020.2
           Tested XRT:  2020.2
           Tested XSA:  xilinx_u50_gen3x16_xdma_201920_3 


Executable Usage
----------------

This application is present under ``L3/benchmarks/gzip_hbm_bandwidth/`` directory. Follow build instructions to generate executable and binary.

The host executable generated is named as "**xil_gzip_8b**" and it is generated in ``./build`` directory.

Following is the usage of the executable:

1. To execute single file for compression                      : ``./build/xgzip -sx ./build/xclbin_<xsa_name>_<TARGET mode>/compress_decompress.xclbin -c <input file_name>``
2. To execute single file for decompression                    : ``./build/xgzip -sx ./build/xclbin_<xsa_name>_<TARGET mode>/compress_decompress.xclbin -d <compressed file_name>``

The usage of the generated executable is as follows:

.. code-block:: bash
 
   Usage: application.exe -[-h-c-d-sx-p-id]
          --help,              -h       Print Help Options
          --compress,          -c       Compress
          --decompress,        -d       DeCompress
          --single_xclbin,     -sx      Single XCLBIN          Default: [single]
          --max_cr,            -mcr     Maximum CR             Default: [20]
          --multi_process,     -p       Multiple Process       Default: [1]
          --id,                -id      Device ID              Default: [0]
===========================================================
