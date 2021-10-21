====================================
Xilinx GZip Streaming Decompression
====================================

GZip example resides in ``L2/tests/gzip_decompress`` directory. 

Follow build instructions to generate host executable and binary.

The binary host file generated is named as "**xil_gzip**" and it is present in ``./build`` directory.

Executable Usage
----------------

1. To execute single file for compression 	          : ``./build/xil_gzip -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/decompress.xclbin  -d <file_name>``
2. To execute multiple files for compression    : ``./build/xil_gzip -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/decompress.xclbin  -dfl <files.list>``

	- ``<files.list>``: Contains various file names with current path

The usage of the generated executable is as follows:

.. code-block:: bash
 
   Usage: application.exe -[-h-d-dfl-xbin-B]
          --help,                -h        Print Help Options
          --decompress,          -d        Decompress
          --decompress_list,     -dfl      Decompress List of compressed Input Files
          --max_cr,              -mcr      Maximum CR                                      Default: [10]
          --xclbin,              -xbin     XCLBIN
          --device_id,           -id       Device ID                                       Default: [0]
          --zlib,                -zlib     [0:GZip, 1:Zlib]                                Default: [0]
 
Results
-------

Resource Utilization 
~~~~~~~~~~~~~~~~~~~~~

Table below presents resource utilization of Xilinx GZip Compress/Decompress
kernels. The final Fmax achieved is 240MHz 

========== ===== ====== ===== ===== ===== 
Flow       LUT   LUTMem REG   BRAM  URAM 
========== ===== ====== ===== ===== ===== 
Decompress 9.8K   186    7K    6    2
---------- ----- ------ ----- ----- -----
GzipMM2S   1.5K   281    2K    2    0   
---------- ----- ------ ----- ----- -----
GzipS2MM   2.7K   630    4K    2    0
========== ===== ====== ===== ===== ===== 


Performance Data
~~~~~~~~~~~~~~~~

Table below presents kernel throughput achieved for a single compute
unit. 

============================= =========================
Topic                         Results
============================= =========================
Decompression Throughput       450 MB/s
============================= =========================

Standard GZip Support
---------------------

This application is compatible with standard Gzip/Zlib application (compress/decompres).  
