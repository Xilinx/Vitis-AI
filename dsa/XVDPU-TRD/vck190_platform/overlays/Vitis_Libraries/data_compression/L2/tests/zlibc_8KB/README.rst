=====================================
Xilinx Zlib Streaming 8KB Compression
=====================================

Zlib example resides in ``L2/tests/zlibc_8KB`` directory. 

Follow build instructions to generate host executable and binary.

The binary host file generated is named as "**xil_zlib**" and it is present in ``./build`` directory.

Executable Usage
----------------

1. To execute single file for compression 	    : ``./build/xil_zlib -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/compress.xclbin -c <file_name> -zlib 1``
2. To execute multiple files for compression    : ``./build/xil_zlib -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/compress.xclbin -cfl <files.list> -zlib 1``

	- ``<files.list>``: Contains various file names with current path

The usage of the generated executable is as follows:

.. code-block:: bash
 
   Usage: application.exe -[-h-c-l-xbin-B]
          --help,           -h      Print Help Options
          --xclbin,         -xbin   XCLBIN                                               Default: [compress]
          --compress,       -c      Compress
          --file_list,      -cfl    Compress List of Input Files
          --max_cr,         -mcr    Maximum CR    
          --device_id,      -id     Device ID                                       Default: [0]
          --zlib,           -zlib   [0:GZip, 1:Zlib]                                Default: [0]

Results
-------

Resource Utilization 
~~~~~~~~~~~~~~~~~~~~~

Table below presents resource utilization of Xilinx Zlib Compress/Decompress
kernels. The final Fmax achieved is 265MHz 


========== ===== ====== ===== ===== ===== 
Flow       LUT   LUTMem REG   BRAM  URAM 
========== ===== ====== ===== ===== ===== 
Compress   20K   1.3K   21.8K 31    12    
========== ===== ====== ===== ===== ===== 

Performance Data
~~~~~~~~~~~~~~~~

Table below presents kernel throughput achieved for a single compute
unit. 

============================= =========================
Topic                         Results
============================= =========================
Compression Throughput        500 MB/s
Average Compression Ratio     2.47x (Silesia Benchmark)
============================= =========================

Standard GZip Support
---------------------

This application is compatible with standard Gzip/Zlib application (compress/decompress).  
