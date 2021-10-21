=====================================
LZ4 P2P Application for Decompression
=====================================

This LZ4 P2P Decompress application runs with standard compression and
xilinx decompression flow. This application gives best kernel 
throughput when multiple files run concurrently on both compute units.

Results
-------

Resource Utilization 
~~~~~~~~~~~~~~~~~~~~~

Table below presents resource utilization of Xilinx LZ4 P2P Decompress
kernel with 8 engines for single compute unit. It is possible to extend
number of engines to achieve higher throughput.

========== ===== ====== ===== ===== ===== 
Flow       LUT   LUTMem REG   BRAM  URAM 
========== ===== ====== ===== ===== ===== 
Decompress   34.9K 14.2K  45.1K  145   0    
---------- ----- ------ ----- ----- ----- 
Packer     10.6K  435   13.6K  15     0    
========== ===== ====== ===== ===== ===== 

Throughput & Decompression Ratio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Table below presents the best end to end decompress kernel execution with
SSD write throughput achieved with two compute units during execution of
this application.

=========================== ========
Topic                       Results
=========================== ========
Decompression Throughput 2.5 GB/s
=========================== ========

Note: Overall throughput can still be increased with multiple compute
units.

Executable Usage
----------------

This application is present in ``L3/benchmarks/lz4_p2p_decompress`` directory. Follow build instructions to generate executable and binary.

The binary host file generated is named as "**xil_lz4**" and it is present in ``./build`` directory.

1. To execute single file for decompression   : ``./build/xil_lz4 -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/<decompress.xclbin> -c <file_name>``
2. To execute multiple files for decompression        : ``./build/xil_lz4 -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/decompress.xclbin -l <files.list>``
     - ``<files.list>``: Contains various file names with current path

The usage of the generated executable is as follows:

.. code-block:: bash
         
   Usage: application.exe -[-h-d-l-B-p2p] 
          --help,                -h       Print Help Options
          --decompress_xclbin,   -dx      Decompress XCLBIN                                       Default: [decompress]
          --decompress,          -d       Decompress
          --file_list,           -l       List of Input Files
          --p2p_mod,             -p2p     P2P Mode
          --block_size,          -B       Decompress Block Size [0-64: 1-256: 2-1024: 3-4096]     Default: [0]
          --id,                  -id      Device ID                                             Default: [0]    
