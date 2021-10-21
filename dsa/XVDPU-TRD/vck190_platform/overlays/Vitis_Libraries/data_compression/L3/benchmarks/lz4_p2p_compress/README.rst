===================================
LZ4 P2P Application for Compression
===================================

This LZ4 P2P Compress application runs with Xilinx compression and
standard decompression flow. This application gives best kernel 
throughput when multiple files run concurrently on both compute units.


Results
-------

Resource Utilization 
~~~~~~~~~~~~~~~~~~~~~

Table below presents resource utilization of Xilinx LZ4 P2P compress
kernel with 8 engines for single compute unit. It is possible to extend
number of engines to achieve higher throughput.

========== ===== ====== ===== ===== ===== 
Flow       LUT   LUTMem REG   BRAM  URAM 
========== ===== ====== ===== ===== ===== 
Compress   51.7K 14.2K  64.2K 58    48    
---------- ----- ------ ----- ----- ----- 
Packer     10.9K 1.8K   16.7K 16     0    
========== ===== ====== ===== ===== ===== 

Throughput & Compression Ratio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Table below presents the best end to end compress kernel execution with
SSD write throughput achieved with two compute units during execution of
this application.

=========================== ========
Topic                       Results
=========================== ========
Compression Throughput 1.6 GB/s
=========================== ========

Note: Overall throughput can still be increased with multiple compute
units.

Executable Usage
----------------

This application is present in ``L3/benchmarks/lz4_p2p_compress`` directory. Follow build instructions to generate executable and binary.

The binary host file generated is named as "**xil_lz4**" and it is present in ``./build`` directory.

1. To execute single file for compression   : ``./build/xil_lz4 -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/<compress.xclbin> -c <file_name>``
2. To execute multiple files for compression        : ``./build/xil_lz4 -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/compress.xclbin -l <files.list>``

     - ``<files.list>``: Contains various file names with current path

The usage of the generated executable is as follows:

.. code-block:: bash
      
   Usage: application.exe -[-h-c-l-B-p2p] 
          --help,                -h       Print Help Options
          --compress_xclbin,     -cx      Compress XCLBIN                                       Default: [compress]
          --compress,            -c       Compress
          --file_list,           -l       List of Input Files
          --p2p_mod,             -p2p     P2P Mode
          --block_size,          -B       Compress Block Size [0-64: 1-256: 2-1024: 3-4096]     Default: [0]
          --id,                  -id      Device ID                                             Default: [0]
