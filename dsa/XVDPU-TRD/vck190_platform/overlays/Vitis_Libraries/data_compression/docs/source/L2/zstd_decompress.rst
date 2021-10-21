=========================================
Xilinx ZSTD Decompression
=========================================

ZSTD example resides in ``L2/demos/zstd_decompress`` directory. 

Follow build instructions to generate host executable and binary.

The binary host file generated is named as "**xil_zstd**" and it is present in ``./build`` directory.



Results
-------

Overall Resource Utilization 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Table below presents resource utilization of Xilinx Zstd decompress kernel supporting Window Size of 32KB. The final Fmax achieved is 252MHz.

========== ===== ====== ===== ===== ===== 
Flow       LUT   LUTMem REG   BRAM  URAM  
========== ===== ====== ===== ===== ===== 
DeCompress 21K   456    17K   33    6    
---------- ----- ------ ----- ----- -----
GzipMM2S   1.4K  295    1.6K  1     0  
---------- ----- ------ ----- ----- -----
GzipS2MM   2.7K  630K   4K    2     0
========== ===== ====== ===== ===== ===== 


Performance Data
~~~~~~~~~~~~~~~~

+----------------------------+------------------------+
| Topic                      | Kernel Throughput      |
+============================+========================+
| Decompression              | 463 MB/s               |
+----------------------------+------------------------+

.. [*] Decompression uses feasibile options (Bitwidth: 32bit, Window Size: 32KB) 

Executable Usage:

1. To execute single file for decompression           : ``./build/xil_zlib -dx ./build/xclbin_<xsa_name>_<TARGET mode>/compress_decompress.xclbin -d <compressed file_name>``
2. To decompress multiple files                       : ``./build/xil_zlib -dx ./build/xclbin_<xsa_name>_<TARGET mode>/compress_decompress.xclbin -l <files.list>``

	- ``<files.list>``: Contains various file names with current path

The usage of the generated executable is as follows:

.. code-block:: bash
 
   Usage: application.exe -[-h-d-sx-l]
        --help,                 -h      Print Help Options   Default: [false]
        --decompress,           -d      Decompress
        --decompress_xclbin,    -dx     Decompress XCLBIN
        --file_list,            -l      List of Input Files

