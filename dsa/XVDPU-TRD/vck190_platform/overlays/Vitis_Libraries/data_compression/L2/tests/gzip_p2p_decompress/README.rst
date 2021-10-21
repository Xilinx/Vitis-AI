===================================
Xilinx GZip-Streaming Decompression
===================================

GZip example resides in ``L2/tests/gzip_p2p_decompress`` directory. 

Follow build instructions to generate host executable and binary.

The binary host file generated is named as "**xil_gzip**" and it is present in ``./build`` directory.

Results
-------

Resource Utilization 
~~~~~~~~~~~~~~~~~~~~

Table below presents resource utilization of Xilinx Zlib Decompress Streaming
kernel. 

========== ===== ====== ==== ===== ===== ======
Flow       LUT   LUTMem REG  BRAM  URAM  Fmax
========== ===== ====== ==== ===== ===== ======
DeCompress 12.3K  226   8.4K   3    2    188MHz
========== ===== ====== ==== ===== ===== ======

Performance Data
~~~~~~~~~~~~~~~~

Table below presents best kernel throughput achieved for a single compute
unit (Single Engine). 

============================= =========================
Topic                         Results
============================= =========================
Best Kernel Throughput        442.48 MB/s
============================= =========================

Note: Overall throughput can still be increased with multiple compute units.


Executable Usage:

1. To execute single file for decompression           : ``./build_dir.<TARGET>/ -dx ./build_dir.<TARGET>/xclbin_<xsa_name>_<TARGET mode>/decompress_stream.xclbin -d <compressed file_name>``

The usage of the generated executable is as follows:

.. code-block:: bash
 
   Usage: application.exe -[-h-d]
        --help,                 -h      Print Help Options   Default: [false]
        --decompress,           -d      Decompress
