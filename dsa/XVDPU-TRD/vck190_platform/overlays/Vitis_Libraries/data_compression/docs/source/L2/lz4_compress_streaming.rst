================================
Xlinx LZ4 Streaming Compression 
================================

LZ4 Compress Streaming example resides in ``L2/tests/lz4_compress_streaming`` directory. 

Follow build instructions to generate host executable and binary.

The binary host file generated is named as **xil_lz4_streaming** and it is present in ``./build`` directory.

Executable Usage
----------------

1. To execute single file for compression             : ``./build/xil_lz4_streaming -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/compress_streaming.xclbin -c <input file_name>``
2. To execute multiple files for compression    : ``./build/xil_lz4_streaming -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/compress_streaming.xclbin -cfl <files.list>``

    - ``<files.list>``: Contains various file names with current path

The usage of the generated executable is as follows:

.. code-block:: bash
       
   Usage: application.exe -[-h-c-cfl-xbin-id]
          --help,                -h        Print Help Options
          --compress,            -c        Compress
          --compress_list,       -cfl      Compress List of Input Files
          --max_cr,              -mcr      Maximum CR                                            Default: [10]
          --xclbin,              -xbin     XCLBIN
          --device_id,           -id       Device ID                                             Default: [0]
          --block_size,          -B        Compress Block Size [0-64: 1-256: 2-1024: 3-4096]     Default: [0]

Resource Utilization 
~~~~~~~~~~~~~~~~~~~~~

Table below presents resource utilization of Xilinx LZ4 Streaming Compression kernels. 
The final Fmax achieved is 300MHz                                                                                                                   

========== ===== ====== ===== ===== ===== 
Flow       LUT   LUTMem REG   BRAM  URAM 
========== ===== ====== ===== ===== ===== 
Compress   3K     124   3.5K   5     6
========== ===== ====== ===== ===== ===== 

Performance Data
~~~~~~~~~~~~~~~~

Table below presents kernel throughput achieved for a single compute
unit. 

============================= =========================
Topic                         Results
============================= =========================
Compression Throughput        290 MB/s
Average Compression Ratio     2.13x (Silesia Benchmark)
============================= =========================
