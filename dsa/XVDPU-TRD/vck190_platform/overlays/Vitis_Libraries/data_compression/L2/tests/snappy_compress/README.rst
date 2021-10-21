================================
Xlinx Snappy Compression 
================================

Snappy Compress Streaming example resides in ``L2/tests/snappy_compress`` directory. 

Follow build instructions to generate host executable and binary.

The binary host file generated is named as **xil_snappy** and it is present in ``./build`` directory.

Executable Usage
----------------

1. To execute single file for compression             : ``./build/xil_snappy -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/compress.xclbin -c <input file_name>``
2. To execute multiple files for compression    : ``./build/xil_snappy -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/compress.xclbin -cfl <files.list>``

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

Table below presents resource utilization of Xilinx Snappy Streaming Compression kernels. 
The final Fmax achieved is 300MHz                                                                                                                   

========== ===== ====== ===== ===== ===== 
Flow       LUT   LUTMem REG   BRAM  URAM 
========== ===== ====== ===== ===== ===== 
Compress   52K    10.8K  57K   48    48
========== ===== ====== ===== ===== ===== 

Performance Data
~~~~~~~~~~~~~~~~

Table below presents kernel throughput achieved for a single compute
unit. 

============================= =========================
Topic                         Results
============================= =========================
Compression Throughput        2.2 GB/s
Average Compression Ratio     2.13x (Silesia Benchmark)
============================= =========================
