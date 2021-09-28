====================================
Xlinx Snappy Streaming Decompression 
====================================

Snappy Compress Streaming example resides in ``L2/tests/snappy_dec_streaming_parallelByte8`` directory. 

Follow build instructions to generate host executable and binary.

The binary host file generated is named as **xil_snappy_decompress_streaming** and it is present in ``./build`` directory.

Executable Usage
----------------

1. To execute single file for decompression             : ``./build/xil_snappy_decompress_streaming  -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/decompress_streaming.xclbin -d <input file_name>``
2. To execute multiple files for decompression    : ``./build/xil_snappy_decompress_streaming  -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/decompress_streaming.xclbin -dfl <files.list>``

    - ``<files.list>``: Contains various file names with current path

The usage of the generated executable is as follows:

.. code-block:: bash
       
   Usage: application.exe -[-h-d-dfl-xbin-id]
          --help,                -h        Print Help Options
          --decompress,          -d        Decompress
          --decompress_list,     -dfl      Decompress List of compressed Input Files
          --max_cr,              -mcr      Maximum CR                                            Default: [10]
          --xclbin,              -xbin     XCLBIN
          --device_id,           -id       Device ID                                             Default: [0]
          --block_size,          -B        Compress Block Size [0-64: 1-256: 2-1024: 3-4096]     Default: [0]

Resource Utilization 
~~~~~~~~~~~~~~~~~~~~~

Table below presents resource utilization of Xilinx Snappy Streaming Decompression kernels. 
The final Fmax achieved is 290MHz                                                                                                                   

========== ===== ====== ===== ===== ===== 
Flow       LUT   LUTMem REG   BRAM  URAM 
========== ===== ====== ===== ===== ===== 
Decompress 6.4K  316    5.7K   0     4
========== ===== ====== ===== ===== ===== 

Performance Data
~~~~~~~~~~~~~~~~

Table below presents kernel throughput achieved for a single compute
unit. 

============================= =========================
Topic                         Results
============================= =========================
Decompression Throughput       1.97 GB/s
============================= =========================
