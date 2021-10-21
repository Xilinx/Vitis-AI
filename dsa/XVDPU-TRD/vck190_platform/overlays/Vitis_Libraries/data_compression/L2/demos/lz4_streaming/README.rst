==================================================
Xilinx LZ4-Streaming Compression and Decompression
==================================================

LZ4-Streaming example resides in ``L2/demos/lz4_streaming`` directory. To compile and test run this example execute the following commands:

Follow build instructions to generate host executable and binary.

The binary host file generated is named as "**xil_lz4_streaming**", using `PARALLEL_BLOCK` value of 8 (default), and present in ``./build`` directory.

Results
-------

Resource Utilization 
~~~~~~~~~~~~~~~~~~~~~

Table below presents resource utilization of Xilinx LZ4 Streaming 
compress/decompress kernels (excluding data movers). It achieves Fmax of 300MHz.

========== ===== ====== ==== ===== ===== 
Flow       LUT   LUTMem REG  BRAM  URAM  
========== ===== ====== ==== ===== ===== 
Compress   3K    99     3.4K 5     6     
---------- ----- ------ ---- ----- ----- 
DeCompress 6K    370    5K   0     4     
========== ===== ====== ==== ===== ===== 

Performance Data
~~~~~~~~~~~~~~~~

Table below presents kernel throughput achieved for a single compute
unit (Single Engine). 

============================= =========================
Topic                         Results
============================= =========================
Compression Throughput        287 MB/s
Decompression Throughput      1.8 GB/s
Average Compression Ratio     2.23x (Silesia Benchmark)
============================= =========================

Note: Overall throughput can still be increased with multiple compute units.

Executable Usage
~~~~~~~~~~~~~~~

1. To execute single file for compression 	: ``./build/xil_lz4_streaming -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/<compress_decompress_streaming.xclbin> -c <file_name>``
2. To execute single file for decompression	: ``./build/xil_lz4_streaming -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/<compress_decompress_streaming.xclbin> -d <file_name.lz4>``
3. To validate single file (compress & decompress) : ``./build/xil_lz4_streaming -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/<compress_decompress_streaming.xclbin> -t <file_name>``
4. To execute multiple files for compression           : ``./build/xil_lz4_streaming -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/<compress_decompress_streaming.xclbin -cfl <files.list>``
5. To execute multiple files for decompression          : ``./build/xil_lz4_streaming -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/<compress_decompress_streaming.xclbin -dfl <compressed files.list>``   
6. To validate multiple files (compress & decompress)      : ``./build/xil_lz4_streaming -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/<compress_decompress_streaming.xclbin -l <files.list>``  
	
      - ``<files.list>``: Contains various file names with current path

The usage of the generated executable is as follows:

.. code-block:: bash
   
   Usage: application.exe -[-h-c-d-t-cfl-dfl-l-B-id]
          --help,                -h        Print Help Options
          --compress,            -c        Compress
          --decompress,          -d        Decompress
          --test,                -t        Xilinx compress & Decompress
          --compress_list,       -cfl      Compress List of Input Files
          --decompress_list,     -dfl      Decompress List of compressed Input Files
          --test_list,           -l        Xilinx Compress & Decompress on Input Files
          --max_cr,              -mcr      Maximum CR                                            Default: [10]
          --xclbin,              -xbin     XCLBIN
          --device_id,           -id       Device ID                                             Default: [0]
          --block_size,          -B        Compress Block Size [0-64: 1-256: 2-1024: 3-4096]     Default: [0]
