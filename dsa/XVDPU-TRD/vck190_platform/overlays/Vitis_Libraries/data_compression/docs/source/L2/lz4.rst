=========================================
Xilinx LZ4 Compression and Decompression
=========================================

LZ4 demo resides in ``L2/demos/lz4`` directory.

Xilinx LZ4 compression/decompression is FPGA based implementation of
standard LZ4. Xilinx implementation of LZ4 application is aimed at
achieving high throughput for both compression and decompression. This
Xilinx LZ4 application is developed and tested on Xilinx Alveo U200. To
know more about standard LZ4 application please refer
https://github.com/lz4/lz4

This application is accelerated using generic hardware architecture for
LZ based data compression algorithms.

Results
-------

Resource Utilization 
~~~~~~~~~~~~~~~~~~~~~

Table below presents resource utilization of Xilinx LZ4 Compress/Decompress
kernels. The final Fmax achieved is 262MHz 

========== ===== ====== ===== ===== ===== 
Flow       LUT   LUTMem REG   BRAM  URAM 
========== ===== ====== ===== ===== ===== 
Compress   47.6K 10.7K  50.7K 56    48    
---------- ----- ------ ----- ----- ----- 
DeCompress 7.3K  1.1K   7K    2     4     
========== ===== ====== ===== ===== ===== 

Performance Data
~~~~~~~~~~~~~~~~

Table below presents kernel throughput achieved for a single compute
unit. 

============================= =========================
Topic                         Results
============================= =========================
Compression Throughput        1.7 GB/s
Decompression Throughput      443 MB/s
Average Compression Ratio     2.13x (Silesia Benchmark)
============================= =========================

Note: Overall throughput can still be increased with multiple compute
units.

Software & Hardware
-------------------

::

     Software: Xilinx Vitis 2021.1
     Hardware: xilinx_u200_xdma_201830_2 (Xilinx Alveo U200)

Executable Usage
----------------
 
1. To execute single file for compression             : ``./build/xil_lz4 -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/<compress_decompress.xclbin> -c <file_name>``
2. To execute single file for decompression           : ``./build/xil_lz4 -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/<compress_decompress.xclbin> -d <file_name.lz4>``
3. To validate single file (compress & decompress)    : ``./build/xil_lz4 -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/<compress_decompress.xclbin> -t <file_name>``
4. To execute multiple files for compression     : ``./build/xil_lz4 -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/<compress_decompress.xclbin> -cfl <files.list>``
5. To execute multiple files for decompression     : ``./build/xil_lz4 -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/<compress_decompress xclbin> -dfl <compressed files.list>``
6. To validate multiple files (compress and decompress) : ``./build/xil_lz4 -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/<compress_decompress xclbin> -l <files.list>``  
           
      - ``<files.list>``: Contains various file names with current path

      - Note: Default arguments are set in Makefile

The usage of the generated executable is as follows:

.. code-block:: bash

   Usage: application.exe -[-h-c-d-t-cfl-dfl-l-xbin-B-id]
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
