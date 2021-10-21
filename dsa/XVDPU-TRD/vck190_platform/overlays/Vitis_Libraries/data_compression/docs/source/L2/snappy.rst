===========================================
Xilinx Snappy Compression and Decompression
===========================================

Snappy example resides in ``L2/demos/snappy`` directory. 

Xilinx Snappy compression/decompression is FPGA based implementation of
standard Snappy. Xilinx implementation of Snappy application is aimed at
achieving high throughput for both compression and decompression. This
Xilinx Snappy application is developed and tested on Xilinx Alveo U200.
To know more about standard Snappy application please refer
https://github.com/snappy/snappy

This application is accelerated using generic hardware architecture for
LZ based data compression algorithms.

Results
-------

Resource Utilization 
~~~~~~~~~~~~~~~~~~~~~

Table below presents resource utilization of Xilinx Snappy
compress/decompress kernels with 8 engines for single compute unit.
The final Fmax achieved for this design is 299MHz 

========== ===== ====== ===== ===== ===== 
Flow       LUT   LUTMem REG   BRAM  URAM  
========== ===== ====== ===== ===== ===== 
Compress   48K   10.8K  50.6K 48    48    
---------- ----- ------ ----- ----- ----- 
DeCompress 12.4K 2.9K   16.5K 16    4    
========== ===== ====== ===== ===== ===== 

Performance Data
~~~~~~~~~~~~~~~~

Table below presents the kernel throughput achieved with single
compute unit during execution of this application.

============================= =========================
Topic                         Results
============================= =========================
Compression Throughput        1.8 GB/s
Decompression Throughput      1 GB/s
Average Compression Ratio     2.14x (Silesia Benchmark)
============================= =========================

Note: Overall throughput can still be increased with multiple compute
units.

Software & Hardware
-------------------

::

     Software: Xilinx Vitis 2021.1
     Hardware: xilinx_u200_xdma_201830_2 (Xilinx Alveo U200)

Usage
-----

Build Steps
~~~~~~~~~~~

Emulation flows
^^^^^^^^^^^^^^^

::

     make run TARGET=<sw_emu/hw_emu> DEVICE=xilinx_u200_xdma_201830_2
     
     Note: This command compiles for targeted emulation mode and executes the
           application.

Hardware
^^^^^^^^

::

     make all TARGET=hw DEVICE=xilinx_u200_xdma_201830_2

     Note: This command compiles for hardware execution. It generates kernel binary ".xclbin" file. 
           This file is placed in ./build/xclbin*/ directory under Snappy folder.

Executable Usage
----------------
 
1. To execute single file for compression             : ``./build/xil_snappy -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/<compress_decompress.xclbin> -c <file_name>``
2. To execute single file for decompression           : ``./build/xil_snappy -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/<compress_decompress.xclbin> -d <file_name.snappy>``
3. To validate single file (compress & decompress)    : ``./build/xil_snappy -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/<compress_decompress.xclbin> -t <file_name>``
4. To execute multiple files for compression     : ``./build/xil_snappy -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/<compress_decompress.xclbin> -cfl <files.list>``
5. To execute multiple files for decompression     : ``./build/xil_snappy -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/<compress_decompress.xclbin> -dfl <compressed files.list>``
6. To validate multiple files (compress and decompress) : ``./build/xil_snappy -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/<compress_decompress.xclbin> -l <files.list>``  
               
      - ``<files.list>``: Contains various file names with current path

      - Note: Default arguments are set in Makefile

The usage of the generated executable is as follows:

.. code-block:: bash

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
