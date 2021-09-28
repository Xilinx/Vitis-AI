===============
LZ4 Application
===============

LZ4 data compression application falls under Limpel Ziev based byte compression
scheme. It is widely known for achieving decompression throughput >GB/s on
high end single core high end CPU. 

This demo presents usage of FPGA accelerated LZ4 compression &
decompression which achieves throughput >GB/s and this application is scalable.


Executable Usage
----------------

This application is present in ``L3/demos/lz4_app`` directory. Follow build instructions to generate executable and binary.

The binary host file generated is named as "**xil_lz4**" and it is present in ``./build`` directory.

1. To execute single file for compression 	: ``./build/xil_lz4 -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/<compress_decompress.xclbin> -c <file_name>``
2. To execute single file for decompression	: ``./build/xil_lz4 -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/<compress_decompress.xclbin> -d <file_name.lz4>``
3. To validate single file (compress & decompress) : ``./build/xil_lz4 -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/compress_decompress.xclbin -t <input file_name>``
4. To execute multiple files for compression        : ``./build/xil_lz4 -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/compress_decompress.xclbin -cfl <files.list>``
5. To execute multiple files for decompression        : ``./build/xil_lz4 -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/compress_decompress.xclbin -dfl <compressed files.list>``   
6. To validate multiple files files (compress & decompress)		: ``./build/xil_lz4 -xbin ./build/xclbin_<xsa_name>_<TARGET mode>/<compress_decompress.xclbin> -l <files.list>``
	
	- ``<files.list>``: Contains various file names with current path

The usage of the generated executable is as follows:

.. code-block:: bash
   
   Usage: application.exe -[-h-c-l-d-B-x]         
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
