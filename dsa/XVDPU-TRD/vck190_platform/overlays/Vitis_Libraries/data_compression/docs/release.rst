.. _release_note:

Release Note
============

.. toctree::
   :hidden
   :maxdepth: 1

2021.1
------

Following is the 2021.1 release notes.

* **GZIP Multi Core Compression**
   New GZIP Multi-Core Compress Streaming Accelerator which is purely stream only
   solution (free running kernel), it comes with many variant of different block
   size support of 4KB, 8KB, 16KB and 32KB. 

* **Facebook ZSTD Compression Core**
   New Facebook ZSTD Single Core Compression accelerator with block size 32KB.
   Multi-cores ZSTD compression is in progress (for higher throughput).

* **GZIP Low Latency Decompression**
   A new version of GZIP decompress with improved latency for each block, lesser
   resources (35% lower LUT, 83% lower BRAM) and improved FMax.

* **ZLIB Whole Application Acceleration using U50**
   L3 GZIP solution for U50 Platform, containing 6 Compression core to saturate
   full PCIe bandwidth. It is provided with Efficient GZIP SW Solution to
   accelerate CPU libz.so library which provide seamless Inflate and deflate API
   level integration to end customer software without recompiling. 

* **Versal Platform Supports**


2020.2
------

Following is the 2020.2 release notes.

* **LIBZ Library Acceleration using U50** 

  - Achieved seamless acceleration of libz standard APIs (deflate, compress2 and
    uncompress)
  - Ready to use libz.so library to accelerate any host code without any code change 
  - Provided xzlib standalone executable for for both gzip/zlib compress &
    decompress

* **New ZSTD Decompression**
   Facebook ZSTD decompression implemented

* **New Snappy Dual Core Kernel**
   Google snappy Dual Core Decompression to get 2x throughput for single file
   decompress.
* **New GZIP Compress Kernel**
   Implemented new GZIP Quad Core Compress Kernel (in build , LZ77 , TreeGen,
   Huffman encoder). Reduced overall resource >20%, and reduce 50% DDR bandwidth
   requirement. 
* **New GZIP Compress Streaming Kernel**
   Implemented fully standard compliance GZIP(include header & footer) streaming
   free running kernels.
* **GZIP/ZLIB L3 Application on U50**
   Provided GZIP/ZLIB Application in L3 , optimized for U50 (HBM) and U250. Single
   xclbin supports both zlib & gzip format for compress and uncompress
* **Porting Library to U50**
   Most of Library functions (LZ4, Snappy, GZIP, ZLIB) ported to U50 platform.
* **Low Latency GZIP/ZLIB Decompress**
   Reduced Decompression initial latency from 5K to 2.5K (latency improvement for
   for small block size 4KB/8KB/16KB) 
