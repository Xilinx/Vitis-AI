.. CompressionLib_Docs documentation master file, created by
   sphinx-quickstart on Thu Jun 20 14:04:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. meta::
   :keywords: Vitis, Library, Data Compression, Xilinx
   :description: Vitis Data Compression Library top level overview
   :xlnxdocumentclass: Document
   :xlnxdocumenttypes: Tutorials


Vitis Data Compression Library
==============================

Vitis Data Compression library is an open-sourced data compression library written
in C++ for accelerating data compression applications in a variety of
use cases. The library covers two levels of acceleration: the module level
and the pre-defined kernel level, and will evolve to offer the third
level as pure software APIs working with pre-defined hardware overlays.

-  **L1**: Module level, it provides optimized hardware implementation of
   the core LZ based and data compression specific modules like lz4
   compress and snappy compress.
-  **L2**: Kernel level, a demo on lz4, snappy, zlib and zstd data compression
   algorithms are shown via kernel which internally uses the optimized
   hardware modules.
-  **L3**: The software API level will wrap the details of offloading
   acceleration with prebuilt binary (overlay) and allow users to
   accelerate data compression tasks on Alveo cards without hardware
   development.

Advanced users can easily tailor, optimize or
combine with property logic at any levels as all the kernel code is developed in HLS C++ with the permissive
Apache 2.0 license. Demos of different data
compression acceleration are also provided with the library for easy
on-boarding.


.. toctree::
   :maxdepth: 1
   :caption: Library Overview
   
   source/overview.rst
   release.rst

.. toctree::
	:maxdepth: 2
	:caption: User Guide

	source/usecase.rst
	source/L1/L1.rst
	source/L2/L2.rst
	source/L3/L3.rst

.. toctree::
	:caption: Benchmark Results
	:maxdepth: 1
	
	benchmark.rst
