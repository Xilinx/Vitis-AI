.. CompressionLib_Docs documentation master file, created by
   sphinx-quickstart on Thu Jun 20 14:04:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. meta::
   :keywords: Vitis, Library, Data Compression, Xilinx, L1 Modules, Data Compression HLS, ZLIB HLS, LZ4 HLS, Google Snappy HLS , HLS Streams, HLS Dataflow, HLS Pipeline, HLS Unroll, HLS Array Partition
   :description: User Guide of L1 modules under Vitis Data Compression Library

==================
Primitive Overview
==================

The Algorithm library provides a range of primitives for implementing data compression in C++ for Vitis. Headers for these hardware APIs can be found in the ``include`` directory of the package.

Stream-based Interface
``````````````````````
The interface of primitives in this library are mostly HLS streams, with the main input stream along with output stream throughout the dataflow.

Within a HLS dataflow region, all primitives connected via HLS streams can work in parallel resulting in the FPGA acceleration.
