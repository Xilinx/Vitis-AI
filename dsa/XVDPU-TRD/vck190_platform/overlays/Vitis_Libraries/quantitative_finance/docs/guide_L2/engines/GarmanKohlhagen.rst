..
   Copyright 2019 Xilinx, Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. meta::
   :keywords: Garman Kohlhagen, Foreign Exchange, FX, BSM, Black Scholes Merton
   :description: The Garman Kohlhagen model prices Foreign Exchgange (FX) Options based on the Black Scholes Merton model.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


***********************************
Internal Design of Garman Kohlhagen
***********************************

Overview
========

The Garman Kohlhagen model prices Foreign Exchange (FX) Options. It is based on the Black Scholes Merton model, where the BSM interest rate parameter is replaced with the domestic interest rate and the BSM dividend yield parameter is replaced with the foreign interest rate.


Design Structure
================

There are two layers to the kernel; the engine itself and the IO wrapper. 

The Engine
===========================

The engine is simply the :ref:`Black Scholes Merton Engine <black_scholes_merton_engine>`
The engine performs a single Black Scholes Merton Closed Form solution for a European Call.

IO Wrapper (gk_kernel.cpp)
==========================

The kernel is the HLS wrapper level which implements the pipelining and parallelization to allow high throughput. The kernel uses a dataflow methodology with streams to pass the data through the design.

The top level in and out ports are 512 bit wide, designed to match the whole DDR bus width and allowing vector access. In the case of float data type (4 bytes), sixteen parameters can be accessed from the bus in parallel. Each port is connected to its own AXI master with arbitration handled by the AXI switch and DDR controller under the hood.

These ports are interfaced via functions in bus_interface.hpp which convert between the wide bus and a template number of streams. Once input stream is formed, each stream is passed to a separate instance of the engine. The engine is wrapped inside bsm_stream_wrapper() which handles the stream processing. Here the initiation interval (II) and loop unrolling is controlled. One cfBSMEngine engine is instanced per stream allowing for parallel processing of multiple parameter sets. Additionally, the engines are in an II=1 loop, so that each engine can produce one price and its associated Greeks on each cycle.

This wrapper also handles the mapping between the Garman-Kohlhagen parameters ('domestic interest rate' and 'foreign interest rate') to the BSM parameters ('risk free interest rate' and 'dividend yield').

Resource Utilization
====================

The floating point kernel Area Information:

:FF:         82849 (10% of SLR on u200 board)  
:LUT:        70748 (17% of SLR on u200 board)   
:DSP:        524 (22% of SLR on u200 board)
:BRAM:       408 (28% of SLR on u200 board)
:URAM:       0


Throughput
==========

Throughput is composed of two processes: transferring data to/from the FPGA and running the computations. The demo contains options to measure timings as described in the README.md file.

As an example, processing a batche of 16384 call calculations with a floating point kernel breaks down as follows:

Total time (memory transfer time plus calculation time) = 974us

Calculation time (kernel execution time) = 235us

Memory transfer time = 739us



.. toctree::
   :maxdepth: 1
