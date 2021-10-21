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
   :keywords: Cox-Ross-Rubinstein, Binomial
   :description: The Cox-Ross-Rubinstein Binomial Tree method is a numerical implementation of the assumptions in the Black-Scholes financial model. 
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


****************************************************
Internal Design of Cox-Ross-Rubinstein Binomial Tree
****************************************************

Overview
========

The Cox-Ross-Rubinstein Binomial Tree method is a numerical implementation of the assumptions in the Black-Scholes financial model. The detail is described in the "Numerical Methods" section here.

The equations for obtaining the option price can be found online (see for example https://en.wikipedia.org/wiki/Binomial_options_pricing_model) and will not be reproduced here. 


Design Structure
================

The heart of the implementation is the Binomial tree engine which implements the model. 
It takes one set of input parameters and a flag to control which type of option (American/European Call or Put) should be calculated. It then returns the option price.

These input parameters are in the form of a structure that contains:

- S - Stock price
- K - Strike price
- T - Expiration time
- rf - Risk free interest rate
- V - Volatility
- q - Dividend yield
- N - Height of the binomial tree

This function is mostly standard C++ with some exception of using the HLS maths library to replace it.
Layered on top of the design is the HLS specific kernel wrapper which is responsible for gathering the input data sets (from DDR or HBM for example), converting them to parallel streams and passing them into the kernel. 
It then writes the results back out. This level is where the HLS #pragmas are used to control the amount of pipelining and unrolling.


bt_engine (bt_engine.hpp)
=======================

The code is an implementation of the Cox, Ross, & Rubinstein (CRR) method and is template to accept different data types (float/double). 
It uses standard C++ and allows the code to be easily used in a software only environment by swapping to the standard math namespace. 

The implementation is broken into a number of steps:

- Calculation of the option at each final node i.e. at the time of expiration
- Sequential calculation of the option value at each preceding node (working backwards through the tree towards the valuation)
- Calculation of the early exercise (in the case of the American option only) at each stage.

There are some optimizations to the algorithm for the FPGA to allow for parallelization, i.e to obtain an II value of 1 for each loop; the generated report shows:

Pipelining function 'pow_generic<double>'.
Pipelining result : Target II = 1, Final II = 1, Depth = 89.
Pipelining loop 'Loop 1'.
Pipelining result : Target II = 1, Final II = 1, Depth = 112.
Pipelining loop 'Loop 2'.
Pipelining result : Target II = 1, Final II = 1, Depth = 5.
Pipelining loop 'Loop 3.1'.
Pipelining result : Target II = 1, Final II = 1, Depth = 19.
Pipelining loop 'Loop 3.2'.
Pipelining result : Target II = 1, Final II = 1, Depth = 117.
Pipelining loop 'Loop 1'.
Pipelining result : Target II = 1, Final II = 1, Depth = 3.
Pipelining loop 'Loop 3'.
Pipelining result : Target II = 1, Final II = 1, Depth = 3.
Finished kernel compilation




binomialtreekernel (binomialtreekernel.cpp)
===========================================

The kernel is the HLS wrapper level which implements the pipelining and parallelization to allow high throughput. The kernel uses a dataflow methodology to pass the data through the design.

The top level's input and output ports are 512 bit wide, which is designed to match the whole DDR bus width and allowing vector access. In the case of float data type (4 bytes), sixteen parameters can be accessed from the bus in parallel. Each port is connected to its own AXI master with arbitration handled by the AXI switch and DDR controller under the hood.


Resource Utilization
====================
 
========================== ============ ============ ============ ============ ============= =============
  Name                      LUT          LUTAsMem     REG          BRAM         URAM          DSP        
========================== ============ ============ ============ ============ ============= =============
  Platform                   161585       19362        234662       320          0             7          
                              [13.67%]     [ 3.27%]     [ 9.92%]     [14.81%]     [ 0.00%]      [ 0.10%]  
  User Budget                1020655      572478       2129818      1840         960           6833       
                              [100.00%]    [100.00%]    [100.00%]    [100.00%]    [100.00%]     [100.00%] 
     Used Resources          44393        4269         49900        125          0             446        
                              [  4.35%]    [  0.75%]    [  2.34%]    [  6.79%]    [  0.00%]     [  6.53%] 
     Unused Resources        976262       568209       2079918      1715         960           6387       
                              [ 95.65%]    [ 99.25%]    [ 97.66%]    [ 93.21%]    [100.00%]     [ 93.47%] 
     BinomialTreeKernel_1    44393        4269         49900        125          0             446        
                              [  4.35%]    [  0.75%]    [  2.34%]    [  6.79%]    [  0.00%]     [  6.53%] 
========================== ============ ============ ============ ============ ============= =============

The hardware resources are listed in the table above. This is for the demonstration as configured by default (one engine), achieving a 300 MHz clock rate.

The number of engines in a build may be configured by the user.  For an example build of eight engines, the following table shows the resources used:

========================== ============ ============ ============ ============ ============= =============
  Name                      LUT          LUTAsMem     REG          BRAM         URAM          DSP        
========================== ============ ============ ============ ============ ============= =============
  Platform                   161579       19362        234660       320          0             7          
                              [13.67%]     [ 3.27%]     [ 9.92%]     [14.81%]     [ 0.00%]      [ 0.10%]  
  User Budget                1020661      572478       2129820      1840         960           6833       
                              [100.00%]    [100.00%]    [100.00%]    [100.00%]    [100.00%]     [100.00%] 
     Used Resources          334087       33438        355699       559          0             3568       
                              [ 32.73%]    [  5.84%]    [ 16.70%]    [ 30.38%]    [  0.00%]     [ 52.22%] 
     Unused Resources        686574       539040       1774121      1281         960           3265       
                              [ 67.27%]    [ 94.16%]    [ 83.30%]    [ 69.62%]    [100.00%]     [ 47.78%] 
     BinomialTreeKernel_1    334087       33438        355699       559          0             3568       
                              [ 32.73%]    [  5.84%]    [ 16.70%]    [ 30.38%]    [  0.00%]     [ 52.22%] 
========================== ============ ============ ============ ============ ============= =============



Throughput
==========

The demo application Makefile has a check target option which can be used to verify the output from the Binomial tree Kernel compared to CPU/Quantlib and the throughput.

For a 1 engine kernel with a tree height of 1024 we obtain a throughput of approximately 0.7K option calculations per second. 

For a 4 engine kernel with a tree height of 1024 we obtain a throughput of approximately 2.7K option calculations per second.

Both these values are obtained when calculating 49 options (i.e. the stock and volatility test grid). The values are the same, whether European or American option prices are being calculated. 

.. toctree::
   :maxdepth: 1
