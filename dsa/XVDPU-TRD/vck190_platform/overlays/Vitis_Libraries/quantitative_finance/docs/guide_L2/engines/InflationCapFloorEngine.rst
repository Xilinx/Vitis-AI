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
   :keywords: Inflation CapFloor, inflation
   :description: Inflation option can be cap and floor. An inflation cap (floor) is a financial asset that offers protection against inflation being higher (lower) than a given rate of inflation, and can therefore be used by investors to insure against such inflation outcomes.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


*************************************************
Internal Design of Inflation CapFloor Engine
*************************************************


Overview
========
Inflation option can be cap and floor. An inflation cap (floor) is a financial asset that offers protection against inflation being higher (lower) than a given rate of inflation, 
and can therefore be used by investors to insure against such inflation outcomes.

Implemention
============
the `YoYInflationBlackCapFloorEngine` is year-on-year inflation cap/floor engine based on black formula. The structure of the engine is shown as the figure below:

.. _my-figure1:
.. figure:: /images/inflationEngine.png
    :alt: Figure 1 architecture on FPGA
    :width: 50%
    :align: center

As we can see from the figure, the engine mainly contains 4 functions.

1. function discountFactor: the discount factor is calculated at the corresponding time point.
2. function totalVariance: the total variance of volatility is calculated at the corresponding time point.
3. function yoyRateImpl: the year-on-year forward rate is calculated at the corresponding time point.
4. function blackFormula: the black formula calculates the value of the option based on the results of the three functions mentioned above.

Finally, the addition of the results from each time point is the final price (NPV).

Profiling
=========

The hardware resource utilizations are listed in the following table (from Vivado 19.1 report).

.. table:: Table 1 Hardware resources
    :align: center

    +------------------------------------+----------+----------+----------+----------+---------+-----------------+
    |  Engine                            |   BRAM   |   URAM   |    DSP   |    FF    |   LUT   | clock period(ns)|
    +------------------------------------+----------+----------+----------+----------+---------+-----------------+
    |  YoYInflationBlackCapFloorEngine   |    0     |    0     |    170   |   33129  |  31999  |       3.210     |
    +------------------------------------+----------+----------+----------+----------+---------+-----------------+


