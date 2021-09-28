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
   :keywords: Zero Coupon Bond, zero, coupon, bond, engine
   :description: A zero-coupon bond is a bond which is purchased at a price below the face value of the bond. It does not pay coupon during the contract period, and repays the face value at the time of maturity.   
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


*************************************************
Internal Design of Zero Coupon Bond Engine
*************************************************


Overview
========
A zero-coupon bond is a bond which is purchased at a price below the face value of the bond. It does not pay coupon during the contract period, and repays the face value at the time of maturity.

Implemention
============
This engine uses the linear interpolation method (:math:`linearInterpolation`) as defined in L1 to calculate the price based on time (the difference between the maturity date and the reference date with the unit in year) and face value. The linear interpolation method implements 1-dimensional linear interpolation. 

Profiling
=========

The hardware resource utilizations are listed in the following table (from Vivado 18.3 report).

.. table:: Table 1 Hardware resources
    :align: center

    +-----------------+----------+----------+----------+----------+---------+-----------------+
    |  Engine         |   BRAM   |   URAM   |    DSP   |    FF    |   LUT   | clock period(ns)|
    +-----------------+----------+----------+----------+----------+---------+-----------------+
    |  ZCBondEngine   |    0     |    0     |    46    |   12478  |  7997   |       2.580     |
    +-----------------+----------+----------+----------+----------+---------+-----------------+


