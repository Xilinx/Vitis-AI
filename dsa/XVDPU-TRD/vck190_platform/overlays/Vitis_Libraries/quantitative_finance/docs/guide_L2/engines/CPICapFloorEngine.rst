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
   :keywords: consumer price index, CPI
   :description: A consumer price index (CPI) Cap/Floor is a call/put on the CPI.  
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


*************************************************
Internal Design of CPI CapFloor Engine
*************************************************


Overview
========
A consumer price index (CPI) Cap/Floor is a call/put on the CPI. 

Implemention
============
This engine uses the linear interpolation method (:math:`linearInterpolation2D`) as defined in L1 to calculate the price based on time (the difference between the maturity date and the reference date with unit in year) and strike rate. The linear interpolation mothed implements 2-dimensional linear interpolation. 

Profiling
=========

The hardware resource utilizations are listed in the following table (from Vivado 18.3 report).

.. table:: Table 1 Hardware resources
    :align: center

    +----------------------+----------+----------+----------+----------+---------+-----------------+
    |  Engine              |   BRAM   |   URAM   |    DSP   |    FF    |   LUT   | clock period(ns)|
    +----------------------+----------+----------+----------+----------+---------+-----------------+
    |  CPICapFloorEngine   |    0     |    0     |    22    |   11385  |  7625   |       2.966     |
    +----------------------+----------+----------+----------+----------+---------+-----------------+


