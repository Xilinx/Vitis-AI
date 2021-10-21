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
   :keywords: Tree Bermudan Swaption, engine, Swap, capfloor, callable
   :description: Swap engine, swaption engine, capfloor engine and callable engine are all pricing engines to price the interested products.   
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


*************************************************
Internal Design of Tree Bermudan Swaption Engine
*************************************************


Overview
========
Swap engine, swaption engine, capfloor engine and callable engine are all pricing engines to price the interested products. The swap is mainly an interest rate swap. For both parties to the swap, the buyer needs execute a specified swap agreement with the issuer on a specified future date.

The swaption mainly refers to an option to do swap. The buyer acquires the right but not the obligation to enter into a specified swap agreement with the issuer on a specified future date. 

The capfloor includes 2 interest rate derivatives: cap, and floor. For the interest rate cap, the buyer receives payments from the issuer at the end of each period in which the interest rate exceeds the agreed strike price. For interest rate floor, the buyer receives payments from the issuer at the end of each period in which the interest rate is below the agreed strike price.

The callable bond is a type of bond that provides the issuer of the bond with the right, but not the obligation, to redeem the bond on a specified future date before its maturity date.

Implemention
============
As shown in the figure below, this engine uses the framework of Tree Lattice in L1. It has a Rate Model and 1 or 2 Stochastic Process as input.

.. _my-figure1:
.. figure:: /images/tree/treeEngine.png
    :alt: Figure 1 treeSwaptionEngine architecture on FPGA
    :width: 60%
    :align: center


1. From the input parameters, the time grid and the corresponding counter of exercise times and payment times of fixed or floating interest rate are obtained (All time points are relative values based on the reference date in year, and the engine only supports the case where the time point is not less than 0).
2. By calling the function setup of the framework, the floating interest rates and tree related parameters are calculated from 0 to N timepoint-by-timepoint to prepare the interest rates and the tree related parameters for the following calculations.
3. Take treeSwaptionEngine, for example, by calling the function rollback of the framework using the same structure of the tree, the net present value (NPV) is calculated from N to 0 timepoint-by-timepoint. The implementation is shown in the figure below, where the data flow along with the arrows.


.. _my-figure2:
.. figure:: /images/tree/swaptionRollback.png
    :alt: Figure 2 treeSwaptionEngine's Rollback Module architecture on FPGA
    :width: 80%
    :align: center



Profiling
=========

The hardware resources are listed in the following table (from Vivado 18.3 report).

.. table:: Table 1 Hardware resources
    :align: center

    +-----------------------+----------------+----------+----------+----------+----------+---------+-----------------+
    |  Engine               |  Models        |   BRAM   |   URAM   |    DSP   |    FF    |   LUT   | clock period(ns)|
    +-----------------------+----------------+----------+----------+----------+----------+---------+-----------------+
    |  treeSwaptionEngine   |  HWModel       |    112   |    0     |    452   |   87469  |  67212  |       3.053     |
    +-----------------------+----------------+----------+----------+----------+----------+---------+-----------------+
    |  treeSwaptionEngine   |  BKModel       |    116   |    0     |    495   |   99209  |  82034  |       3.190     |
    +-----------------------+----------------+----------+----------+----------+----------+---------+-----------------+
    |  treeSwaptionEngine   |  CIRModel      |    104   |    0     |    417   |   82910  |  51160  |       3.110     |
    +-----------------------+----------------+----------+----------+----------+----------+---------+-----------------+
    |  treeSwaptionEngine   |  ECIRModel     |    116   |    0     |    442   |   102802 |  81395  |       3.205     |
    +-----------------------+----------------+----------+----------+----------+----------+---------+-----------------+
    |  treeSwaptionEngine   |  VModel        |    104   |    0     |    377   |   74551  |  48322  |       3.054     |
    +-----------------------+----------------+----------+----------+----------+----------+---------+-----------------+
    |  treeSwaptionEngine   |  G2Model       |    18    |    136   |    625   |   139467 |  90205  |       3.896     |
    +-----------------------+----------------+----------+----------+----------+----------+---------+-----------------+
    |  treeSwapEngine       |  HWModel       |    104   |    0     |    408   |   84628  |  65744  |       3.896     |
    +-----------------------+----------------+----------+----------+----------+----------+---------+-----------------+
    |  treeCapFloorEngine   |  HWModel       |    104   |    0     |    364   |   79489  |  64863  |       3.180     |
    +-----------------------+----------------+----------+----------+----------+----------+---------+-----------------+
    |  treeCallableEngine   |  HWModel       |    104   |    0     |    320   |   76577  |  62445  |       3.043     |
    +-----------------------+----------------+----------+----------+----------+----------+---------+-----------------+


The following table shows the comparison of the performance between U250 result and CPU based Quantlib result. (treeSwaptionEngine+G2Model FPGA System Clock: 250MHz, others FPGA System Clock: 300MHz)


.. table:: Table 2 Comparison between CPU and FPGA
    :align: center

    +----------------------+---------------+----------------------------+-------+-------+-------+--------+
    |      Engine          |    Models     | Timesteps                  | 50    | 100   | 500   | 1000   |
    +----------------------+---------------+----------------------------+-------+-------+-------+--------+
    |                      |               | CPU Execution time(ms)     | 1.0   | 4.8   | 353.9 | 2493.5 |
    |  treeSwaptionEngine  |    HWModel    +----------------------------+-------+-------+-------+--------+
    |                      |               | FPGA Execution time-HLS(ms)| 0.28  | 0.61  | 5.72  | 18.17  |
    +----------------------+---------------+----------------------------+-------+-------+-------+--------+
    |                      |               | CPU Execution time(ms)     | 1.9   | 8.6   | 438.2 | 2813.1 |
    |  treeSwaptionEngine  |    BKModel    +----------------------------+-------+-------+-------+--------+
    |                      |               | FPGA Execution time-HLS(ms)| 0.72  | 1.53  | 11.93 | 34.21  |
    +----------------------+---------------+----------------------------+-------+-------+-------+--------+
    |                      |               | CPU Execution time(ms)     | 0.5   | 1.4   | 26.6  | 100.7  |
    |  treeSwaptionEngine  |    CIRModel   +----------------------------+-------+-------+-------+--------+
    |                      |               | FPGA Execution time-HLS(ms)| 0.16  | 0.31  | 2.22  | 6.18   |
    +----------------------+---------------+----------------------------+-------+-------+-------+--------+
    |                      |               | CPU Execution time(ms)     | 1.1   | 5.5   | 439.5 | 3322.5 |
    |  treeSwaptionEngine  |    ECIRModel  +----------------------------+-------+-------+-------+--------+
    |                      |               | FPGA Execution time-HLS(ms)| 0.72  | 1.36  | 10.17 | 28.47  |
    +----------------------+---------------+----------------------------+-------+-------+-------+--------+
    |                      |               | CPU Execution time(ms)     | 0.5   | 1.8   | 40.1  | 161.7  |
    |  treeSwaptionEngine  |    VModel     +----------------------------+-------+-------+-------+--------+
    |                      |               | FPGA Execution time-HLS(ms)| 0.14  | 0.29  | 2.42  | 7.42   |
    +----------------------+---------------+----------------------------+-------+-------+-------+--------+
    |                      |               | CPU Execution time(ms)     | 258.0 | 2133.5|       |        |
    |  treeSwaptionEngine  |    G2Model    +----------------------------+-------+-------+-------+--------+
    |                      |               | FPGA Execution time-HLS(ms)| 1.93  | 14.56 |       |        |
    +----------------------+---------------+----------------------------+-------+-------+-------+--------+
    |                      |               | CPU Execution time(ms)     | 1.0   | 4.3   | 291.2 | 2056.5 |
    |  treeSwapEngine      |    HWModel    +----------------------------+-------+-------+-------+--------+
    |                      |               | FPGA Execution time-HLS(ms)| 0.28  | 0.61  | 5.61  | 18.16  |
    +----------------------+---------------+----------------------------+-------+-------+-------+--------+
    |                      |               | CPU Execution time(ms)     | 0.7   | 3.4   | 217.6 | 1581.3 |
    |  treeCapFloorEngine  |    HWModel    +----------------------------+-------+-------+-------+--------+
    |                      |               | FPGA Execution time-HLS(ms)| 0.30  | 0.64  | 5.89  | 18.51  |
    +----------------------+---------------+----------------------------+-------+-------+-------+--------+
    |                      |               | CPU Execution time(ms)     | 1.4   | 3.5   | 155.2 | 1142.0 |
    |  treeCallableEngine  |    HWModel    +----------------------------+-------+-------+-------+--------+
    |                      |               | FPGA Execution time-HLS(ms)| 0.28  | 0.60  | 5.67  | 17.89  |
    +----------------------+---------------+----------------------------+-------+-------+-------+--------+


