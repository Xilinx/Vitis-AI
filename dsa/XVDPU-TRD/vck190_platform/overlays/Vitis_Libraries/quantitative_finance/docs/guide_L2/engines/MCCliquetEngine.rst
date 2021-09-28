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
   :keywords: Cliquet, pricing, engine, MCCliquetEngine
   :description: Cliquet option pricing engine uses Monte Carlo Simulation to estimate the value of Cliquet Option. Here, we assume the process of asset pricing applies to Black-Scholes process.  
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials



*************************************************
Internal Design of Cliquet Option Pricing Engine
*************************************************

Overview
=========

The Cliquet option pricing engine uses Monte Carlo Simulation to estimate the value of Cliquet Option. Here, we assume the process of asset pricing applies to Black-Scholes process. 

The Cliquet Option is an exotic option. It is constructed by a series of forward start options. 
The start dates, also called resets dates, are pre-determined in the contract in advance. Generally, 
the reset dates are periodical, such as semiannual or  quarterly. 

At every reset date, the option calculates the relative performance between the old and new underlying price
and pays that out as a profit.

The payoff of the option at the maturity is the sum of the payout at every reset date. 

The option price is calculated as:

.. math::
   
   Option Price = \sum_{i=0}^N {\exp {(-r*t_i)} * R_i}
 
   R_i = payoff(\frac{S_i}{S_{i-1}})

Where :math:`N` is the number or reset dates and it is equal to timesteps. :math:`r` is the risk-free interest rate.

Implementation
==============

The path pricer for Cliquet option fetches the lognormal `S` and calculate the payoff at each reset dates.
Because the path generator generates the :math:`logS` instead of :math:`S`, the divider in path pricer could be optimized by subtraction.

.. NOTE::
   Because the calendar date for each month is different, the time interval between each resets date is different. FPGA is not efficient to calculate
   the calendar dates, so the customer needs to input them through the resetDate array.


The detailed procedure of Monte Carlo Simulation is as follows:

- For :math:`i` = 1 to samplesRequired

  - For :math:`j` = 1 to :math:`N`
    
    - generate a normal random number;
    - simulate the log of asset price :math:`\ln S^i_j`;
    - calculate the payoff based on above formula and discount to time zero for option price.

Then sum up all the prices for the mean value.

