
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
   :keywords: fintech, Ornstein-Uhlenbeck
   :description: Ornstein-Uhlenbeck is a stochastic process which uses the Random Number Generator (RNG) to generate locations for mesher.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


**************************
Ornstein-Uhlenbeck Process
**************************

Overview
=========
`Ornstein-Uhlenbeck process` is a stochastic process which uses the Random Number Generator (RNG) to generate locations for mesher. It uses a reference time point :math:`\Delta w` and a time step :math:`\Delta t` to calculate the drift and diffusion. The Ornstein-Uhlenbeck process is a simple stochastic processes, whose feature of interest is its mean-reverting drift term :math:`a(r-x)` and its constant diffusion term :math:`\sigma`.

The Ornstein-Uhlenbeck process can be described by

.. math::
   dx=a(r-x_{t})dt+\sigma dW_{t}

Implementation
===================
The implementation of `OrnsteinUhlenbeckProcess` contains a few methods. The implementation can be introduced as follows:

1. init: The initialization process to set up arguments as below:

   a)speed, the spreads on interest rates;

   b)vola, the overall level of volatility;

   c)x0, the initial value of level;

   d)level, the width of fluctuation on interest rates.

2. expectation: The expectation method returns the expectation of the process at time :math:`E(x_{t_{0}+\Delta t}|x_{t_{0}}=x_{0})`. 

3. stdDeviation: The stdDeviation method returns the standard deviation :math:`S(x_{t_{0}+\Delta t}|x_{t_{0}}=x_{0})` of the process with a time period :math:`\Delta t` according to the given discretization. 

4. variance: The variance method returns the variance :math:`V(x_{t_{0}+\Delta t}|x_{t_{0}}=x_{0})` of the process with a time period :math:`\Delta t` according to the given discretization. 

5. evolve: The evolve method returns the asset value after a time interval :math:`\Delta t` according to the given discretization. It returns,

.. math::
   E(x_{0},t_{0},\Delta t)+S(x_{0},t_{0},\Delta t)*\Delta w

where :math:`E` is the expectation and :math:`S` the standard deviation.

