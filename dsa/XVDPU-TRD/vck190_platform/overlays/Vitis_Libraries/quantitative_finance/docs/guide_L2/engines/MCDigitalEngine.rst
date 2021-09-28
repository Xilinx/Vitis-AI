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
   :keywords: Digital, pricing, engine, MCDigitalEngine
   :description: Digital Option pricing engine uses Monte Carlo Simulation to estimate the value of digital option. Here, we assume the process of asset pricing applies to Black-Scholes process.  
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


***************************************************
Internal Design of Digital Option Pricing Engines
***************************************************

Overview
=========

Digital Option pricing engine uses Monte Carlo Simulation to estimate the value of digital option. Here, we assume the process of asset pricing applies to Black-Scholes process. 

Digital option is an option whose payoff is characterized as having only two potential values: a fixed payout, when the option is in the money or a zero payout otherwise. 
It is not related to how far the asset price at maturity is above (call) or below (put) the strike.

Digital options are attractive to buyers because the option payoff is a known constant amount, and this amount can be adjusted to provide the exact quantity of protection required. It overcomes a fundamental problem with the vanilla options, where the potential loss is unlimited, referring to the wiki.

.. NOTE::
   Only one type of digital options is supported:

   * Cash-or-nothing option: Pays some fixed amount of cash if the option expires in the money.
   * Asset-or-nothing option (not supported)

Implementation
===============

The implementation of digital option pricing engine is very similar to the barrier option pricing engine. It also uses the Brownian Bridge to correct the hitting error.
Here, the exercise of digital option could be at the expiry time or any time between the expiry time and the 0 time, which is configured the by `exEarly` input arguments.
When argument `exEarly` is false, the fixed cash is paid at the expiry time and it is discounted to time zero for the value of option. 
Otherwise, once the maximum or the minimum of the asset price hits the strike value, the fixed cash is paid and it is discounted to time 0 for the value of option.


In the following, we will take put digital option as an example to elaborate.
Let :math:`T` be the maturity time of option. The maturity time :math:`T` is discretized by time steps :math:`N`. 
The strike value is :math:`K`. :math:`C` is the fixed cash which paid when option is in the money.

The detailed procedure of Monte Carlo Simulation is as follows:

1. For :math:`i` = 1 to :math:`M`

  - For :math:`j` = 1 to :math:`N`

    - generate a normal random number and uniform random number :math:`u`;
    - simulate the price of asset :math:`S^i_j`;
    - simulate the maximum price of asset :math:`M` during time interval :math:`[t_{j-1}, t_j]`.

.. math::
   x = \ln \frac {S^i_j}{S^i_{j-1}}
.. math::
   y = \frac {x - \sqrt {(x^2 - 2\sigma^2 \Delta t\log u)}} {2}
.. math::
   M = S^i_j\exp (y)

2. Calculate the payoff and discount it to time 0 for option price :math:`P_i`. if :math:`M > K`,

.. math::
   P_i = C\exp (-rt_j), exEarly = true

.. math::
   P_i = C\exp (-rT), exEarly = false

or if :math:`M` for all time interval never cross strike, 

.. math::
   P_i = 0

So, the estimated value of option is the average of all the samples.
  
.. math::
   c = \frac{1}{M}\sum_{i=1}^{M} P_i


