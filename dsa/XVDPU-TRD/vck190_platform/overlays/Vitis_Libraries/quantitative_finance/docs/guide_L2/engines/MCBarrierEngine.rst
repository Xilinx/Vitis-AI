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
   :keywords: Barrier, pricing, engine, MCBarrierEngine
   :description: Barrier option pricing engine uses Monte Carlo Simulation method to estimate the payoff of barrier option. Here, we assume the process of asset pricing applies to Black-Scholes process. 
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials



*************************************************
Internal Design of Barrier Option Pricing Engine
*************************************************
Overview 
=========

The Barrier option pricing engine uses Monte Carlo Simulation method to estimate the payoff of barrier option. Here, we assume the process of asset pricing applies to Black-Scholes process.

Barrier option is a kind of option whose payoff depends on whether the option is effective at the maturity time. Only when the option is effective, the holder of the option has the right, but not the obligation, to buy/sell the underlying asset at the strike price. If the option is effective, the payoff of it will be calculated as the European Option.

There are four barrier option types, including:

* 'UpIn': The option becomes effective when the price of the underlying asset is greater than the barrier value. 

* 'UpOut': The option becomes ineffective when the price of the underlying asset is greater than the barrier value.

* 'DownIn': The option becomes effective when the price of the underlying asset is less than the barrier value.

* 'DownOut': The option becomes ineffective when the price of the underlying asset is less than the barrier value.

The rebate is a fixed value which is paid when the option is ineffective.

.. NOTE::
   In our implementation, barrier option means continuous and single barrier option. That is to say, the barrier event could happen at any time during the lifetime of option. 

Implementation
================

We provide two kinds of pricing engines, MCBarrierEngine and MCBarrierNoBiasEngine to evaluate the barrier option. 

MCBarrierEngine generates the result with bias because it only considers whether the asset price at each discrete time point hits the barrier level.

However, the asset price during each discrete time interval maybe go up and down to hit the barrier level, which causes the bias of result. MCBarrierNoBiasEngine is to approximate the price process during the interval by a Brownian bridge.
After simulating the asset price at each discrete time point, draw the maximum or minimum of the stock price on the interval using the known theoretical distribution of a Brownian Bridge, refer to **''Going to Extremes: Correcting Simulation Bias in Exotic Option Valuation - D.R. Beaglehole, P.H. Dybvig and G. Zhou, Correcting for Simulation Bias in Monte Carlo methods to Value Exotic Opitons in Models Driven by Levy Process - Claudia Ribeiro and Nick Webber.''**

In the following, we will take up-and-out barrier option as an example to elaborate the two kinds of pricing engines.
Let :math:`T` be the maturity time of option, the barrier level is :math:`B`. The maturity time :math:`T` is discretized by time steps :math:`N`. 
The rebate value is :math:`R`. If a barrier option fails to exercise, the seller may pay a rebate to the buyer of the option. Knock-outs may pay a rebate when they are knocked out, and knock-ins may pay a rebate if they expire without ever knocking in.


MCBarrierEngine
----------------

MCBarrierEngine is similar to the MCEuropeanEngine except for the path pricer. 

In this engine, the path pricer samples the asset price path at the discrete time intervals :math:`[t_j, t_j+1]`, :math:`j=1, ..., N`. The barrier condition is tested
at each time step. When the barrier level is hit, the option becomes non-effective and the rebate is paid for the owner of option. The rebate is discounted to today for value of option.
If the barrier is never hit, the payoff of the option is calculated by value of asset at maturity time :math:`T` and it is discounted to today for value of option.

The detailed procedure of Monte Carlo Simulation is as follows:

- For :math:`i` = 1 to :math:`M`

  - For :math:`j` = 1 to :math:`N`

    - generate a normal random number;
    - simulate the price of asset :math:`S^i_j`;
    - calculate the payoff and discount it to today.

if :math:`S^i_j > B`,

.. math::
   P_i = R\exp (-rt_j)

or if :math:`S^i_1, S^i_2, ..., S^i_N` never cross the barrier, 

.. math::
   P_i = max(S^i_N - K, 0)\exp (-rT). 

So, the estimated value of option is the average of all the samples.
  
.. math::
   c = \frac{1}{M}\sum_{i=1}^{M} P_i

The :math:`c` is a biased estimate of the barrier option value, because the sample path may have exceeded the barrier level, knocking out the option, during the time interval. 

MCBarrierNoBiasEngine
----------------------

For barrier option, the payoff is decided by the maximum or minimum value of the underlying asset during the lifetime of option.
The maximum of the price process on a discrete set of times is always lower than the maximum for all times, so the MCBarrierEngine always underestimate the option price.  

Here, we use the Brownian bridge approach to eliminate the bias. For the discrete time interval :math:`[t_j, t_j+1]`, when :math:`S(t_j)` and :math:`S(t_{j+1})` is fixed,
the process of :math:`S(t), t\in[t_j,t_j+1]` is a Brownian bridge. We will consider :math:`Pr` is the probability of the maximum asset price during time interval :math:`[t_i, t_i+1]` less than barrier level.
Then, we have:

.. math::
   Pr[max_{t\in{t_j, t_j+1}}(S_t) <= B|S(t_j), S(t_{j+1})] = 1 - \exp (-2\frac {(\underline{B}-R_j)(\underline{B}-R_{j+1})} {\sigma^2 \Delta t})

where :math:`\underline{B} = \ln \frac {B}{S(t_j)}, R_j = \ln \frac {S(t_{j+1})}{S(t_j)}`.  

Let :math:`u` ~ [0,1] be a draw of a standard uniform variate, then 

.. math::
   M = \frac {R_{j+1} + R_j - \sqrt {(R_{j+1} - R_j)^2 - 2\sigma^2 \Delta t\log u}} {2}

is a draw from the distribution of the maximum of :math:`\ln \frac{S(t)}{S(t_j)}`. So :math:`S(t_j)\exp (M)` is a draw from maximum of :math:`S(t)`.

The detailed procedure of Monte Carlo Simulation is as follows:

- For :math:`i` = 1 to :math:`M`

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

The calculation of payoff is similar to the step 3 in MCBarrierEngine except that the :math:`S^i_j` is replaced by :math:`M`.


