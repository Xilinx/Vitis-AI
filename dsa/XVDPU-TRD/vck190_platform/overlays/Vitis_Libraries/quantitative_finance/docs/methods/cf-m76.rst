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
   :keywords: Merton 76, Closed-Form, Black Scholes, Diffusion, Poisson
   :description: Merton's Jump Diffusion Model superimposes a jump component on the Black Scholes Diffusion Component. This jump component is composed of log-normal jumps driven by a Poisson process.  
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


*********************************
Merton 76 Closed-Form Solution
*********************************

.. toctree::
   :maxdepth: 1

Overview
========

One of the assumptions in the Black Scholes model is that the stock price moves as geometric Brownian Motion. In reality the stock price will have 'jumps' due to events such as takeover announcements and court actions. Merton's Jump Diffusion Model does this by superimposing a jump component on the Black Scholes Diffusion Component. This Jump Component is composed of log-normal jumps driven by a Poisson process.   

An equation can be derived for the resulting Call Option price:

.. math::
   F(S,\tau) = \sum_{n=0}^{\infty} \frac{\mathrm{e}^{-\lambda\tau}(\lambda\tau)^n}{n!} \left[\epsilon_n\lbrace W(SX_n\mathrm{e}^{-\lambda k\tau},\tau ;E,\sigma^2,r)\rbrace\right]

where W is the standard :ref:`Black Scholes <black_scholes>` formula and :math:`X_n` is a complicated random variable which is out of the scope of this explanation. 
This is very difficult to solve but Merton came up with a specific case where :math:`X_n` has a log-normal distribution.
This simplifies the closed form call price solution to:

.. math::
   F(S,\tau) = \sum_{n=0}^{\infty} \frac{\mathrm{e}^{-\lambda^{'}\tau}(\lambda^{'}\tau)^n}{n!} W_n(S,\tau ;E,v_n^2,r_n)

.. math::
   v_n^2 = \left[\sigma^2 + n\delta^2 / \tau \right]

.. math::
   r_n = r - \lambda k + n \gamma / \tau

where:

S = stock price

E = strike price

:math:`X_n` = complicated random variable!

:math:`\tau` = time to maturity

:math:`\lambda` = mean jump count per unit time

:math:`\lambda^{'}` = :math:`\lambda' (1+k)`

:math:`\sigma` = volatility

:math:`\delta^2` = variance of `ln` (:math:`\gamma`)

r = interest rate

k = Expected(:math:`\gamma` -1)

:math:`\gamma` = random Poisson variable


References
==========

.. [M76.1] Merton, Robert C. (1976). "Option Pricing when underlying stock returns are discontinuous", Journal of Financial Economics 3 (1976) 125-144.

.. [M76.2] Merton, Robert C. (1976). "The Impact on Option Pricing of Specification Error in the Underlying Stock Price Returns", The Journal of Finance, Vol XXXI, No 2.

