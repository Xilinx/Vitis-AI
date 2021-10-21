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
   :keywords: Heston, Closed-Form, Heston PDE
   :description: The Heston model extends the well-known Black-Scholes options pricing model by adding a stochastic process for the stock volatility.  
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials



*********************************
Heston Model Closed-Form Solution
*********************************

.. toctree::
   :maxdepth: 1

Overview
========

The `Heston Model`_ , published by Steven Heston in paper "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options" in 1993 [HEST1993]_ , extends the well-known Black-Scholes options pricing model by adding a stochastic process for the stock volatility.  

.. _Heston Model: https://en.wikipedia.org/wiki/Heston_model

The stochastic equations of the model, and the partial differential equation (PDE) derived from them, are shown in the section on the Heston Model under "Models".  There are a number of ways in which the 
equations may be solved, including Monte-Carlo and Finite-Difference methods. Here, we show the closed-form solution originally obtained in [HEST1993]_ - sometimes referred to as the "semi-analytic" solution 
since an analytic solution of its integrals is not known. 

The expression for a European Call option, derived from the Heston PDE, is shown below, (in the form presented by Crisostomo in [CHRSO2014]_ ): 

.. math::
   C_0 = S_0.\Pi_1  - \mathrm{e}^{-rT}K.\Pi_2

where here and in the following: :math:`T` (and :math:`t`) = Time to Expiration; :math:`K` = Option Strike Price; :math:`S_0` = stock price at :math:`t = 0`; :math:`r` = interest-rate; :math:`V_0` = stock-price variance at :math:`t = 0`; 
:math:`\eta` = "volatility-of-volatility" (elsewhere :math:`\sigma`); :math:`a` = rate-of-reversion (elsewhere :math:`\kappa`); :math:`\tilde{V}` = long-term average variance (elsewhere :math:`\theta`); :math:`\rho` = correlation of 
:math:`z_1(t)` and :math:`z_2(t)` processes. 

Using a solution based on Characteristic functions, the values of probabilities :math:`\Pi_1` and :math:`\Pi_2` are given by: 

.. math::
   \Pi_1 = \frac{1}{2} + \frac{1}{\pi} \int_0^\infty Re\left[\frac{\mathrm{e}^{-i.w.ln(K)}.\Psi_{lnS_T}(w - i)}{i.w.\Psi_{lnS_T}(-i)}  \right]\mathrm{d}w

.. math::
   \Pi_2 = \frac{1}{2} + \frac{1}{\pi} \int_0^\infty Re\left[\frac{\mathrm{e}^{-i.w.ln(K)}.\Psi_{lnS_T}(w)}{i.w}  \right]\mathrm{d}w

where the Characteristic function ψ is:  

.. math::
   \Psi_{lnS_T}(w) = \mathrm{e}^{[C(t,w).\tilde{V} + D(t,w).V_0 + i.w.ln(S_0.\mathrm{e}^{rt})]} 

where:

.. math::
   C(t,w) = a.\left[r_-.t - \frac{2}{\eta^2}.ln\left(\frac{1 - g.\mathrm{e}^{-ht}}{1 - g}\right)\right]

   D(t,w) = r_- .\frac{1 -\mathrm{e}^{-ht}}{1 - g.\mathrm{e}^{-ht}}

   r_{\pm} = \frac{\beta \pm h}{\eta^2};  h = \sqrt{\beta^2 - 4.\alpha.\gamma} 

   g = \frac{r_-}{r_+} 

   \alpha = -\frac{w^2}{2} - \frac{iw}{2} ;       \beta = \alpha - \rho.\eta. i. w ;       \gamma = \frac{\eta^2}{2} 

To obtain a solution for Call :math:`C_0` the integrands in the :math:`\Pi_1` and :math:`\Pi_2` terms must be evaluated using a selected numeric integration technique suited to integration from 0 to ∞, and 
the internal characteristic function terms must be obtained using complex-number computation.    



References
==========

.. [HEST1993] Heston, S. L., "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options", The Review of Financial Studies (1993)

.. [CHRSO2014] Crisostomo, R., "An Analysis of the Heston Stochastic Volatility Model: Implementation and Calibration using Matlab", CNMV Working Paper 58: 1-46, (2014).

