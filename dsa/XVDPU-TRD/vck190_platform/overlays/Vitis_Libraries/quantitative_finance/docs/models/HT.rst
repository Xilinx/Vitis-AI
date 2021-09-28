

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
   :keywords: Model, finance, Heston, Stochastic, PED
   :description: Heston model is a mathematical model that describing dynamics of underlying asset price.  
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

*******************
Heston Model
*******************

Overview
=========
Heston model is a mathematical model that describing dynamics of underlying asset price. 
It is a stochastic volatility model which assumes the volatility of the asset price is not constant but follows a random process. 
In this case, volatility follows square root process which means volatility is always non-negative.


Stochastic Process Equations of the Heston Model
================================================
The continuous form of Heston Model is:

.. math::
   \mathrm{d} S_t = \mu S_t \mathrm{d}t + \sqrt[2]{\nu} S_t \mathrm{d}W_t^S

.. math::
   \mathrm{d} \nu_t = \kappa (\theta - \nu_t) \mathrm{d}t + \sigma \sqrt[2]{\nu_t} \mathrm{d}W_t^\nu

.. math::
   Corr(W_t^S, W_t^\nu)) = \rho

Where :math:`S_t` is random variable, represent a stock price at time :math:`t`. 
:math:`\mu` is the stock's expected rate of return. 
:math:`\sqrt[2]{\nu_t}` is the volatility of stock price and :math:`\nu_t` is also a random variable. 
:math:`\theta` is the long term variance. 
:math:`\kappa` is the rate at which :math:`\nu_t` revert to :math:`\theta`. 
:math:`\sigma` is the volatility of the volatility. 
:math:`\mathrm{d}W_t^S` and :math:`\mathrm{d}W_t^\nu` are Wiener processes with correlation :math:`\rho`. 

Partial Differential Equation (PED) of Heston Model
===================================================
In the provided solver, the PDE is the Heston Pricing Model [HESTON1993]_. Using the naming conventions of Hout & Foulon [HOUT2010]_, the PDE is given as:

.. math::
   \frac{\partial u}{\partial t} = \tfrac{1}{2}s^2v\frac{\partial^{2} u}{\partial s^2} + \rho\sigma sv\frac{\partial^{2} u}{\partial s\partial v} + \tfrac{1}{2}\sigma^2v\frac{\partial^{2} u}{\partial v^2} + (r_d - r_f)s\frac{\partial u}{\partial s} + \kappa(\eta - v)\frac{\partial u}{\partial v} - r_d u

where:

:math:`u` - the price of the European option;

:math:`s` - the underlying price of the asset;

:math:`v` - the volatility of the underlying price;

:math:`\sigma` - the volatility of the volatility;

:math:`\rho` - the correlation of Weiner processes;

:math:`\kappa` - the mean-reversion rate;

:math:`\eta` - the long term mean. 

The Heston PDE then describes the evolution of an option price over time (:math:`t`) and a solution of this PDE results in the specific option price for an :math:`(s,v)` pair for a given maturity date :math:`T`. The finite difference solver maps the :math:`(s,v)` pair onto a 2D discrete grid, and solves for option price :math:`u(s,v)` after :math:`N` time-steps.


Implementations
=============================================
The discrete form (Euler-Maruyama Form) of Heston Model is:

.. math::
   \ln{S(j\Delta)} = \ln{S((j-1)\Delta)} + \mu\Delta + \sqrt[2]{\nu((j-1)\Delta)} \Delta W_t^S

.. math::
   \nu(j\Delta) = \nu((j-1)\Delta) + \kappa(\theta - \nu((j-1)\Delta))\Delta + \sigma\sqrt[2]{\nu((j-1)\Delta)}\Delta W_t^\nu

Where :math:`\Delta` stands for unit timestep length. 
:math:`S(j\Delta)` stands for S in j th timesteps, AKA :math:`S(j * \Delta t)`. 
:math:`\nu(j\Delta)` has a similar meaning. 
To simplify the process, we use :math:`\ln{S(j\Delta)}` instead of :math:`S` since multiplication becomes addition after \logarithm. 

:math:`\Delta W_t^S` and :math:`\Delta W_t^\nu` could be calculated by:

.. math::
    \Delta W_t^S = \Delta Z_S

.. math::
    \Delta W_t^\nu = \Delta Z_\nu

Where :math:`Z_S` and :math:`Z_\nu` are two uniform distributed random numbers that have correlation :math:`\rho`

Although :math:`\nu` is non-negative in continuous form, it would become negative if we use the Euler-Maruyama form above directly. 
There are several variations to solve this issue, here we provide 5 of most commonly used.

**kDTReflection**: use absolute value of the volatility of the last iteration.

.. math::
   \ln{S(j\Delta)} = \ln{S((j-1)\Delta)} + \mu\Delta + \sqrt[2]{|\nu((j-1)\Delta)|} \Delta W_t^S

.. math::
   \nu(j\Delta) = |\nu((j-1)\Delta)| + \kappa(\theta - |\nu((j-1)\Delta)|)\Delta + \sigma\sqrt[2]{|\nu((j-1)\Delta)|}\Delta W_t^\nu


**kDTPartialTruncation**: use only absolute value of volatility of the last iteration in sqrt root part.

.. math::
   \ln{S(j\Delta)} = \ln{S((j-1)\Delta)} + \mu\Delta + \sqrt[2]{|\nu((j-1)\Delta)|} \Delta W_t^S

.. math::
   \nu(j\Delta) = \nu((j-1)\Delta) + \kappa(\theta - \nu((j-1)\Delta))\Delta + \sigma\sqrt[2]{|\nu((j-1)\Delta)|}\Delta W_t^\nu


**kDTFullTruncation**: use only positive part or zero of volatility of the last iteration.

:math:`\nu((j-1)\Delta)^+ = \nu((j-1)\Delta)` if :math:`\nu((j-1)\Delta) > 0` :math:`\nu((j-1)\Delta)^+ = 0` if :math:`\nu((j-1)\Delta) \leq 0`.

.. math::
   \ln{S(j\Delta)} = \ln{S((j-1)\Delta)} + \mu\Delta + \sqrt[2]{\nu((j-1)\Delta)^+} \Delta W_t^S

.. math::
   \nu(j\Delta) = \nu((j-1)\Delta) + \kappa(\theta - \nu((j-1)\Delta)^+)\Delta + \sigma\sqrt[2]{\nu((j-1)\Delta)^+}\Delta W_t^\nu


**kDTQuadraticExponential** and **kDTQuadraticExponentialMartingale** are more accurate variation, details could be found in reference papers [ANDERSON2005]_. 
They use a different approximation method to calculate :math:`\nu`, here's brief on its algorithm

Step 1: we calculate first order moment and second order moment of :math:`\nu`.

.. math::
     m = \theta + (\nu((j-1)\Delta) - \theta)e^{-\kappa \theta}

.. math::
    s^2 = \frac{\nu((j-1)\Delta)\sigma^2 e^{-\kappa \Delta} }{\kappa} (1 - e^{-\kappa \Delta}) + \frac{\theta \sigma^2}{2 \kappa}(1 - e^{-\kappa \Delta})^2

Step 2: Calculate :math:`\Psi = s^2 / m^2`

Step 3: If :math:`\Psi \leq \Psi_{sw}, \Psi_{sw} = 1.5`, Then

Step 3.1: Calculate :math:`a` and :math:`b^2`

.. math::
    b^2 = 2\Psi^{-1} - 1 + \sqrt[2]{2\Psi^{-1}} \sqrt[2]{2\Psi^{-1}-1}

.. math::
    a = \frac{m}{1+b^2}

Step 3.2: Calculate :math:`\nu(j\Delta)`

.. math::
    \nu(j\Delta) = a(b+Z_\nu)^2

Step 4: If Step 3 does not hold, Then

Step 4.1: Calculate :math:`\beta` and :math:`p`

.. math::
    p = \frac{\Psi - 1}{\Psi + 1}

.. math::
    \beta = \frac{2}{m(\Psi + 1)}

Step 4.2: Calculate :math:`U_\nu = \Psi(Z_\nu)`

Step 4.3: Calculate :math:`\nu(j\Delta)`

.. math::
    \nu(j\Delta) = 0 \:\: if \:\: U_\nu \leq p

.. math::
    \nu(j\Delta) = \frac{1}{\beta}(\frac{\log{(1-p)}}{1-U_\nu}) \:\: if \:\:  p < U_\nu
    
It should be noticed that they both have two branches for value in different range. 
These two branches have a similar calculation process. 
Furthermore, only one branch is active at the same time. 
By merging these two branches into one branch and manually binding calculations to DSPs, it will cut off DSP cost. 
This won't change its performance and accuracy.

In Monte Carlo Simulation, we need to compute stock prices of multiple paths at multiple time steps.
Therefore we need two loops to calculate prices and volatilities, the inner loop is either timestep loop or path loop.
Price at each time step is calculated using last time step's price and volatility as input.
And we use 1-D array to store price and volatility of each path's history (last timestep).

If the inner loop is timestep loop, as red arrows demonstrate in the diagram below, it will keep update the same array element until reaches max timesteps.
Such operation can not achieve initiation interval (II)=1 and will greatly slow down the calculation process.
If the inner loop is path loop, as green arrows demonstrate in the diagram below, it will keep updating different array element each time.
Such operation will avoid dependency issue and reach II=1, which is used in this implementation.

.. image:: /images/inner_loop.png
   :alt: Inner Loop of timesteps and path
   :width: 80%
   :align: center




References
==========

.. [HESTON1993] Heston, "A closed-form solution for options with stochastic volatility with applications to bond and currency options", Rev. Finan. Stud. Vol. 6 (1993)

.. [HOUT2010] Hout and Foulon, "ADI Finite Difference Schemes for Option Pricing in the Heston Model with correlation", International Journal of Numerical Analysis and Modeling, Vol 7, Number 2 (2010).

.. [ANDERSON2005] Anderson, L, "Efficient Simulation of the Heston Stochastic Volatility Model", (2005).
