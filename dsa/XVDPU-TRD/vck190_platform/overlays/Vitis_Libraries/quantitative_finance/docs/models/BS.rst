
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
   :keywords: Model, finance, Black-Scholes
   :description: The Black-Scholes Model is a mathematical model for the dynamics of a financial market containing derivative investment instruments. 
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


.. _black_scholes:

*******************
Black-Scholes Model
*******************

Overview
=========
The Black-Scholes Model is a mathematical model for the dynamics of a financial market containing derivative investment instruments (from wikipedia).

This section explains the stochastic differential equation (stock price process) in continue form and in discrete form (used as stock price process).
They are core part for option pricing. 

Black-Scholes Model
===================
The Continue form of Black-Scholes is:

.. math::
   dS_t = \mu S_t dt + \sigma S_t dz

where :math:`S_t` is the stock price at time :math:`t`. :math:`\mu` is the stock's expected rate of return. :math:`\sigma` is the volatility of the stock price.
The random variable :math:`z` follows Wiener process, i.e. :math:`z` satisfies the following equation.    

  1. The change of :math:`\Delta z` during a sample period of time :math:`\Delta t` is :math:`\Delta z = \epsilon \sqrt{\Delta t}`, where :math:`\epsilon` has a standardized normal distribution :math:`\phi(0,1)`. 
  2. The value of :math:`\Delta z` for any two different short intervals of time, :math:`\Delta t`, are independent.

It follows that :math:`\Delta z` has a normal distribution with the mean of :math:`0` and the variance of :math:`\Delta t`.

The Discrete form of Black-Scholes is:

.. math::
   \Delta S_t = \mu S_t \Delta t + \sigma S_t \epsilon \sqrt{\Delta t} => \frac{\Delta S_t}{S_t} = \mu \Delta t + \sigma \epsilon \sqrt{\Delta t}

The left side of the equation is the return provided by the stock in a short period of time, :math:`\Delta t`. The term :math:`\mu \Delta t` is the expected value of this return, 
and the :math:`\sigma \epsilon \sqrt{\Delta t}` is the stochastic component of the return. It is easy to see that :math:`\frac{\Delta S_t}{S_t} \sim \phi (\mu \Delta t, \sigma^2 \Delta t)`,
i.e. the mean of :math:`\frac{\Delta S_t}{S_t}` is :math:`\mu \Delta t` and the variance is :math:`\sigma^2 \Delta t`.

:math:`Ito` lemma and direct corollary
------------------------------------------

Suppose that the value of a variable :math:`x` follows the :math:`Ito` process

.. math::
        dx = a(x,t) dt + b(x,t) dz

where :math:`dz` is a Wiener process and :math:`a` and :math:`b` are functions of :math:`x` and :math:`t`. The variable :math:`x` has a drift rate of :math:`a` and a variance rate of :math:`b^2`. :math:`Ito` lemma shows that a function :math:`G(x,t)` follows the process

.. math::
        dG = (\frac{\partial G}{\partial x} a + \frac{\partial G}{\partial t} + \frac{1}{2} \frac{\partial^2 G}{\partial x^2} b^2) dt + \frac{\partial G}{\partial x} b dz.

Thus :math:`G` also follows an :math:`Ito` process, with a drift rate of :math:`\frac{\partial G}{\partial x} a + \frac{\partial G}{\partial t} + \frac{1}{2} \frac{\partial^2 G}{\partial x^2} b^2` and a variance rate of :math:`(\frac{\partial G}{\partial x} b)^2`.

In stock price process :math:`a(x,t)=\mu S` and :math:`b(x,t)=\sigma S`. 

Corollary: lognormal property of :math:`S`
-------------------------------------------
Let us define :math:`G = \ln S`. With :math:`Ito` lemma, we have 

.. math::
   dG = (\mu - \frac{\sigma^2}{2}) dt + \sigma dz,

using 

.. math::
   \frac{\partial G}{\partial S} = \frac{1}{S}, 
   \frac{\partial^2 G}{\partial S^2} = -\frac{1}{S^2},
   \frac{\partial G}{\partial t} = 0.

Thus

.. math::

        \ln S_T - \ln S_0 \sim \phi [(\mu-\frac{\sigma^2}{2})T, \sigma^2 T].

where :math:`S_T` is the stock price at a future time :math:`T`, :math:`S_0` is the stock price at time 0. In other words, :math:`S_T` has a lognormal distribution, which can take any value between 0 and :math:`+\infty`. 

Implementation of B-S model
--------------------------------

The discrete form of process for :math:`\ln S` is as:

.. math::
   \ln S(t+\Delta t) - \ln S(t) = (\mu - \frac{\sigma^2}{2})\Delta t + \sigma \epsilon \sqrt{\Delta t}

Its equivalent form is:

.. math::
   S(t+\Delta t) = S(t)\exp [(\mu - \frac{\sigma^2}{2})\Delta t + \sigma \epsilon \sqrt{\Delta t}]

The formula above is used to generate the path for :math:`S`. In order to optimize the multiplication of :math:`S` with adder operator in path pricer, in our implementation, the B-S path generator
will fetch the normal random number :math:`\epsilon` from RNG sequence and output the :math:`\ln S` to path pricer.

.. image:: /images/bs_1.PNG
   :alt: impl of BS
   :width: 80%
   :align: center

Because there is accumulation of :math:`\ln S`, the initiation interval (II) cannot achieve 1. Here, change order between paths and steps. Because the input random number are totally independent, 
the change of order will not affect the accurate of the result. The pseudo-code is shown as follows.


.. image:: /images/bs.PNG
   :alt: Optimization of Algorithm
   :width: 80%
   :align: center


