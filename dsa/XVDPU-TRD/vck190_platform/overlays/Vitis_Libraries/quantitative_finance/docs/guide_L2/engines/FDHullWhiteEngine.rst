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
   :keywords: Finite-Difference, FDM, Bermudan Swaption, Hull-White
   :description: The pricing engine is based on Finite-difference methods (FDM) to estimate the value of Bermudan Swaption with an assumption that the floating rate at each time point conforms to Hull-White model.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


********************************************************************************
Internal Design of Finite-Difference Hull-White Bermudan Swaption Pricing Engine
********************************************************************************

Overview
========

Using the Finite-difference methods (FDM) to estimate the value of Bermudan Swaption. Here, we assume that the floating rate at each time point conforms to Hull-White model.

In Bermudan swaption, the owner is allowed to enter the swap on several pre-specified dates, usually coupon dates of the underlying swap. Notice that we evaluate the value of the swaption as a payer who pay the fixed leg and receive the floating leg of the interest rates.


Implementation
==============

The pricing process of Finite-Difference Hull-White Bermudan Swaption engine is shown in the figure below:

.. image:: /images/fd_hullwhite_bermudan_swaption_engine_workflow.png
        :alt: pricing process of FdHullWhiteEngine
        :width: 100%
        :align: center

As we can see from the figure, the engine has two main modules: engineInitialization and rollbackImplementation. The former one is responsible for engine initialization, including read specific time points of the swaption from DDR, set initial values, create the mesher using Ornstein-Uhlenbeck process, and build up the second-order differential operator.

After Initialization, the pricing engine evolves back step by step from the last exercise date (maturity) to settlement date (typically :math:`t=0`) using FDM, with the :math:`StepDistance=\frac{maturity - 0}{tGrid}`. Notice that when hitting an exercise time, the engine automatically evolves back from :math:`now` to the current exercise time. Meanwhile, the asset price which is stored in :math:`array\_` of the engine should be the maximum between its continuation (result evolved by douglasSchemeStep) and the intrinsic value (result calculated with current interest rates at the current exercise time by applyTo), then continue evolving back from current exercise time to the :math:`next` time point.

.. image:: /images/fd_hullwhite_engine_evolveback_process.png
        :alt: evolveback process of FdHullWhiteEngine
        :width: 60%
        :align: center

Since engineInitialization process will be executed for only once, while applyTo process will run :math:`\_ETSize` times in a single pricing process, additionally, both of them have a latency which is much shorter than douglasSchemeStep process, so they’re optimized for minimum resource utilizations with a reasonable overall latency. But as with douglasSchemeStep process, we try our best to decrease its latency to reduce the whole latency in the pricing process.

Mesher
======

In order to describe the desired range for the underlying value, we utilize the mesher which is stored in :math:`locations\_` of the engine to store the discretization of one dimension of the problem domain. In our implementation, we employ Ornstein-Uhlenbeck process to generate the mesher, a sketch of the mesher is shown as the following figure:

.. image:: /images/fd_hullwhite_engine_mesher.png
        :alt: mesher of FdHullWhiteEngine
        :width: 60%
        :align: center

Differential operator
=====================

In finite-difference methods, a differential operator :math:`D` is used to transform a function :math:`f(x)` into one of its derivatives, for instance :math:`{f}'(x)` or :math:`{f}''(x)`. As differentiation is linear, so it can be written as a matrix while using linear algebra methods to solve it. As you know, FDM doesn’t give the exact discretization of the derivative but an approximation, like :math:`{f}'_{i}={f}'(x_{i})+\epsilon _{i}`, notice that the error will decreasing along with the decreasing of the spacing of the grid, say, the tighter the grids (including :math:`t` axis and :math:`x` axis), the better approximation quality, but the worse simulation duration.

As you may refer to Tylor’s polynomial, we just provide differential operators including the first and the second derivative to obtain a manageable approximation error, they can be defined as:

.. math::
        {f}'{(x_{i})}\approx \frac{f(x_{i+1})-f(x_{i-1})}{2(x_{i}-x_{i-1})}
.. math::
        {f}''{(x_{i})}\approx \frac{f(x_{i+1})-2f(x_{i})+f(x_{i-1})}{(x_{i}-x_{i-1})^{2}}

As we can see from the equations that the value of the derivative at any given index :math:`i` only determined by the adjacent three values of the function with the middle of the same index, thus the differential operator can be written as a tridiagonal matrix, like:

.. math::
        \begin{bmatrix}
        m_{0} & u_{0} &  &  &  &  & \\ 
        l_{0} & m_{1} & u_{1} &  &  &  & \\ 
        & l_{1} & m_{2} & u_{2} &  &  & \\ 
        &  & ... & ... & ... &  & \\ 
        &  &  & l_{n-4} & m_{n-3} & u_{n-3} & \\ 
        &  &  &  & l_{n-3} & m_{n-2} & u_{n-2}\\ 
        &  &  &  &  & l_{n-2} & m_{n-1}
        \end{bmatrix}

To save storage resources and avoid redundant computations, we store the upper, main, and lower diagonals of the matrix in the :math:`dzMap\_` of the pricing engine and compute it by Thomson algorithm while evolving back, instead of using a traditional matrix with a large number of zeros and many meaningless additions and multiplications in the pricing process. 

Evolution scheme
================

A partial differential equation (PDE) can be written as:

.. math::
        \frac{\partial f}{\partial t}=D\cdot f

As mentioned above, a differential operator is used to discretize the derivatives on the right-hand side, while the evolution scheme discretizes the time derivative on the left-hand side.

As you know, finite-difference methods in finance start from a known state :math:`f(T)`, where :math:`T` stand for the maturity of the swaption, and evolve backwards to settlement date :math:`f(0)`. At each time step, we need to evaluate :math:`f(t)` based on :math:`f(t+\Delta t)`.

The Explicit Euler (EE) scheme can be written as below:

.. math::
        \frac{f(t+\Delta t)-f(t)}{\Delta t}=D\cdot f(t+\Delta t)

Which can be simplified as:

.. math::
        f(t)=(I-\Delta t\cdot D)\cdot f(t+\Delta t)

That only simple matrix multiplication is needed to approximate the equation which is mentioned at the first of this subsection makes the EE scheme becomes the simplest one in FDM.

The Implicit Euler (IE) scheme can be written this way:

.. math::
        \frac{f(t+\Delta t)-f(t)}{\Delta t}=D\cdot f(t)

Simplified as:

.. math::
        f(t)=(I+\Delta t\cdot D)^{-1}\cdot f(t+\Delta t)

Which makes it a more complex scheme to approximate the PDE.

In our implementation, we adopt a generic template to support different schemes, which can be written as:

.. math::
        \frac{f(t+\Delta t)-f(t)}{\Delta t}=D\cdot [(1-\theta )\cdot f(t+\Delta t)+\theta \cdot f(t)]

The formula transforms to EE scheme if we set :math:`\theta =0`, and it transforms to IE scheme if we let :math:`\theta =1` instead. Any value from 0 to 1 can be used, for example, we give a :math:`\theta =\frac{1}{2}` to utilize the Crank-Nicolson scheme.


Profiling
=========

The hardware resource utilizations and timing performance for a single Finite-Difference Hull-White Bermudan Swaption prcing engine with :math:`\_xGridMax=101` are listed in :numref:`tab1FDHWU` below:

.. _tab1FDHWU:

.. table:: Hardware resources for single finite-difference Hull-White bermudan swaption pricing engine
    :align: center

    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |   BRAM   |    DSP   |    FF    |    LUT   |    CLB    |    SRL    | clock period(ns)|
    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |    10    |    487   |  167609  |  103349  |   21520   |   10834   |      3.245      |
    +----------+----------+----------+----------+-----------+-----------+-----------------+
