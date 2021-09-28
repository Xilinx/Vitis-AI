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
   :keywords: Finite Difference, PDE, Heston Pricing Model, Hout, Foulon
   :description: Finite Difference Methods are a family of numerical techniques to solve partial differential equations (PDEs). 
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


*************************
Finite Difference Methods
*************************

.. toctree::
   :maxdepth: 1

Overview
========

Finite Difference Methods are a family of numerical techniques to solve partial differential equations (PDEs). They do this by discretizing the continuous equation in the spatial dimensions (forming a multidimensional grid), and then iteratively evolving the system over a series of N discrete time steps. The result is a discrete solution for each individual grid point.

In the provided solver, the PDE is the Heston Pricing Model [HESTON1993]_. Using the naming conventions of Hout & Foulon [HOUT2010]_, the PDE is given as:

.. math::
   \frac{\partial u}{\partial t} = \tfrac{1}{2}s^2v\frac{\partial^{2} u}{\partial s^2} + \rho\sigma sv\frac{\partial^{2} u}{\partial s\partial v} + \tfrac{1}{2}\sigma^2v\frac{\partial^{2} u}{\partial v^2} + (r_d - r_f)s\frac{\partial u}{\partial s} + \kappa(\eta - v)\frac{\partial u}{\partial v} - r_d u

Where:

:math:`u` - the price of the European option;

:math:`s` - the underlying price of the asset;

:math:`v` - the volatility of the underlying price;

:math:`\sigma` - the volatility of the volatility;

:math:`\rho` - the correlation of Weiner processes;

:math:`\kappa` - the mean-reversion rate;

:math:`\eta` - the long term mean.

The Heston PDE then describes the evolution of an option price over time (:math:`t`) and a solution of this PDE results in the specific option price for an :math:`(s,v)` pair for a given maturity date :math:`T`. The finite difference solver maps the :math:`(s,v)` pair onto a 2D discrete grid, and solves for option price :math:`u(s,v)` after :math:`N` time-steps.

Implementation
==============

The included implementation uses a Douglas `Alternating Direction Implicit (ADI) method`_ to solve the PDE [DOUGLAS1962]_. Implicit methods are needed to provide a stable and useful solution, and the Douglas solver is the most direct of these methods.

.. _Alternating Direction Implicit (ADI) method: https://en.wikipedia.org/wiki/Alternating_direction_implicit_method

.. image:: /images/fd.png
   :alt: FD Heston Dataflow
   :width: 100%
   :align: center


Assumptions/Limitations
-----------------------

The following limitations apply to this implementation:

- The dividend yield term (:math:`r_f` above) is fixed to zero;
- Boundary conditions are fixed in time.

Dataflow Description
--------------------

Precalculate algorithm fixed matrices
+++++++++++++++++++++++++++++++++++++

A function in the host software generates the finite-difference grid (possibly non-uniform) and the :math:`\alpha`, :math:`\beta`, :math:`\gamma` and :math:`\delta` coefficients based on the grid. These are converted into a set of :math:`\mathbf{A}` matrices (in sparse form) and :math:`\mathbf{b}` boundary vectors, plus modified forms of the :math:`\mathbf{A}` matrices for the tridiagonal and pentadiagonal solvers (the :math:`\mathbf{X}` matrices). The initial condition of the grid is also calculated on the host, and all of these cofficients are moved into the DDR memory on the Alveo card.

The following three steps are performed in hardware on the Alveo card for each timestep of the simulation.

Explicit estimation at timestep t
+++++++++++++++++++++++++++++++++

A direct estimate of the PDE at the next timestep is made using the current values for the grid coefficients.

Implicit correction in s
++++++++++++++++++++++++

The estimate is corrected using the new values in the PDE in the s direction. This takes the form of a multiplication by a tridiagonal matrix and then a subsequent tridiagonal solve.

Implicit correction in v
++++++++++++++++++++++++

The estimate is then corrected using the new values in the PDE in the v direction. This takes the form of a multiplication by a pentadiagonal matrix and then a subsequent pentadiagonal solve.

Extract price grid
++++++++++++++++++

On completion of the simulation of all timesteps on the Alveo card, the host moves the final grid values back from the DDR memory and then extracts the final pricing information from those grid values.


References
==========

.. [HESTON1993] Heston, "A closed-form solution for options with stochastic volatility with applications to bond and currency options", Rev. Finan. Stud. Vol. 6 (1993)

.. [HOUT2010] Hout and Foulon, "ADI Finite Difference Schemes for Option Pricing in the Heston Model with correlation", International Journal of Numerical Analysis and Modeling, Vol 7, Number 2 (2010).

.. [DOUGLAS1962] Douglas Jr., J, "Alternating direction methods for three space variables", Numerische Mathematik, 4(1), pp41-63 (1962).
