
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
   :keywords: Model, finance, Hull-White, HWModel
   :description: The Hull-White model is a model of future interest rates.  
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

*******************
Hull-White Model
*******************

Overview
=========
In financial mathematics, the Hull-White model is a model of future interest rates. In its most generic formulation, it belongs to the class of no-arbitrage models that are able to fit today's term structure of interest rates. It is relatively straightforward to translate the mathematical description of the evolution of future interest rates onto a tree or lattice and so interest rate derivatives such as Bermudan Swaptions can be valued in the model. The first Hull-White model was described by John C. Hull and Alan White in 1990. The model is still popular in the market today (from Wiki).

Implementation
===================
This section mainly introduces the implementation process of short-rate and discount, which is core part of option pricing, and applied in Tree Engine and FD (finite-difference method) Engine.

As an important part of the Tree/ FD Engines, the class :math:`HWModel` implements the single-factor Hull-White model to calculate short-rate and discount by using continuous compounding, including 4 functions (treeShortRate, fdShortRate, discount, discountBond). The implementation process is introduced as follows:

1. Function treeShortRate: 

   a) The short-rates is calculated at time point :math:`t` with the duration :math:`dt` from 0 to N point-by-point. As in the calculation process, the variable :math:`value` needs to be calculated first. To improve the initiation interval (II), an array :math:`values16` is implemented to store the intermediate results from each iteration. Then, an addition tree is performed subsequently to achieve an II = 1 for the whole process. Finally, the short rate is calculated using variable :math:`value`.

   b) For implementing the generic Tree framework, the :math:`state\_price` calculating process is moved from Tree Lattice to this Model.

2. Function fdShortRate: The short-rate is calculated at time point :math:`t`.
3. Function discount: The discount is calculated at time point :math:`t` with the duration :math:`dt` based on the short-rate.
4. Function discountBond: The discount bond is calculated at time point :math:`t` with the duration :math:`dt=T-t` based on the short-rate.

