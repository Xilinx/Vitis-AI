
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
   :keywords: Model, finance, Vasicek, VModel
   :description: The Vasicek Model is a mathematical model describing the evolution of interest rates.  
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

************************
Vasicek Model
************************

Overview
=========
In financial mathematics, the Vasicek Model is a mathematical model describing the evolution of interest rates. It is a type of one-factor short rate model as it describes interest rate movements as driven by only one source of market risk. The model can be used in the valuation of interest rate derivatives, and has also been adapted for credit markets (from Wiki).

As widely-used of the Hull-White model, the Vasicek Model is an outdated model.

Implementation
===================
This section mainly introduces the implementation process of short-rate and discount, which is applied in Tree Engine.
As a key part of Tree Engine, the class :math:`VModel` implements the single-factor Vasicek model to calculate short-rate and discount by utilizing continuous compounding. Here, the implementation process is introduced.

1. a) Since the short-rate at the current time point is independent from the short-rate at the previous time point, there is no need to calculate the short-rate in this module.
   b) For implementing the generic Tree framework, this model only performs the calculation of some trinomial tree related parameters.
2. The discount is calculated at time point :math:`t` with the duration :math:`dt` based on the short-rate.

