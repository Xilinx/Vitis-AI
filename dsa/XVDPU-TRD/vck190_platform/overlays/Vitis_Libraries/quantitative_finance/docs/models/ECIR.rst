
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
   :keywords: Model, finance, Cox-Ingersoll-Ross, extended, ECIR
   :description: The Extended Cox-Ingersoll-Ross (ECIR) model is a mathematical model of the term structure of interest rates. 
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

*********************************
Extended Cox-Ingersoll-Ross Model
*********************************

Overview
=========

In financial mathematics, by replacing the coefficients in the Cox-Ingersoll-Ross model with the time varying functions, the Extended Cox-Ingersoll-Ross (ECIR) model is a mathematical model of the term structure of interest rates. It is a type of "one factor model" (short rate model) as it describes interest rate movements as driven by only one source of market risk. The model can be used in the valuation of interest rate derivatives (from Wiki).

Implementation
===================
This section mainly introduces the implementation process of short-rate and discount, which is applied in Tree Engine. They are core part for option pricing. 

As a critical part of Tree Engine, the class :math:`ECIRModel` implements the single-factor ECIR model to calculate short-rate and discount, by using continuous compounding. The implementation process is introduced as follows.

1. a) The short-rate is calculated at time point :math:`t` with the duration :math:`dt` from 0 to N point-by-point by functions treeShortRate, initRate and iterRate. As the core part of the treeShortRate, the outer loop_rateModel is used to ensure the results under pre-specified tolerance. For the internal functions, the functionality of initRate and iterRate is similar with each other, but initRate can produce 3 intermediate results while the iterRate gives only one per iteration. In order to achieve intiation interval (II)=1, the array values16 is added to store the intermediate results. Then an addtion tree is performed subsequently for the whole process.
   b) For implementing the generic Tree framework, the :math:`state\_price` calculating process is moved from Tree Lattice to this Model.
2. The discount is calculated at time point :math:`t` with the duration :math:`dt` based on the short-rate.

