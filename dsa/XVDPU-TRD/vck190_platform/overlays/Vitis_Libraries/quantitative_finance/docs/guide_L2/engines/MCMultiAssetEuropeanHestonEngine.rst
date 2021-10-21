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
   :keywords: MCMultiAssetEuropeanHestonEngine
   :description: MCMultiAssetEuropeanHestonEngine aims to calculate pay off of European option whose underlying asset is sum of multiple underlying assets. These assets may influence each other which means their volatility is not independent.   
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


***************************************************
Internal Design of MCMultiAssetEuropeanHestonEngine
***************************************************


Overview
========

Heston Model is the most classic model for stock price. 
European option is an option that can only be exercised at the expiration date.
MCMultiAssetEuropeanHestonEngine aims to calculate pay off of European option whose underlying asset is sum of multiple underlying assets.
These assets may influence each other which means their volatility is not independent.
We use a matrix to describe their correlations.
This engine uses this matrix so calculate random variables that have that correlation.
Then it uses large number of random samples to simulate stock prices' dynamic based on Heston Model.
And finally calculates value of option which use these stock as underlying assets.

Implementation
==============

This engine is similar to MCEuropeanHestonEngine.
The main difference is that it have additional stage to calculate random variables that have certain correlation.
Assume there're :math:`N` underlying asset, each asset needs :math:`2` random variable, :math:`2N` random variable in total.
The correlation matrix is and :math:`2N` by :math:`2N` matrix, but the right upper triangle is all zeros after LU decomposition, leaving only :math:`(2N + 1)N` non-zero elements. By cutting of none zeros elements, it will save nearly half DSPs to calculate correlated random variables.


.. image:: /images/mcht_masset.png
   :alt: Diagram of MCMultiAssetEuropeanHestonEngine
   :width: 80%
   :align: center

Optimization comes in two parts. 

- 1. The first and also the most is optimization of L1 functions. 
- 2. Save one call of cumulative distribution function in single underlying assets since it can get the value directly from RNGs. It may not work for multiple underlying assets because it will lose direct link between Gaussian random number and its corresponding uniform random number.

Variations 
==========

In this release we provide five variations of Heston Model implementation, 
including kDTFullTruncation, kDTPartialTruncation, kDTReflection, kDTQuadraticExponential and kDTQuadraticExponentialMartingale. 
The first three is relatively simple dealing with negative volatility. 
kDTQuadraticExponential and kDTQuadraticExponential Martingale use better approximation method to get result with better precision while taking more resources.

