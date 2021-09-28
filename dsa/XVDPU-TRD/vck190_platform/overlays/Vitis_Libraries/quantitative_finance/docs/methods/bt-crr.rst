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
   :keywords: Binomial Tree, Cox-Ross-Rubinstein, BOPM
   :description: The binomial tree is used to model the propagation of stock price in time towards a set of possibilities at the Expiration date, based on the stock Volatility.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


******************************************
Binomial Tree, Cox-Ross-Rubinstein, Method
******************************************

.. toctree::
   :maxdepth: 1

Overview
========

The Cox-Ross-Rubinstein Binomial Tree method is an instance of the `Binomial Options Pricing Model (BOPM)`_ , published originally by Cox, Ross and Rubinstein in their 1979 paper "Option 
Pricing: A Simplified Approach" [CRR1979]_.  

In this method, the binomial tree is used to model the propagation of stock price in time towards a set of possibilities at the Expiration date, based on the stock Volatility. For "N" time steps into which the model scenario duration is subdivided, there 
are N+1 possible stock prices at the expiration time.

Based on the N+1 Call or Put Option values at expiration, option values are backward-propagated to the initial time using step probabilities and the interest-rate, to obtain the Call or 
Put Option price. Comparing intermediate Call/Put values during back-propagation to stock prices allows American Option prices to be calculated. 

Cox-Ross-Rubinstein show that as N tends to ∞, the binomial European Put/Call solutions tend towards the Black-Scholes solutions.  (Both models make the same underlying assumptions.)  In an example where K = $35.00 and 
N = 150, they show the difference is less than $0.01.

In a later paper, Leisen & Reimer [LR1995]_ propose a method to increase the convergence speed of the CRR binomial lattice to converge faster.   

.. _Binomial Options Pricing Model (BOPM): https://en.wikipedia.org/wiki/Binomial_options_pricing_model

.. image:: /images/bt_crr_1.PNG
   :alt: CRR Binomial Tree
   :width: 60%
   :align: center

The diagram above shows an example of a binomial tree, where the number of time steps is :math:`n`. (Note that :math:`n` steps results in :math:`n + 1` separate propagated :math:`S` values after the `n`-th step.) 

At each step the initial stock price :math:`S_0` is propagated in an Up path and a Down path from each node, with Up and Down factors :math:`u` and :math:`d`. The "Up" probability 
is :math:`p`; Down is :math:`1 - p`.   

The equations in the diagram show the derivation, where :math:`\sigma` is the stock volatility, :math:`r` the "risk-free rate", :math:`t` the scenario duration and :math:`n` the number of time steps. The dividend
yield in the above is assumed to be zero and not included in the expression for :math:`p`, but may be included when required. 

(*Diagram source: Wikipedia article* `Binomial Options Pricing Model (BOPM)`_ .)


References
==========

.. [CRR1979] Cox, J. C., Ross, S. A., Rubinstein, M., "Option Pricing: A Simplified Approach", Journal of Financial Economics (1979)

.. [LR1995] Leisen, D., Reimer, M., "Binomial Models for Option Valuation - Examining and Improving Convergence", Rheinische Friedrich-Wilhelms-Universität, Bonn, (1995).

