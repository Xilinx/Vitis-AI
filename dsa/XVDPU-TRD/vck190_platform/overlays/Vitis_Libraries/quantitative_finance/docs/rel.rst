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
   :keywords: Finance, Library, Vitis Quantitative Finance Library, fintech
   :description: Vitis Quantitative Finance library release notes.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


Release Note
============

Version 0.5
-----------


Vitis Quantitative Finance Library 0.5 provides engines and primitives for the acceleration of quantitative financial applications on FPGA. It comprises two approaches to pricing:

* A family of 10 Monte-Carlo based engines for 6 equity options (including European and American options) using Black-Scholes and Heston models; all of these pricing engines are based on a provided generic Monte Carlo simulation API, and work in parallel due to their streaming interface;

* A finite-difference PDE solver for the Heston model with supporting application code and APIs.

In addition, the library supports low-level functions, such as random number generator (RNG), singular value decomposition (SVD), and tridiagonal and pentadiagonal matrix solvers.


Version 1.0
-----------


Vitis Quantitative Finance library 1.0 provides engines and primitives for the acceleration of quantitative financial applications on FPGA. It comprises two approaches to pricing:

* A family of Trinomial-Tree based pricing engines for 4 interest rate derivatives (including swaption, swap, cap/floor and callable bond), using 6 short-term interest rate models (including Hull-White, Two-additive-factor gaussian, Vasicek, Cox-Ingersoll-Ross, Extended Cox-Ingersoll-Ross and BlackKarasinski). All of these pricing engines are based on a provided generic Trinomial-Tree Framework.

* 2 Finite-difference method based pricing engines for swaption, using Hull-White model and Two-additive-factor gaussian model. 1 Monte-Carlo based pricing engine for cap/floor, using Hull-White model, based on the Monte-Carlo simulation API we provided in release 0.5. 

* 3 close form pricing engine for inflation cap/floor, CPI cap/floor, and discounting bond.

