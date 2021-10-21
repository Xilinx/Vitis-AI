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

*******************************************
Monte-Carlo European Dow Jones Engine (DJE)
*******************************************

The MC European DJE is a modified Monte Carlo European Option Pricer that is designed specifically to calculate the 
Dow Jones Industrial Average (DJIA)




Normal MC
*********
In normal Monte Carlo operation, the following operations are done:

* Calculate the option price at a future date for each asset
* Calculate Payoff of each asset
* Derate payoff for each asset

MC DJE 
*******
The operation of the Dow Jones Engine is subtly different:

* Calculate the stock price for each stock
* Calculate the average of **ALL** the assets using a supplied Dow Divisor

Typically the DJIA is calculated using 30 specific stocks.  The **MCEuropeanDJE** class can take data arrays for any number of assets.






.. toctree::
   :maxdepth: 1

.. include:: ../../../rst_L3/class_xf_fintech_MCEuropeanDJE.rst

