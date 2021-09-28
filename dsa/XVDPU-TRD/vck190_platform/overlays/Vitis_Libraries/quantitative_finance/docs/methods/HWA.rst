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

****************************************
Hull White Analytic Closed-Form Solution
****************************************

.. toctree::
   :maxdepth: 1

Overview
========

In financial mathematics, the Hull-White model is a model of future interest rates and is an extension the Vasicek model.

Its an no-arbitrage model that is able to fit todays term structure of interest rates.

It assumes that the short-term rate is normally distributed and subject to mean reversion.


The stochastic differential equation describing Hull-White is:

.. math::
        \delta{r} = [\theta(t) - ar]\delta{t} + \sigma\delta{z}

These input parameters are:

:math:`\delta r` - is the change in the short-term interest rate over a small interval

:math:`\theta (t)` - is a function of time determining the average direction in which r moves (derived from yield curve)

:math:`a` - the mean reversion

:math:`r` - the short-term interest rate

:math:`\delta t` - a small change in time

:math:`\sigma` - the volatility

:math:`\delta z` - is a Wiener (Random) process
