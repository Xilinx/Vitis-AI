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


*************************************************
Internal Design of HWA Engine
*************************************************

Implementation
==============

Zero Coupon Bond Price
**********************

The zero coupon bond price for Hull White One Factor Model is calculated using the following:

.. math::
        P(t,T) = A(t,T) e ^ {-B(t,T) r(t)}

.. math::
        B(t,T) = 1 - \frac {e ^ {-(T-t)} } {a}

.. math::
        lnA(t,T) = ln \frac{P(0,T)}{P(0,t)} - B(t,T) -\frac{\delta ln P(0,t)}{\delta t} - \frac {\sigma ^ {2}}{4a^{2}} (e ^ {-aT} - e ^ {-at}) ^ {2} (e ^ {2at} - 1)


These input parameters are:

:math:`a` - the mean reversion

:math:`\sigma` - the volatility

:math:`t` - the current time

:math:`T`- the maturity

:math:`r(t)` - the short rate at time t



Equity Option Pricing
*********************

The price at time t of a European Call option with strike X, maturity T on a discount bond maturing at time S is given by:

.. math::
        ZBC(t,T,S,X) = P(t,S)\theta(h) - XP(t,T)\theta(h-\sigma^{p})


The price at time t of a European Put option with strike X, maturity T on a discount bond maturing at time S is given by:

.. math::
        ZBP(t,T,S,X) = XP(t,T)\theta(-h+\sigma^{p}) - P(t,T)\theta(-h)


The terms are derived from:

.. math::
        {\sigma^{p}} = \sigma \sqrt{1 - \frac{e ^ {-2a(T-t)}}{2a}} B(T,S)

.. math::
        h = \frac{1}{\sigma^{p}} ln(\frac{P(t,S)}{P(t,T)X}) + \frac{\sigma ^ {p}}{2}



Cap/Floor
*********

ZBC & ZBP can be used to price caps & floors since they can be viewed as portfolios of zero-bond options:

.. math::
        Cap(t,T,N,X) = N \sum_{i=1}^{n}[P(t,t_{i-1})\theta(-h_i + \sigma_p^i) - (1+X_{Ti})P(t,t_i)\theta(-h_i)]


.. math::
        Flr(t,T,N,X) = N \sum_{i=1}^{n}[(1+X_{Ti})P(t,t_i)\theta(h_i)-P(t,t_{i-1})\theta(h_i-\sigma_p^i)]


The terms are derived from:

.. math::
        {\sigma_p^i} = \sigma{\sqrt{\frac{1-e^{2a(t_{i-1}-t)}}{2a}}}B(t_{i-1},t)

.. math::
        {h_i} = {\frac{1}{\sigma_p^i}} ln (\frac{P(t,t_i)(1+X_{Ti})}{P(t,t_{i-1})}) + \frac{\sigma_p^i}{2}


Implemention
============
The framework is split into host and kernel code.

Kernel
******
The kernel directory contains the 3 kernels based on the above formula:

- HWA_K0.cpp contains the bond pricing engine
- HWA_k1.cpp contains the option pricing engine
- HWA_k2.cpp contains the cap/floor engine


Host
****
The host code (*main.cpp*) contains the OpenCL calls to invoke each of the kernels and test for accuracy compared to the CPU model (*cpu.cpp*).




