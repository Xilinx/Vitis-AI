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

*******************
Credit Default Swap
*******************

.. toctree::
   :maxdepth: 1

Overview
========

A Credit Default Swap (CDS) is a financial contract between two counterparties in which one party pays the other party credit protection against possible default of the underlying asset.

A series of premium payments are made at regular intervals, these payments continue to be made as long as the underlying asset survives and doesn't go into default.

The price of the contract is obtained by computing the sum of the present value of each leg (Premium Leg) and the sum of expected default payments (Protection Leg).


.. math::
        Premium Leg = \sum_{i=1}^N \pi.N.P(T_i).\delta t.DF_i

        Protection Leg = \sum_{i=1}^N N.(1-R).[P(T_{i-1}) - P(T_i)].DF_i

        \pi (CDS Spread) = \frac{Premium Leg}{Protection Leg}


For fair pricing these legs must be equal with :math:`\pi` being the price of the contract of the fair spread.

:math:`P(T)` - Probability of survival at time T

:math:`DF_i` - Discount Factor at time t

:math:`R` - Recovery Rate

:math:`N` - Notional Value

