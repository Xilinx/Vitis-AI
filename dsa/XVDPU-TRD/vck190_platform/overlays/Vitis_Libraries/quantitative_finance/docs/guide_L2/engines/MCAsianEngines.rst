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
   :keywords: Asian, pricing, engine, MCAsianAPEngine
   :description: Asian option pricing engine uses Monte Carlo simulation to estimate the value of the Asian option. Here, we assume the process of asset pricing applies to Black-Scholes process. 
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials



************************************************
Internal Design of Asian Option Pricing Engine
************************************************

Overview
========

The Asian option pricing engine uses Monte Carlo Simulation to estimate the value of the Asian option. Here, we assume the process of asset pricing applies to Black-Scholes process. 

Asian Option is kind of exotic option. The payoff is path dependent and it is dependent on the
average price of underlying asset over the settled period of time :math:`T`.

The payoff of Asian options is determined by the arithmetic or geometric average underlying price over some pre-set period of time. This is different from the case of usual European Option and American Option, where the payoff of the option depends on the price of the underlying at exercise. One advantage of Asian option is the relative cost of Asian option compared to American options. Because of the averaging feature, Asian options are typically cheaper than American options.    

The average price of underlying asset could be used as strike price or the underlying settlement price at the expiry time.
When the average price of underlying asset is used as the underlying settlement price at the expiry time, the payoff is calculated as follows:

   payoff of put option  = :math:`\max(0, K - A(T))`

   payoff of call option = :math:`\max(0, A(T) - K)`

When the average price of underlying asset is used as the strike price at the expiry time, the payoff is calculated as follows:

   payoff of put option  = :math:`\max(0, A(T) - S_T)`

   payoff of call option = :math:`\max(0, S_T - A(T))`

Where :math:`T` is the time of maturity, :math:`A(T)` is the average price of asset during time :math:`T`, 
:math:`K` is the fixed strike, :math:`S_T` is the price of underlying asset at maturity.

The average could be arithmetic or geometric, which is configurable. :math:`N` is the
number of discrete steps from :math:`0` to :math:`T`.

   Arithmetic average: :math:`A(T) = \frac{\sum_{i=0}^N S_i}{N}`

   Geometric average: :math:`A(T) = \sqrt[n]{\prod_{i=0}^N S_i}`


MCAsianAPEngine
================

The pricing process of Asian Arithmetic Pricing engine is as follows:

1. Generate independent stock paths by using Mersenne Twister Uniform MT19937 Random Number Generator (RNG) followed by Inverse Cumulative Normal Uniform Random Numbers.
2. Start at :math:`t = 0` and calculate the stock price of each path firstly in order to achieve initiation interval (II) = 1.
3. Calculate arithmetic average and geometric average value of each path.

.. math::
        Price_{Arithmetic} = \frac{1}{M+1} * \sum_{i=0}^{M} (S(i\Delta t))
.. math::
        Price_{Geometric} = exp{( \frac{1}{M+1} * \sum_{i=0}^{M} (\log{S(i\Delta t)}) )} 

4. Calculate the payoff difference of arithmetic average price and geometric average price.

.. math::
        Payoff_{gap} = \max(0,Strike - Price_{Arithmetic}) - \max(0,Price_{Geometric} - Strike)\> for \> put \> options
.. math::
        Payoff_{gap} = \max(Strike - Price_{Arithmetic}, 0) - \max(Price_{Geometric} - Strike, 0)\> for \> call \> options

5. Calculate the payoff off of geometric average price :math:`Payoff_{ref}` based on analytical method.
6. Payoff of average pricing is :math:`Payoff = Payoff_{ref} + Payoff_{gap}` 

The pricing architecture on FPGA can be shown as the following figure:

.. _my-figureAP:
.. figure:: /images/AP/framework.png
    :alt: McAsianAPEngine pricing architecture on FPGA
    :width: 60%
    :align: center
    
    
MCAsianASEngine
===============

The pricing process of Asian Arithmetic Strike engine is as follows:

1. Generate independent stock paths by using Mersenne Twister Uniform MT19937 Random Number Generator (RNG) followed by Inverse Cumulative Normal Uniform Random Numbers.
2. Start at :math:`t = 0` and calculate the stock price of each path firstly in order to achieve II = 1.
3. Accumulate the sum of lognormal stock price using the following analytical solution.

.. math::
        Price_{Arithmetic} = \frac{1}{M} * \sum_{i=0}^{M-1} (S_0*exp((i+1)*
.. math::
                    (riskFreeRate - dividend - 0.5 * volatility^2)*dt + \sum_{j=0}^{i} volatility*\sqrt{dt}*dw_j))

where :math:`M` is the total timesteps, :math:`dt` is the time interval
4. Calculate the final payoff by taking the strike price :math:`Strike_t` as the previous arithmetic average price :math:`\frac{\sum_{j=0}^i S_j}{M}`.

.. math::
        Payoff = \max(0,Strike_t - Price_t) \> for \> call \> options
.. math::
        Payoff = \max(0,Price_t - Strike_t) \> for \> put \> options

The pricing architecture on FPGA can be shown as the following figure:

.. _my-figureAS:
.. figure:: /images/AS/framework.png
    :alt: McAsianASEngine pricing architecture on FPGA
    :width: 60%
    :align: center
    
    
MCAsianGPEngine
================

The pricing process of Asian Geometric Pricing engine is as follows:

1. Generate independent stock paths by using Mersenne Twister Uniform MT19937 Random Number Generator (RNG) followed by Inverse Cumulative Normal Uniform Random Numbers.
2. Start at :math:`t = 0` and calculate the stock price of each path firstly in order to achieve II = 1.
3. Transfer the geometric average of stock price to sum of lognormal stock price.
4. Accumulate the sum of lognormal stock price using the following analytical solution.

.. math::
        Price_{Geometric} = S_0*exp((riskFreeRate - dividend - 0.5*volatility^2)dt)^\frac{n+1}{2} *
.. math::
                    exp(\sum_{i=0}^{n-1} \frac{n-i}{n}*volatility*\sqrt{dt}*dw_i)

5. Calculate the final payoff by using a fixed strike price.

.. math::
        Payoff = \max(0,Strike - Price_t) \> for \> call \> options
.. math::
        Payoff = \max(0,Price_t - Strike) \> for \> put \> options

The pricing architecture on FPGA can be shown as the following figure:

.. _my-figureGP:
.. figure:: /images/GP/framework.png
    :alt: McAsianGPEngine pricing architecture on FPGA
    :width: 60%
    :align: center
    
    
.. note::

    The 3 figures above shows the pricing part of McAsianAPEngine, McAsianASEngine and McAsianGPEngine respectively; the other parts, for example, PathGenerator, MCSimulation and other modules, are the same as in MCEuropeanEngine.


Profiling
=========

The hardware resources are listed in :numref:`tab1MCU`. The Arithmetic and Geometric Asian Engines demand similar amount of resources.

.. _tab1MCU:

.. table:: Hardware resources for single MCU
    :align: center

    +--------------------------+----------+----------+----------+----------+----------+-----------------+
    |          Engines         |   BRAM   |    DSP   | Register |    LUT   |  Latency | clock period(ns)|
    +--------------------------+----------+----------+----------+----------+----------+-----------------+
    | McAsianArithmeticAPEngine|    12    |    59    |   24664  |   26826  |   53276  |       3.423     |
    +--------------------------+----------+----------+----------+----------+----------+-----------------+
    | McAsianArithmeticASEngine|    12    |    65    |   26683  |   29362  |   53196  |       3.423     |
    +--------------------------+----------+----------+----------+----------+----------+-----------------+
    |  McAsianGeometricAPEngine|    10    |    61    |   24626  |   26657  |   53222  |       3.423     |
    +--------------------------+----------+----------+----------+----------+----------+-----------------+


:numref:`tab_CPU_vs_FPGA` shows the performance improvement in comparison with CPU-based Quantlib result (Tolerance = 0.02)

.. _tab_CPU_vs_FPGA:

.. table:: Comparison between CPU and FPGA
    :align: center

    +---------------------------------+-----------------+----------------+-----------------+
    |          Engines                | McAsianAPEngine | McAsianASEngine| McAsianGPEngine |
    +---------------------------------+-----------------+----------------+-----------------+
    | SampNum                         |    25951        |    33642       |   46805         |
    +---------------------------------+-----------------+----------------+-----------------+
    | CPU result                      |    1.98441      |    3.10669     |   3.28924       | 
    +---------------------------------+-----------------+----------------+-----------------+
    | CPU Execution time (us)         |    224911       |    310856      |   601068        |
    +---------------------------------+-----------------+----------------+-----------------+
    | FPGA result                     |    1.89144      |    3.0866      |   3.26228       | 
    +---------------------------------+-----------------+----------------+-----------------+
    | FPGA Kernel Execution time (us) |    563.05       |    734.42      |   830.130       |
    +---------------------------------+-----------------+----------------+-----------------+
    | FPGA SampNum                    |    28672        |    34816       |   49152         | 
    +---------------------------------+-----------------+----------------+-----------------+
    | FPGA E2E Execution time (ms)    |    1            |    1           |   1             |
    +---------------------------------+-----------------+----------------+-----------------+
    | Number of MCM                   |    2            |    2           |   4             |
    +---------------------------------+-----------------+----------------+-----------------+


:numref:`tab_Max_Performance` shows the max performance of McAsianEngines in one SLR of Xilinx xcu250-figd2104-2L-e (Vivado report).

.. _tab_Max_Performance:

.. table:: Hardware resources for max perforamnce
    :align: center

    +--------------------------+----------+----------+----------+----------+----------+--------------+---------------+
    |          Engines         |   BRAM   |    DSP   | Register |    LUT   |  Latency |  Frequency   | Max Unroll Num|
    |                          |          |          |          |          |          |    (MHz)     |               |
    +--------------------------+----------+----------+----------+----------+----------+--------------+---------------+
    | McAsianArithmeticAPEngine|   192    |   749    |  246325  |  205626  |   54056  |       300    |       16      |
    +--------------------------+----------+----------+----------+----------+----------+--------------+---------------+
    | McAsianArithmeticASEngine|   192    |   755    |  247524  |  207476  |   54109  |       300    |       16      |
    +--------------------------+----------+----------+----------+----------+----------+--------------+---------------+
    |  McAsianGeometricAPEngine|   160    |   781    |  242927  |  200106  |   54002  |       300    |       16      |
    +--------------------------+----------+----------+----------+----------+----------+--------------+---------------+



.. toctree::
   :maxdepth: 1

