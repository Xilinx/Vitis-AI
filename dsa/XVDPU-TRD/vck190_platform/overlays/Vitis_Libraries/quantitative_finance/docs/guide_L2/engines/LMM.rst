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
*************************************
Internal Design of LIBOR Market Model
*************************************
Overview
========
Thie `LIBOR Market Model` (LMM) framework, also known as `Brace Gatarek Musiela` model (BGM) is a financial model of interest rates. The quantities that are modeled are a set of `forward rates` (also called forward LIBORs)
unlike the HJM framework which uses instantaneous forward rates. The main advantage of this model over HJM is that the forward rates are observable in the market and their volatilities are naturally linked to traded contracts.
Our implementation is modelled by a lognormal process and solved by Monte-Carlo simulation.

Design Structure
================
For a given tenor structure :math:`T_0,T_1,...,T_n` evenly spaced with :math:`\tau = T_{i+1} - T_i, \forall i=1,...,n`, and a number of factors :math:`F`,
we evolve the LIBOR rates for all :math:`n` maturities with the following stochastic equation:

.. math::
    L_i(T_{j+1})=L_i(T_j)exp[\tau_{j+1}(\mu_i(T_j)-\frac{1}{2}\sigma_i(T_j)^2)+\sigma_i(T_j)\sqrt{\tau_{i+1}}dW_i]

Where :math:`\sigma_i(t)` are calibrated volatilities, :math:`dW_i` is a Brownian motion scaled by the pseudo-sqrt of the correlations matrix
and :math:`\mu_i(t)` is the drift defined in terms of the volatilities and correlations between tenors:

.. math::
    dW_i=\sum_{k=1}^{F}\eta_{i,k}W_k

.. math::
    \mu_i(t)=-\sigma_i(t)\sum_{m=i+1}^{n}\frac{\tau_mL_m(t)\sigma_m(t)\tilde{\rho}_{i,m}}{1+\tau_iL_m(t)}

Calibration of the Model
========================

In order to correctly use the model, the tenor correlations and volatilities must be calibrated to market data. The correlation and volatilities matrix is an input to the model and can potentially acept any user-defined data,
but we also provide some common parametric functions that can generate suitable calibrations.

Correlation calibration
***********************

The instantaneous tenor correlation matrix :math:`\rho` is an input to the LMM framework. These correlation matrices should be in the form of symmetric, positive and monotonically decreasing, as that would be expected
from real correlations from the market data. Our implementation provides a family of parametric correlation functions that can be chosen. In order to avoid noise in calibration it is recommended to use as few parameters
as possible. With that in mind the following functions are available:

* One-parametric instantaneous correlation. User needs to specify :math:`\beta` with a value :math:`0 < \beta <= 1`:

.. math::
    \rho_{i,j}=e^{-\beta|T_i-T_j|}

* Two-parametric instantaneous correlation. User needs to specify :math:`\beta_0,\beta_1` with values :math:`0<\beta_0\beta_1<=1`:

.. math::
    \rho_{i,j}=\beta_0+(1-\beta_0)e^{-\beta_1|T_i-T_j|}

Once a correlation matrix is generated, a Principal Component Analysis will be performed to reduce the dimensionality of the data to :math:`F` factors, 
in order to calculate the reduced factor correlation matrix (:math:`\bar{\rho}`) and the pseudo-sqrt of the correlation matrix (:math:`\eta`) used in the LMM framework. The dimensionality reduction is applied as follows:

1. Calculate the :math:`F` factors loadings matrix of the correlation matrix :math:`\rho`:

.. math::
    L = pca\_loadings(F, \rho)

2. The :math:`\eta` matrix is the loadings matrix normalised by the standard deviation (sqrt of the covariance matrix's diagonal):

.. math::
    \eta_{i,j} = \frac{L_{i,j}}{\sqrt{diag(L\cdot L^T)_i}}

3. From matrix :math:`\eta`, we can reduce the dimensionality of the original data set to obtain :math:`\tilde{\rho}`:

.. math::
    \tilde{\rho} = \eta\cdot \eta^T


Volatility Calibration
**********************

For the calibration of volatilities, we provide a volatily generator with values that can be bootstrapped from a vector of caplet implied volatitilies obtained via the Black76 model.
Implied volatitilies are the values that, when put into the Black formula, return the price that the option currently has in the market. 
It is market practise to quote the price of a caplet for a tenor :math:`T_i` by just their implied volatily and not the actual price.
Our implementation uses a calibration formula that will bootstrap the implied volatilities into a stationary piecewise constant volatility vectors used by the LMM.
This implies that the calculated volatilities are identical for all fixed times to maturity and they change over time as time to maturity changes :math:`\gamma(t,T) = \gamma(T - t)`

In order to calibrate the model, our provided function we take a vector of implied caplet volatilities :math:`\hat{\sigma}_i` as input and will generate the volatility matrix as follows:

.. math::
    \sigma_i(t) = \sqrt{\frac{\hat{\sigma}_i^2T_i-\sum_{k=0}^{i-1}\sigma_k^2(t)\tau_k}{\tau_0}}

Since as we advance :math:`t` up to maturity time each tenor expires, our generated volatilities will take the form of a lower triangular matrix:

+----------------+-----------------------+---------------------------+---------------------------+-----+---------------------------+-----------------------+
|                | :math:`[0,T_0]`       | :math:`(T_0,T_1]`         | :math:`(T_1,T_2]`         | ... | :math:`(T_{n-2},T_{n-1}]` | :math:`(T_{n-1},T_n]` |
+----------------+-----------------------+---------------------------+---------------------------+-----+---------------------------+-----------------------+
| :math:`L_1(t)` | :math:`\sigma_1(T_0)` | expired                   | expired                   | ... | expired                   | expired               |
+----------------+-----------------------+---------------------------+---------------------------+-----+---------------------------+-----------------------+
| :math:`L_2(t)` | :math:`\sigma_2(T_0)` | :math:`\sigma_1(T_1)`     | expired                   | ... | expired                   | expired               |
+----------------+-----------------------+---------------------------+---------------------------+-----+---------------------------+-----------------------+
|       ...      |          ...          |            ...            |            ...            | ... |            ...            |          ...          |
+----------------+-----------------------+---------------------------+---------------------------+-----+---------------------------+-----------------------+
| :math:`L_n(t)` | :math:`\sigma_n(T_0)` | :math:`\sigma_{n-1}(T_1)` | :math:`\sigma_{n-2}(T_2)` | ... | :math:`\sigma_1(T_n)`     | expired               |
+----------------+-----------------------+---------------------------+---------------------------+-----+---------------------------+-----------------------+

Pricing Algorithms
==================

Currently we support several choices for pricing algorithms by default, but the model can accept any custom implementation as a parameter.
Each pricer will consume one generated LIBOR rates path and output a price, which will be accumulated and averaged out to provide the Monte-Carlo solution.

Cap Pricing
***********

A cap is a basket of caplets, where all caplets have the same strike (caprate). Each caplet will have a payoff at time :math:`T_1, T_2, ..., T_n`. The price of the cap will be the sum of all the caplets.
The pricing of caps with the LMM framework is interesting because we can use it to validate the model and the calibrations by comparing the output of the MonteCarlo simulation with the output from the analytical
Black76 model. Once we are satisfied with the results from the model, we can use the same parameters to compute the pricing of other options that don't have analytical formulas.

The general formula for the price of a cap with notional :math:`N` and caprate :math:`K` is given by:

.. math::
    Cap = \sum_{i=1}^n Caplet(T_i)

Analytically, we can use the Black76 formula to calculate the price of a caplet with:

.. math::
    Caplet_{Black76}(t) = P(t, T_{i+1})\tau_iN(L_i(t)\phi(d_1) - K\phi(d_2))

.. math::
    d_1 = \frac{log(\frac{L_i(t)}{K})+\frac{1}{2}\sigma^2t}{\sigma\sqrt{t}}, d_2 = d_1 - \sigma\sqrt{t}

.. math::
    \phi(t) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{t}e^{-\frac{1}{2}x^2}

.. math::
    P(t, T_i) = e^{-rT_i}, r = \frac{1}{\tau_i}log(1+\frac{1}{i}\sum_{k=1}^{i}(L_k(t)\tau_k))

With the LIBOR market model, we can calculate the price of a caplet with the following formula:

.. math::
    Caplet_{LMM}(t)=\tau_{t-1}N(L_t(t)-K)^+\frac{B(0)}{B(t+1)}

.. math::
    B(t) = [\prod_{k=t}^{n}(1+\tau_kL_k(t))]^{-1}

After generating enough paths, the average of all Cap prices with the LMM will converge to the value from the Black76 formula provided the model is correctly calibrated.

Ratchet Floater Pricing
***********************

A ratchet floater is a path dependent interest rate product. This option is a good example for the use of the LIBOR market model, since no analytic formula exists.
At each time :math:`T_i, i > 0`, the ratchet pays a coupon amount :math:`c_i`. The ratchet floater price is the sum of all coupons.

For a ratchet floater with notional :math:`N`, constant spreads :math:`X` and :math:`Y` and fixed cap :math:`\alpha` the price can be calculated with:

.. math::
    RFloater = \sum_{i=0}^{n}(N(\tau_i(L_i(T_i) + X) - c_i)\frac{B(0)}{B(i+1)})

.. math::
    c_i = c_{i-1} + min\{(\tau_i(L_i(T_i) + Y) - c_{i_1})^+, \alpha\}

.. math::
    c_1 = \tau_1(L_1(T_1) + Y)

.. math::
    B(t) = [\prod_{k=t}^{n}(1+\tau_kL_k(t))]^{-1}

This means that the coupon :math:`c_i` is at least as much as the previous coupon amount, but no more than the previous coupon plus a fixed constant :math:`N\alpha`

Ratchet Cap Pricing
*******************

Ratchet caps have a similar structure as standard caps. The main difference is that while caps have a fixed caprate for every caplet,
for ratchet caps we will have a variable caprate dependent on earlier LIBOR resets for every ratchet caplet plus a spread.

The price of a ratchet cap with notional :math:`N`, spread :math:`s` and initial spread :math:`\kappa_0` is given by the following formula:

.. math::
    RCap = \sum_{i=0}^{n}RCaplet(i)

.. math::
    RCaplet(i) = \tau_iN(L_i(T_i)-K_i)^+\frac{B(0)}{B(i+1)}

.. math::
    K_i=\begin{cases}s+\kappa_0 & i = 0\\L_i(T_i)+s & i > 0\end{cases}

.. math::
    B(t) = [\prod_{k=t}^{n}(1+\tau_kL_k(t))]^{-1}

Internal Architecture
=====================

The internal framework implementation allows to easely parallelise the generation and pricing of LIBOR rates matrices by modifying the `UN`
parameter. Each unrolled implementation will contain a RNG sequence generator for :math:`F` uncorrelated factors, a LMM path generator and a copy of
the chosen path pricer. Since the calibration data (:math:`\eta,\bar{\rho},\sigma`) is computed once and then read only, each MonteCarlo module will also contain
a copy of the accessed elements of those matrices.

The path generator will compute a set of LIBOR rates, which are of the form of a lower triangular matrix, with the following process:

.. image:: /images/lmm/LMM_PathGen.png
    :alt: LMM Path generation process
    :align: center

Each LIBOR rates path matrix will be fed via an HLS stream into the path pricer in a strided pattern: First the :math:`L_0` column, from :math:`T_0` to :math:`T_n`.
Then the :math:`L_1` column and so on.

It is responsibility of the path pricer to compute the option price and to consume the all the data fed from the path generator.

The full implementation of the LIBOR Market Model framework has the following architecture:

.. image:: /images/lmm/LMM_Architecture.png
    :alt: LMM Architecture
    :align: center