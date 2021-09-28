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
   :keywords: Vitis, Finance, Library, Vitis Quantitative Finance Library, fintech
   :description: The Vitis Quantitative Finance Library is a Vitis Library aimed at providing a comprehensive FPGA acceleration library for quantitative finance. It is an open-sourced library that can be used in a variety of financial applications, such as modeling, trading, evaluation and risk management.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

**********************************
Vitis Quantitative Finance Library
**********************************

The Vitis Quantitative Finance Library is a Vitis Library aimed at providing a comprehensive FPGA acceleration library for quantitative finance. 
It is an open-sourced library that can be used in a variety of financial applications, such as modeling, trading, evaluation and risk management.

The Vitis Quantitative Finance Library provides extensive APIs at three levels of abstraction:

* L1, the basic functions heavily used in higher level implementations. It includes statistical functions such as Random Number Generation (RNG), numerical methods, e.g., Monte Carlo Simulation, and linear algebra functions such as Singular Value Decomposition (SVD), and tridiagonal and pentadiagonal matrix solvers.

* L2, the APIs provided at the level of pricing engines. Various pricing engines are provided to evaluate different financial derivatives, including equity products, interest-rate products, foreign exchange (FX) products, and credit products. At this level, each pricing engine API can be seen as a kernel. The customers may write their own CPU code to call different pricing engines under the framework of OpenCL.  

* L3, the software level APIs. APIs of this level hide the details of data transfer, kernel related resources configuration, and task scheduling in OpenCL. Software application programmers may quickly use L3 high-level APIs to run various pricing options without touching the dependency of OpenCL tasks and hardware configurations. 
  
Library Contents
================

+------------------------------------------------------------------------------------------------+---------------------------+-------+
| Library Class                                                                                  | Description               | Layer |
+================================================================================================+===========================+=======+
| :ref:`MT19937 <cid-xf::fintech::mt19937>`                                                      | Random number generator   | L1    |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`MT2203 <cid-xf::fintech::mt2203>`                                                        | Random number generator   | L1    |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`MT19937IcnRng <cid-xf::fintech::mt19937icnrng>`                                          | Random number generator   | L1    |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`MT2203IcnRng <cid-xf::fintech::mt2203icnrng>`                                            | Random number generator   | L1    |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
|                                                                                                | Produces a normal         | L1    |
| :ref:`MT19937BoxMullerNormalRng <cid-xf::fintech::mt19937boxmullernormalrng>`                  | distribution from a       |       |
|                                                                                                | uniform one               |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`MultiVariateNormalRng <cid-xf::fintech::multivariatenormalrng>`                          | Random number generator   | L1    |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
|                                                                                                | Quasi-random number       | L1    |
| :ref:`SobolRsg <cid-xf::fintech::sobolrsg>`                                                    | generator                 |       |
|                                                                                                |                           |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
|                                                                                                | Quasi-random number       | L1    |
| :ref:`SobolRsg1D <cid-xf::fintech::sobolrsg1d>`                                                | generator                 |       |
|                                                                                                |                           |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
|                                                                                                | Brownian bridge           | L1    |
| :ref:`BrownianBridge <cid-xf::fintech::brownianbridge>`                                        | transformation using      |       |
|                                                                                                | inverse simulation        |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`TrinomialTree <cid-xf::fintech::trinomialtree>`                                          | Lattice-based trinomial   | L1    |
|                                                                                                | tree structure            |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`TreeLattice <cid-xf::fintech::treelattice>`                                              | Generalized structure     | L1    |
|                                                                                                | compatible with different |       |
|                                                                                                | models and instruments    |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
|                                                                                                | Discretization for finite |       |
| :ref:`Fdm1dMesher <cid-xf::fintech::fdm1dmesher>`                                              | difference method         | L1    |
|                                                                                                |                           |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
|                                                                                                | A simple stochastic       |       |
| :ref:`OrnsteinUhlenbeckProcess <cid-xf::fintech::ornsteinuhlenbeckprocess>`                    | process                   | L1    |
|                                                                                                |                           |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
|                                                                                                | 1-dimentional stochastic  |       |
| :ref:`StochasticProcess1D <cid-xf::fintech::stochasticprocess1d>`                              | process derived by RNG    | L1    |
|                                                                                                |                           |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`HWModel <cid-xf::fintech::hwmodel>`                                                      | Hull-White model for      | L1    |
|                                                                                                | tree engine               |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`G2Model <cid-xf::fintech::g2model>`                                                      | Two-additive-factor       | L1    |
|                                                                                                | gaussian model for        |       |
|                                                                                                | tree engine               |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`ECIRModel <cid-xf::fintech::ecirmodel>`                                                  | Extended Cox-Ingersoll-   | L1    |
|                                                                                                | Ross model                |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`CIRModel <cid-xf::fintech::cirmodel>`                                                    | Cox-Ingersoll-Ross model  | L1    |
|                                                                                                | for tree engine           |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`VModel <cid-xf::fintech::vmodel>`                                                        | Vasicek model for         | L1    |
|                                                                                                | tree engine               |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`HestonModel <cid-xf::fintech::hestonmodel>`                                              | Heston process            | L1    |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`BKModel <cid-xf::fintech::bkmodel>`                                                      | Black-Karasinski model    | L1    |
|                                                                                                | for tree engine           |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`BSModel <cid-xf::fintech::bsmodel>`                                                      | Black-Scholes process     | L1    |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`XoShiRo128PlusPlus <doxid-classxf_1_1fintech_1_1_xo_shi_ro128_plus_plus>`                | XoShiRo128PlusPlus        | L1    |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`XoShiRo128Plus <doxid-classxf_1_1fintech_1_1_xo_shi_ro128_plus>`                         | XoShiRo128Plus            | L1    |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`XoShiRo128StarStar <doxid-classxf_1_1fintech_1_1_xo_shi_ro128_star_star>`                | XoShiRo128StarStar        | L1    |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`BicubicSplineInterpolation <doxid-classxf_1_1fintech_1_1_bicubic_spline_interpolation>`  | Bicubic Spline            |       |
|                                                                                                | Interpolation             | L1    |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`CubicInterpolation <doxid-classxf_1_1fintech_1_1_cubic_interpolation>`                   | Cubic Interpolation       | L1    |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`BinomialDistribution <doxid-classxf_1_1fintech_1_1_binomial_distribution>`               | Binomial Distribution     | L1    |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`CPICapFloorEngine <cid-xf::fintech::cpicapfloorengine>`                                  | Pricing Consumer price    | L2    |
|                                                                                                | index (CPI) using         |       |
|                                                                                                | cap/floor methods         |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`DiscountingBondEngine <cid-xf::fintech::discountingbondengine>`                          | Engine used to price      | L2    |
|                                                                                                | discounting bond          |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`InflationCapFloorEngine <cid-xf::fintech::inflationcapfloorengine>`                      | Pricing inflation using   | L2    |
|                                                                                                | cap/floor methods         |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`FdHullWhiteEngine <cid-xf::fintech::fdhullwhiteengine>`                                  | Bermudan swaption pricing | L2    |
|                                                                                                | engine using finite-      |       |
|                                                                                                | difference methods based  |       |
|                                                                                                | on Hull-White model       |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`FdG2SwaptionEngine <cid-xf::fintech::fdg2swaptionengine>`                                | Bermudan swaption pricing | L2    |
|                                                                                                | engine using finite-      |       |
|                                                                                                | difference methods based  |       |
|                                                                                                | on two-additive-factor    |       |
|                                                                                                | gaussian model            |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`DeviceManager <cid-xf::fintech::devicemanager>`                                          | Used to enumerate         | L3    |
|                                                                                                | available Xilinx devices  |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`Device <cid-xf::fintech::device>`                                                        | A class representing an   | L3    |
|                                                                                                | individual accelerator    |       |
|                                                                                                | card                      |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`Trace <cid-xf::fintech::trace>`                                                          | Used to control debug     | L3    |
|                                                                                                | trace output              |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+

+------------------------------------------------------------------------------------------------+---------------------------+-------+
| Library Function                                                                               | Description               | Layer |
+================================================================================================+===========================+=======+
|                                                                                                | Singular Value            | L1    |
| :ref:`svd <cid-xf::fintech::svd>`                                                              | Decomposition using the   |       |
|                                                                                                | Jacobi method             |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`mcSimulation <cid-xf::fintech::mcsimulation>`                                            | Monte-Carlo Framework     | L1    |
|                                                                                                | implementation            |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
|                                                                                                | Solver for pentadiagonal  | L1    |
| :ref:`pentadiagCr <cid-xf::fintech::pentadiagcr>`                                              | systems of equations      |       |
|                                                                                                | using PCR                 |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`boxMullerTransform <cid-xf::fintech::boxmullertransform>`                                | Box-Muller transform from | L1    |
|                                                                                                | uniform random number to  |       |
|                                                                                                | normal random number      |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`inverseCumulativeNormalPPND7 <cid-xf::fintech::inversecumulativenormalppnd7>`            | Inverse Cumulative        | L1    |
|                                                                                                | transform from random     |       |
|                                                                                                | number to normal random   |       |
|                                                                                                | number                    |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`inverseCumulativeNormalAcklam <cid-xf::fintech::inversecumulativenormalacklam>`          | Inverse CumulativeNormal  | L1    |
|                                                                                                | using Acklam’s            |       |
|                                                                                                | approximation to transform|       |
|                                                                                                | uniform random number to  |       |
|                                                                                                | normal random number      |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
|                                                                                                | Solver for tridiagonal    | L1    |
| :ref:`trsvCore <cid-xf::fintech::trsvcore>`                                                    | systems of equations      |       |
|                                                                                                | using PCR                 |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`PCA <cid-xf::fintech::PCA>`                                                              | Principal Component       | L1    |
|                                                                                                | Analysis library          |       |
|                                                                                                | implementation            |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`bernoulliPMF <cid-xf::fintech::bernoullipmf>`                                            | Probability mass function | L1    |
|                                                                                                | for bernoulli distribution|       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`bernoulliCDF <cid-xf::fintech::bernoullicdf>`                                            | Cumulative distribution   | L1    |
|                                                                                                | function for bernoulli    |       |
|                                                                                                | distribution              |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`covCoreMatrix <cid-xf::fintech::covcorematrix>`                                          | Calculate the covariance  | L1    |
|                                                                                                | of the input matrix       |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`covCoreStrm <cid-xf::fintech::covcorestrm>`                                              | Calculate the covariance  | L1    |
|                                                                                                | of the input matrix       |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`covReHardThreshold <cid-xf::fintech::covrehardthreshold>`                                | Hard-thresholding         | L1    |
|                                                                                                | Covariance Regularization |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`covReSoftThreshold <cid-xf::fintech::covresoftthreshold>`                                | Soft-thresholding         | L1    | 
|                                                                                                | Covariance Regularization |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`covReBand <cid-xf::fintech::covreband>`                                                  | Banding Covariance        | L1    |
|                                                                                                | Regularization            |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`covReTaper <cid-xf::fintech::covretaper>`                                                | Tapering Covariance       | L1    | 
|                                                                                                | Regularization            |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`gammaCDF <cid-xf::fintech::gammacdf>`                                                    | Cumulative distribution   | L1    |
|                                                                                                | function for gamma        |       |
|                                                                                                | distribution              |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`linearImpl <cid-xf::fintech::linearimpl>`                                                | 1D linear interpolation   | L1    |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`normalPDF <cid-xf::fintech::normalpdf>`                                                  | Probability density       | L1    |
|                                                                                                | function for normal       |       |
|                                                                                                | distribution              |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`normalCDF <cid-xf::fintech::normalcdf>`                                                  | Cumulative distribution   | L1    |
|                                                                                                | function for normal       |       |
|                                                                                                | distribution              |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`normalICDF <cid-xf::fintech::normalicdf>`                                                | Inverse cumulative        | L1    |
|                                                                                                | distribution function     |       |
|                                                                                                | for normal distribution   |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`logNormalPDF <cid-xf::fintech::lognormalpdf>`                                            | Probability density       | L1    |
|                                                                                                | function for log-normal   |       |
|                                                                                                | distribution              |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`logNormalCDF <cid-xf::fintech::lognormalcdf>`                                            | Cumulative distribution   | L1    |
|                                                                                                | function for log-normal   |       |
|                                                                                                | distribution              |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`logNormalICDF <cid-xf::fintech::lognormalicdf>`                                          | Inverse cumulative        | L1    |
|                                                                                                | distribution function for |       |
|                                                                                                | log-normal distribution   |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`poissonPMF <cid-xf::fintech::poissonpmf>`                                                | Probability mass          | L1    |
|                                                                                                | function for poisson      |       |
|                                                                                                | distribution              |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`poissonCDF <cid-xf::fintech::poissoncdf>`                                                | Cumulative distribution   | L1    |
|                                                                                                | function for poisson      |       |
|                                                                                                | distribution              |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`poissonICDF <cid-xf::fintech::poissonicdf>`                                              | Inverse cumulative        | L1    |
|                                                                                                | distribution function     |       |
|                                                                                                | for poisson distribution  |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`binomialTreeEngine <cid-xf::fintech::binomialtreeengine>`                                | Binomial tree engine      | L2    |
|                                                                                                | using CRR                 |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`cfBSMEngine <cid-xf::fintech::cfbsmengine>`                                              | Single option price plus  | L2    |
|                                                                                                | associated Greeks         |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`FdDouglas <cid-xf::fintech::fddouglas>`                                                  | Top level callable        | L2    |
|                                                                                                | function to perform the   |       |
|                                                                                                | Douglas ADI method        |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`hcfEngine <cid-xf::fintech::hcfengine>`                                                  | Engine for Hestion        | L2    |
|                                                                                                | Closed Form Solution      |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`M76Engine <cid-xf::fintech::m76engine>`                                                  | Engine for the Merton     | L2    |
|                                                                                                | Jump Diffusion Model      |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
|                                                                                                | Monte-Carlo simulation of | L2    |
|                                                                                                | European-style options    |       | 
| :ref:`MCEuropeanEngine <cid-xf::fintech::mceuropeanengine>`                                    |                           |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`MCEuropeanPriBypassEngine <cid-xf::fintech::mceuropeanpribypassengine>`                  | Path pricer bypass variant| L2    |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
|                                                                                                | Monte-Carlo simulation of | L2    |
|                                                                                                | European-style options    |       | 
| :ref:`MCEuropeanHestonEngine <cid-xf::fintech::mceuropeanhestonengine>`                        | using Heston model        |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
|                                                                                                | Monte-Carlo simulation of | L2    |
|                                                                                                | European-style options    |       |
|                                                                                                | for multiple underlying   |       | 
| :ref:`MCmultiAssetEuropeanHestonEngine <cid-xf::fintech::mcmultiasseteuropeanhestonengine>`    | asset                     |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`MCAmericanEnginePreSamples <cid-xf::fintech::mcamericanenginepresamples>`                | PreSample kernel: this    | L2    |
|                                                                                                | kernel samples some amount|       |
|                                                                                                | of path and store them    |       |
|                                                                                                | to external memory        |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`MCAmericanEngineCalibrate <cid-xf::fintech::mcamericanenginecalibrate>`                  | Calibrate kernel: this    | L2    |
|                                                                                                | kernel reads the sample   |       |
|                                                                                                | price data from external  |       |
|                                                                                                | memory and use them to    |       |
|                                                                                                | calculate the coefficient |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`MCAmericanEnginePricing <cid-xf::fintech::mcamericanenginepricing>`                      | Pricing kernel            | L2    |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`MCAmericanEngine <cid-xf::fintech::mcamericanengine>`                                    | Calibration process and   | L2    |
|                                                                                                | pricing process all in    |       |
|                                                                                                | one kernel                |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`MCAsianGeometricAPEngine <cid-xf::fintech::mcasiangeometricapengine>`                    | Asian Arithmetic Average  | L2    |
|                                                                                                | Price Engine using Monte  |       |
|                                                                                                | Carlo Method Based on     |       |
|                                                                                                | Black-Scholes Model :     |       |
|                                                                                                | geometric average version |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`MCAsianArithmeticAPEngine <cid-xf::fintech::mcasianarithmeticapengine>`                  | arithmetic average version| L2    |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`MCAsianArithmeticASEngine <cid-xf::fintech::mcasianarithmeticasengine>`                  | Asian Arithmetic Average  | L2    |
|                                                                                                | Strike Engine using Monte |       |
|                                                                                                | Carlo Method Based on     |       |
|                                                                                                | Black-Scholes Model :     |       |
|                                                                                                | arithmetic average version|       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`MCBarrierNoBiasEngine <cid-xf::fintech::mcbarriernobiasengine>`                          | Barrier Option Pricing    | L2    |
|                                                                                                | Engine using Monte Carlo  |       |
|                                                                                                | Simulation                |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`MCBarrierEngine <cid-xf::fintech::mcbarrierengine>`                                      | Barrier Option Pricing    | L2    |
|                                                                                                | Engine using Monte Carlo  |       |
|                                                                                                | Simulation                |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`MCCliquetEngine <cid-xf::fintech::mccliquetengine>`                                      | Cliquet Option Pricing    | L2    |
|                                                                                                | Engine using Monte Carlo  |       |
|                                                                                                | Simulation                |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`MCDigitalEngine <cid-xf::fintech::mcdigitalengine>`                                      | Digital Option Pricing    | L2    |
|                                                                                                | Engine using Monte Carlo  |       |
|                                                                                                | Simulation                |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`MCEuropeanHestonGreeksEngine <cid-xf::fintech::mceuropeanhestongreeksengine>`            | European Option Greeks    | L2    |
|                                                                                                | Calculating Engine using  |       |
|                                                                                                | Monte Carlo Method based  |       |
|                                                                                                | on Heston valuation model |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`MCHullWhiteCapFloorEngine <cid-xf::fintech::mchullwhitecapfloorengine>`                  | Cap/Floor Pricing Engine  | L2    |
|                                                                                                | using Monte Carlo         |       |
|                                                                                                | Simulation                |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`McmcCore <cid-xf::fintech::mcmccore>`                                                    | Uses multiple Markov      | L2    |
|                                                                                                | Chains to allow drawing   |       |
|                                                                                                | samples from multi mode   |       |
|                                                                                                | target distribution       |       |
|                                                                                                | functions                 |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`treeSwaptionEngine <cid-xf::fintech::treeswaptionengine>`                                | Tree swaption pricing     | L2    |
|                                                                                                | engine using trinomial    |       |
|                                                                                                | tree based on 1D lattice  |       |
|                                                                                                | method                    |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`treeSwapEngine <cid-xf::fintech::treeswaptionengine-2>`                                  | Tree swap pricing engine  | L2    |
|                                                                                                | using trinomial tree      |       |
|                                                                                                | based on 1D lattice method|       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`treeCapFloprEngine <cid-xf::fintech::treeswapengine>`                                    | Tree cap/floor engine     | L2    |
|                                                                                                | using trinomial tree based|       |
|                                                                                                | on 1D lattice method      |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`treeCallableEngine <cid-xf::fintech::treeswapengine-2>`                                  | Tree callable fixed rate  | L2    |
|                                                                                                | bond pricing engine using |       |
|                                                                                                | trinomial tree based on   |       |
|                                                                                                | 1D lattice method         |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`hjmEngine <cid-xf::fintech::hjmEngine>`                                                  | Full implementation of    | L2    |
|                                                                                                | Heath-Jarrow-Morton       |       |
|                                                                                                | framework Pricing Engine  |       |
|                                                                                                | with Monte Carlo          |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`hjmMcEngine <cid-xf::fintech::hjmMcEngine>`                                              | Monte Carlo only          | L2    |
|                                                                                                | implementation of         |       |
|                                                                                                | Heath-Jarrow-Morton       |       |
|                                                                                                | framework Pricing Engine  |       | 
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`hjmPcaEngine <cid-xf::fintech::hjmPcaEngine>`                                            | PCA only implementation of| L2    |
|                                                                                                | Heath-Jarrow-Morton       |       |
|                                                                                                | framework                 |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+
| :ref:`lmmEngine <cid-xf::fintech::lmmEngine>`                                                  | LIBOR Market Model (BGM)  | L2    |
|                                                                                                | framework implementation. |       |
+------------------------------------------------------------------------------------------------+---------------------------+-------+

Shell Environment
=================

Setup the build environment using the Vitis and XRT scripts, and set the ``PLATFORM_REPO_PATHS`` to installation folder of platform files.

.. code-block:: bash

    source <vitis_path>/Vitis/2021.1/settings64.sh
    source <xrt_path>/xrt/setup.sh
    export PLATFORM_REPO_PATHS=<platform_path>/platforms

Design Flows
============

Recommended design flows are categorized by the target level:

* L1
* L2
* L3

The common tool and library prerequisites that apply across all design flows are documented in the requirements section above.

L1
--

L1 provides the low-level primitives used to build kernels.

The recommend flow to evaluate and test L1 components is described as follows using the Vitis HLS tool. A top level C/C++ testbench (typically `main.cpp` or `tb.cpp`) prepares the input data, passes this to the design under test (typically `dut.cpp` which makes the L1 level library calls) then performs any output data post processing and validation checks.

A Makefile is used to drive this flow with available steps including `CSIM` (high level simulation), `CSYNTH` (high level synthesis to RTL), `COSIM` (cosimulation between software testbench and generated RTL), VIVADO_SYN (synthesis by Vivado), and VIVADO_IMPL (implementation by Vivado). The flow is launched from the shell by calling `make` with variables set as in the example below:

.. code-block:: bash

    # entering specific unit test project
    cd L1/tests/specific_algorithm/
    # Only run C++ simulation on U250
    make run CSIM=1 CSYNTH=0 COSIM=0 VIVADO_SYN=0 VIVADO_IMPL=0 DEVICE=u250_xdma_201830_1

As well as verifying functional correctness, the reports generated from this flow give an indication of logic utilization, timing performance, latency and throughput. The output files of interest can be located at the location examples as below where the file names are correlated with the source code. i.e. the callable functions within the design under test.::

    Simulation Log: <library_root>/L1/tests/bk_model/prj/solution1/csim/report/dut_csim.log
    Synthesis Report: <library_root>/L1/tests/bk_model/prj/solution1/syn/report/dut_csynth.rpt

L2
--

L2 provides the pricing engine APIs presented as kernels.

The available flow for L2 based around the Vitis tool facilitates the generation and packaging of pricing engine kernels along with the required host application for configuration and control. In addition to supporting FPGA platform targets, emulation options are available for preliminary investigations or where dedicated access to a hardware platform may not be available. Two emulation options are available, software emulation performs a high level simulation of the pricing engine while hardware emulation performs a cycle-accurate simulation of the generated RTL for the kernel. This flow is makefile driven from the console where the target is selected as a command line parameter as in the examples below:

.. code-block:: bash

    cd L2/tests/GarmanKohlhagenEngine

    # build and run one of the following using U250 platform

    #  * software emulation
    make run TARGET=sw_emu DEVICE=u250_xdma_201830_1
    #  * hardware emulation
    make run TARGET=hw_emu DEVICE=u250_xdma_201830_1
    #  * actual deployment on physical platform
    make run TARET=hw DEVICE=u250_xdma_201830_1

    # delete all xclbin and host binary
    make cleanall

The outputs of this flow are packaged kernel binaries (xclbin files) that can be downloaded to the FPGA platform and host executables to configure and co-ordinate data transfers. The output files of interest can be located at the locations examples as below where the file names are correlated with the source code.::

    Host Executable: L2/tests/GarmanKohlhagenEngine/bin_#DEVICE/gk_test.exe
    Kernel Packaged Binary: L2/tests/GarmanKohlhagenEngine/xclbin_#DEVICE_#TARGET/gk_kernel.xclbin #ARGS

This flow can be used to verify functional correctness in hardware and enable real world performance to be measured.


L3
--

L3 provides the high level software APIs to deploy and run pricing engine kernels whilst abstracting the low level details of data transfer, kernel related resources configuration, and task scheduling.

The flow for L3 is the only one where access to an FPGA platform is required.

A prerequisite of this flow is that the packaged pricing engine kernel binaries (xclbin files) for the target FPGA platform target have been made available for download or have been custom built using the L2 flow described above.

This flow is makefile driven from the console to initially generate a shared object (``L3/src/output/libxilinxfintech.so``).

.. code-block:: bash

   cd L3/src
   source env.sh
   make


The shared object file is written to the example location as shown below::

    Library: L3/src/output/libxilinxfintech.so

User applications can subsequently be built against this library as in the example provided

.. code-block:: bash

   cd L3/examples/MonteCarlo
   make all
   cd output

   # manual step to copy or create symlinks to xclbin files in current directory

   ./mc_example


.. toctree::
   :maxdepth: 1


.. toctree::
   :caption: Library Overview 
   :maxdepth: 2

   overview.rst
   rel.rst


.. toctree::
   :caption: User Guide
   :maxdepth: 2

   models_and_methods.rst
   guide_L1/L1.rst
   guide_L2/L2.rst
   guide_L3/L3.rst


.. toctree::
   :caption: Benchmark 
   :maxdepth: 2
   
   benchmark.rst


