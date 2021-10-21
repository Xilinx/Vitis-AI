# Vitis Quantitative Finance Library

The Vitis Quantitative Finance Library is an fundamental library aimed at providing a comprehensive FPGA acceleration library for quantitative finance.
It is an open-sourced library for a variety of real use cases, such as modeling, trading, evaluation, risk-management and so on.

The Vitis Quantitative Finance Library provides comprehensive tools from the bottom up for quantitative finance. It includes the lowest level containing basic modules and functions, the middle level providing pre-defined kernels, and the third level as pure software APIs working with pre-defined hardware overlays.

* At the lowest level (L1), it offers useful statistical functions, numerical methods and linear algebra functions to support practical user to implement advanced modeling, such as Random Number Generator (RNG), Monte Carlo Simulation (MC), Singular Value Decomposition (SVD), unique matrix solvers and so on.

* In the middle level (L2), pricing engines are provided as kernel to evaluate common finance derivatives, such as equity products, interest-rate products, foreign exchange (FX) products, and credit products.

* The software API level (L3) wraps the details of offloading acceleration with pre-built binary (overlay) and allow users to accelerate supported pricing tasks on Alveo cards without engaging hardware development.

Check the [comprehensive HTML documentation](https://xilinx.github.io/Vitis_Libraries/quantitative_finance/2021.1/index.html) for more details.

## Library Contents

| Library Class    | Description | Layer |
|------------------|-------------|-------|
| MT19937 | Random number generator | L1 |
| MT2203 | Random number generator | L1 |
| MT19937IcnRng | Random number generator | L1 |
| MT19937BoxMullerNormalRng | Random number generator | L1 |
| MT2203IcnRng | Random number generator | L1 |
| MultiVariateNormalRng | Random number generator | L1 |
| SobolRsg | Multi-dimensional sobol sequence generator | L1 |
| SobolRsg1D | One-dimensional sobol sequence generator | L1 |
| BrownianBridge | Brownian bridge transformation using inverse simulation | L1 |
| TrinomialTree | Lattice-based trinomial tree structure | L1 |
| TreeLattice | Generalized structure compatible with different models and instruments | L1 |
| Fdm1dMesher | One-dimensional discretization | L1 |
| OrnsteinUhlenbeckProcess | A simple stochastic process derived from RNG | L1 |
| StochasticProcess1D | Stochastic process for CIR and ECIR models to simulate rate volatility | L1 |
| HWModel | Hull-White model for tree engine | L1 |
| G2Model | Two-additive-factor gaussian model for tree engine | L1 |
| ECIRModel | Extended Cox-Ingersoll-Ross model | L1 |
| CIRModel | Cox-Ingersoll-Ross model for tree engine | L1 |
| VModel | Vasicek model for tree engine | L1 |
| HestonModel | Heston process | L1 |
| BKModel | Black-Karasinski model for tree engine | L1 |
| BSModel | Black-Scholes process | L1 |
| PCA | Principal Component Analysis library | L1 |
| CPICapFloorEngine | Pricing Consumer price index (CPI) using cap/floor methods | L2 |
| DiscountingBondEngine | Engine used to price discounting bond | L2 |
| InflationCapFloorEngine | Pricing inflation using cap/floor methods | L2 |
| FdHullWhiteEngine | Bermudan swaption pricing engine using finite-difference methods based on Hull-White model | L2 |
| FdG2SwaptionEngine | Bermudan swaption pricing engine using finite-difference methods based on two-additive-factor gaussian model | L2 |
| DeviceManager | Used to enumerate available Xilinx devices | L3 |
| Device | A class representing an individual accelerator card | L3 |
| Trace | Used to control debug trace output | L3 |


| Library Function | Description | Layer |
|------------------|-------------|-------|
| svd | Jacobi singular value decomposition | L1 |
| mcSimulation | Monte-Carlo Framework implementation | L1 |
| pentadiagCr | Solves for u in linear system Pu = r | L1 |
| boxMullerTransform | Box-Muller transform from uniform random number to normal random number | L1 |
| inverseCumulativeNormalPPND7 | Inverse Cumulative transform from random number to normal random number | L1 |
| inverseCumulativeNormalAcklam | Inverse CumulativeNormal using Acklam’s approximation to transform uniform random number to normal random number | L1 |
| trsvCore | Tridiagonal linear solver | L1 |
| binomialTreeEngine | Binomial tree engine using Cox, Ross & Rubinstein | L2 |
| cfBSMEngine | Single option price plus associated Greeks | L2 |
| FdDouglas | Top level callable function to perform the Douglas ADI method | L2 |
| hcfEngine | Engine for Hestion Closed Form Solution | L2 |
| M76Engine | Engine for the Merton Jump Diffusion Model | L2 |
| MCEuropeanEngine | European Option Pricing Engine using Monte Carlo Method | L2 |
| MCEuropeanPriBypassEngine | Path pricer bypass variant | L2 |
| MCEuropeanHestonEngine | European Option Pricing Engine using Monte Carlo Method based on Heston valuation model | L2 |
| MCMultiAssetEuropeanHestonEngine | Multiple Asset European Option Pricing Engine using Monte Carlo Method based on Heston valuation model | L2 |
| MCAmericanEnginePreSamples | PreSample kernel: this kernel samples some amount of path and store them to external memory | L2 |
| MCAmericanEngineCalibrate | Calibrate kernel: this kernel reads the sample price data from external memory and use them to calculate the coefficient | L2 |
| MCAmericanEnginePricing | Pricing kernel | L2 |
| MCAmericanEngine | Calibration process and pricing process all in one kernel | L2 |
| MCAsianGeometricAPEngine | Asian Arithmetic Average Price Engine using Monte Carlo Method Based on Black-Scholes Model (geometric average version) | L2 |
| MCAsianArithmeticAPEngine | Asian Arithmetic Average Price Engine using Monte Carlo Method Based on Black-Scholes Model (arithmetic average version) | L2 |
| MCAsianArithmeticASEngine | Asian Arithmetic Average Strike Engine using Monte Carlo Method Based on Black-Scholes Model (arithmetic average version) | L2 |
| MCBarrierNoBiasEngine | Barrier Option Pricing Engine using Monte Carlo Simulation (using brownian bridge to generate the non-biased result) | L2 |
| MCBarrierEngine | Barrier Option Pricing Engine using Monte Carlo Simulation | L2 |
| MCCliquetEngine | Cliquet Option Pricing Engine using Monte Carlo Simulation | L2 |
| MCDigitalEngine | Digital Option Pricing Engine using Monte Carlo Simulation | L2 |
| MCEuropeanHestonGreeksEngine | European Option Greeks Calculating Engine using Monte Carlo Method based on Heston valuation model | L2 |
| MCHullWhiteCapFloorEngine | Cap/Floor Pricing Engine using Monte Carlo Simulation | L2 |
| McmcCore | Uses multiple Markov Chains to allow drawing samples from multi mode target distribution functions | L2 |
| treeSwaptionEngine | Tree swaption pricing engine using trinomial tree based on 1D lattice method | L2 |
| treeSwapEngine | Tree swap pricing engine using trinomial tree based on 1D lattice method | L2 |
| treeCapFloprEngine | Tree cap/floor engine using trinomial tree based on 1D lattice method | L2 |
| treeCallableEngine | Tree callable fixed rate bond pricing engine using trinomial tree based on 1D lattice method | L2 |
| hjmEngine | Heath-Jarrow-Morton framework pricing engine using Monte Carlo Simulation | L2 |

## Requirements

### Software Platform

This library is designed to work with Vitis 2021.1 and later, and therefore inherits the system requirements of Vitis and XRT.

Supported operating systems are RHEL/CentOS 7.4, 7.5 and Ubuntu 16.04.4 LTS, 18.04.1 LTS.
With CentOS/RHEL 7.4 and 7.5, C++11/C++14 should be enabled via
[devtoolset-6](https://www.softwarecollections.org/en/scls/rhscl/devtoolset-6/).

### PCIE Accelerator Card

Hardware modules and kernels are designed to work with Alveo cards. Specific requirements are noted against each kernel or demonstration. Hardware builds for Alveo board targets require package installs as per:
* [Alveo U200](https://www.xilinx.com/products/boards-and-kits/alveo/u200.html#gettingStarted)
* [Alveo U250](https://www.xilinx.com/products/boards-and-kits/alveo/u250.html#gettingStarted)

### Shell Environment

Setup the build environment using the Vitis and XRT scripts, and set the PLATFORM_REPO_PATHS to installation folder of platform files.

```console
    $ source <install path>/Vitis/2021.1/settings64.sh
    $ source /opt/xilinx/xrt/setup.sh
    $ export PLATFORM_REPO_PATHS=/opt/xilinx/platforms
```

## Design Flows

Recommended design flows are categorized by the target level:

* L1
* L2
* L3

The common tool and library prerequisites that apply across all design flows are documented in the requirements section above.

### L1

L1 provides the low-level primitives used to build kernels.

The recommended flow to evaluate and test L1 components is described as follows using the Vitis HLS tool. A top level C/C++ testbench (typically `main.cpp` or `tb.cpp`) prepares the input data, passes this to the design under test (typically `dut.cpp` which makes the L1 level library calls) then performs any output data post processing and validation checks.

A Makefile is used to drive this flow with available steps including `CSIM` (high level simulation), `CSYNTH` (high level synthesis to RTL), `COSIM` (cosimulation between software testbench and generated RTL), VIVADO_SYN (synthesis by Vivado), and VIVADO_IMPL (implementation by Vivado). The flow is launched from the shell by calling `make` with variables set as in the example below:

```console
# entering specific unit test project
$ cd L1/tests/specific_algorithm/
# Only run C++ simulation on U250
$ make run CSIM=1 CSYNTH=0 COSIM=0 VIVADO_SYN=0 VIVADO_IMPL=0 DEVICE=u250_xdma_201830_1
```

As well as verifying functional correctness, the reports generated from this flow give an indication of logic utilization, timing performance, latency and throughput. The output files of interest can be located at the location examples as below where the file names are correlated with the source code. i.e. the callable functions within the design under test.

    Simulation Log: <library_root>/L1/tests/bk_model/prj/solution1/csim/report/dut_csim.log
    Synthesis Report: <library_root>/L1/tests/bk_model/prj/solution1/syn/report/dut_csynth.rpt

### L2

L2 provides the pricing engine APIs presented as kernels.

The available flow for L2 based around the SDAccel tool facilitates the generation and packaging of pricing engine kernels along with the required host application for configuration and control. In addition to supporting FPGA platform targets, emulation options are available for preliminary investigations or where dedicated access to a hardware platform may not be available. Two emulation options are available, software emulation performs a high level simulation of the pricing engine while hardware emulation performs a cycle-accurate simulation of the generated RTL for the kernel. This flow is makefile driven from the console where the target is selected as a command line parameter as in the examples below:

```console
$ cd L2/tests/GarmanKohlhagenEngine

# build and run one of the following using U250 platform
#  * software emulation
#  * hardware emulation
#  * actual deployment on physical platform

$ make run TARGET=sw_emu DEVICE=u250_xdma_201830_1
$ make run TARGET=hw_emu DEVICE=u250_xdma_201830_1
$ make run TARET=hw DEVICE=u250_xdma_201830_1

# delete all xclbin and host binary
$ make cleanall
```

The outputs of this flow are packaged kernel binaries (xclbin files) that can be downloaded to the FPGA platform and host executables to configure and co-ordinate data transfers. The output files of interest can be located at the locations examples as below where the file names are correlated with the source code

    Host Executable: L2/tests/GarmanKohlhagenEngine/bin_#DEVICE/gk_test.exe
    Kernel Packaged Binary: L2/tests/GarmanKohlhagenEngine/xclbin_#DEVICE_#TARGET/gk_kernel.xclbin #ARGS

This flow can be used to verify functional correctness in hardware and enable real world performance to be measured.

### L3

L3 provides the high level software APIs to deploy and run pricing engine kernels whilst abstracting the low level details of data transfer, kernel related resources configuration, and task scheduling.

The flow for L3 is the only one where access to an FPGA platform is required.

A prerequisite of this flow is that the packaged pricing engine kernel binaries (xclbin files) for the target FPGA platform target have been made available for download or have been custom built using the L2 flow described above.

This flow is makefile driven from the console to initially generate a shared object (`L3/src/output/libxilinxfintech.so`).

```console
$ cd L3/src
$ source env.sh
$ make
```

The shared object file is written to the location examples as below:

    Library: L3/src/output/libxilinxfintech.so

User applications can subsequently be built against this library as in the example provided:

```console
$ cd L3/examples/MonteCarlo
$ make all
$ cd output

# manual step to copy or create symlinks to xclbin files in current directory

$ ./mc_example
```
## Benchmark Result
In `L2/benchmarks`, Kernels are built into xclbins targeting Alveo U200/U250. We achieved a good performance. For more details about the benchmarks, please kindly find them in [benchmark results](https://xilinx.github.io/Vitis_Libraries/quantitative_finance/tree/gh-pages/2021.1/benchmark/benchmark.html).
## Documentations
For more details of the Graph library, please refer to [Vitis Quantitative_Finance Library Documentation](https://xilinx.github.io/Vitis_Libraries/quantitative_finance/2021.1/index.html).

## License

Licensed using the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0).

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
    Copyright 2019 Xilinx, Inc.

## Contribution/Feedback

Welcome! Guidelines to be published soon.


