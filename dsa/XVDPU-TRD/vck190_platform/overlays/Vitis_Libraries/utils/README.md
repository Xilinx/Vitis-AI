# Vitis Utility Library

Vitis Utility Library is an open-sourced Vitis library of common patterns of streaming and storage access.
It aims to assist developers to efficiently access memory in DDR, HBM or URAM, and perform data distribution, collection,
reordering, insertion, and discarding along stream-based transfer.

Check the [comprehensive HTML document](https://xilinx.github.io/Vitis_Libraries/utils/2020.2/index.html) for more details.

## Requirements

### Software Platform

Supported operating systems are RHEL/CentOS 7.4, 7.5 and Ubuntu 16.04.4 LTS, 18.04.1 LTS.

*GCC 5.0 or above* is required for C++11/C++14 support.
With CentOS/RHEL 7.4 and 7.5, C++11/C++14 should be enabled via
[devtoolset-6](https://www.softwarecollections.org/en/scls/rhscl/devtoolset-6/).

### Development Tools

This library is designed to work with Vitis 2020.2,
and a matching version of XRT should be installed.

## Design Flows

The common tool and library pre-requisites that apply across all design flows are documented in the requirements section above.

Recommended design flow is shown as follows:


### Shell Environment

Setup the build environment using the Vitis script, and set the installation folder of platform files via `PLATFORM_REPO_PATHS` variable.

```console
source /opt/xilinx/Vitis/2020.2/settings64.sh
export PLATFORM_REPO_PATHS=/opt/xilinx/platforms
```

Setting the `PLATFORM_REPO_PATHS` to installation folder of platform files can enable makefiles in this library to use the `DEVICE` variable as a pattern.
Otherwise, full path to .xpfm file needs to be provided through the `DEVICE` variable.

### Running HLS cases

L1 provides the modules to work distribution and result collection in different algorithms, manipulate streams:
including combination, duplication, synchronization, and shuffle, updates URAM array in tighter initiation internal (II).

The recommend flow to evaluate and test L1 components is described as follows using Vivado HLS tool.
A top level C/C++ testbench (typically `algorithm_name.cpp`) prepares the input data, passes them to the design under test,
then performs any output data post processing and validation checks.

A Makefile is used to drive this flow with available steps including:

* `CSIM` (high level simulation),
* `CSYNTH` (high level synthesis to RTL),
* `COSIM` (cosimulation between software testbench and generated RTL),
* `VIVADO_SYN` (synthesis by Vivado) and
* `VIVADO_IMPL` (implementation by Vivado).

The flow is launched from the shell by calling `make` with variables set as in the example below:

```console
cd L1/tests/specific_algorithm/
make run CSIM=1 CSYNTH=0 COSIM=0 VIVADO_SYN=0 VIVADO_IMPL=0 \
         DEVICE=/path/to/xilinx_u200_xdma_201830_2.xpfm
```

To enable more than C++ simulation, just switch other steps to `1` in `make` command line.
The default value of all these control variables are ``0``, so they can be omitted from command line
if the corresponding step is not wanted.

As well as verifying functional correctness, the reports generated from this flow give an indication of logic utilization,
timing performance, latency and throughput.
The output files of interest can be located at the location of the test project where the path name is "test.prj".

## License

Licensed using the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0).

    Copyright 2019-2020 Xilinx, Inc.
    
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
        http://www.apache.org/licenses/LICENSE-2.0
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

