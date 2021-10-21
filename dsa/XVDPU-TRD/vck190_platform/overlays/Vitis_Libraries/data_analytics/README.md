# Vitis Data Analytics Library

Vitis Data Analytics Library is an open-sourced Vitis library written in C++ for accelerating
data analytics applications in a variety of use cases.

[Comprehensive documentation](https://xilinx.github.io/Vitis_Libraries/data_analytics/2020.2/index.html)

## API Categories

Three categories of APIs are provided by this library, namely:

* **Data Mining** APIs, including all most common subgroups:
  - _Classification_: decision tree, random forest, native Bayes and SVM algorithms.
  - _Clustering_: K-means algorithm.
  - _Regression_: linear, gradient and decision tree based algorithms.
* **Text Processing** APIs for unstructured information extraction and transformation. New in 2020.2 release!
  - _Regular expression_ with capture groups is a powerful and popular tool of information extraction.
  - _Geo-IP_ enables looking up IPv4 address for geographic information.
  - Combining these APIs, a complete demo has been developed to batch transform Apache HTTP server log into structured JSON text.
* **DataFrame** APIs, also new in 2020.2, can be used to store and load multiple types of data with both fixed and variable length into DataFrames.
  - The in-memory format follows the design principles of [Apache Arrow](https://arrow.apache.org/) with goal of allowing access
    without per-element transformation.
  - Loaders from common formats are planned to be added in future releases, for example CSV and JSONLine loaders.

## API Levels

Like most other Vitis sub libraries, Data Analytics Library also organize its APIs by levels.

* The bottom level, L1, is mostly hardware modules with its software configuration generators.
* The second level, L2, provides kernels that are ready to be built into FPGA binary and invoked with standard OpenCL calls.
* The top level, L3, is meant for solution integrators as pure software C++ APIs.
Little background knowledge of FPGA or heterogeneous development is required for using L3 APIs.

At each level, the APIs are designed to be as reusable as possible, combined with the corporate-friendly
Apache 2.0 license, advanced users are empowered to easily tailor, optimize and assemble solutions.


## Requirements

### Software Platform

Supported operating systems are RHEL/CentOS 7.4, 7.5 and Ubuntu 16.04.4 LTS, 18.04.1 LTS.

_GCC 5.0 or above_ is required for C++11/C++14 support.
With CentOS/RHEL 7.4 and 7.5, it could enabled via
[devtoolset-6](https://www.softwarecollections.org/en/scls/rhscl/devtoolset-6/).

### Development Tools

This library is designed to work with Vitis 2020.2,
and a matching version of XRT should be installed.

## Running Test Cases

This library ships two types of case: HLS cases and Vitis cases.
HLS cases can only be found in `L1/tests` folder, and are created to test module-level functionality.
Both types of cases are driven by makefiles.

### Shell Environment

For command-line developers the following settings are required before running any case in this library:

```console
source /opt/xilinx/Vitis/2020.2/settings64.sh
source /opt/xilinx/xrt/setup.sh
export PLATFORM_REPO_PATHS=/opt/xilinx/platforms
```

For `csh` users, please look for corresponding scripts with `.csh` suffix and adjust the variable setting command accordingly.

The `PLATFORM_REPO_PATHS` environment variable points to directories containing platforms.

### HLS Cases Command Line Flow

```console
cd L1/tests/hls_case_folder/

make run CSIM=1 CSYNTH=0 COSIM=0 VIVADO_SYN=0 VIVADO_IMPL=0 \
    DEVICE=/path/to/xilinx_u200_xdma_201830_2.xpfm
```

Test control variables are:

- `CSIM` for high level simulation.
- `CSYNTH` for high level synthesis to RTL.
- `COSIM` for co-simulation between software test bench and generated RTL.
- `VIVADO_SYN` for synthesis by Vivado.
- `VIVADO_IMPL` for implementation by Vivado.

For all these variables, setting to `1` indicates execution while `0` for skipping.

### Vitis Cases Command Line Flow

```console
cd L2/tests/vitis_case_folder

# build and run one of the following using U200 platform
make run TARGET=sw_emu DEVICE=/path/to/xilinx_u200_xdma_201830_2.xpfm

# delete generated files
make cleanall
```

Here, `TARGET` decides the FPGA binary type
- `sw_emu` is for software emulation
- `hw_emu` is for hardware emulation
- `hw` is for deployment on physical card. (Compilation to hardware binary often takes hours.)


## Benchmark Result

In `L2/benchmarks` and `L2/demo`, these Kernels are built into xclbins targeting Alveo U200/U250/U50. We achieved a good performance. For more details about the benchmarks, please kindly find them in [benchmark results](https://xilinx.github.io/Vitis_Libraries/data_analytics/2021.1/benchmark/benchmark.html).


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

