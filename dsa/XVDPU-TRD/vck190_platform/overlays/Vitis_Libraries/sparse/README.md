# Vitis SPARSE Library

Vitis SPARSE Library accelerates basic linear algebra functions for handling sparse matrices.

Vitis SPARSE Library is an open-sourced Vitis library written in C++ and released under
[Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0)
for accelerating sparse linear algebra functions in a variety of use cases.

The main target audience of this library is users who want to accelerate
sparse matrix vector multiplication (SpMV) with FPGA cards.
Currently, this library offers two levels of acceleration:

* At module level is for the C++ implementation of basic components used in SpMV functions. These implementations are intended to be used by HLS (High Level Synthesis) users to build FPGA logic for their applications. 
* The kernel level is for pre-defined kernels that are the C++ implementation of SpMV functions. These implementations are intended to demonstrate how FPGA kernels are defined and how L1 primitive functions can be used by any Vitis users to build their kernels for their applications. 
Check the [comprehensive HTML document](https://xilinx.github.io/Vitis_Libraries/sparse/2021.1/) for more details.

## Requirements

### FPGA Accelerator Card

Modules and APIs in this library work with Alveo U280 card.

* [Alveo U280](https://www.xilinx.com/products/boards-and-kits/alveo/u280.html)

### Software Platform

Supported operating systems are RHEL/CentOS 7.4, 7.5 and Ubuntu 16.04.4 LTS, 18.04.1 LTS.

_GCC 5.0 or above_ is required for C++11/C++14 support.
With CentOS/RHEL 7.4 and 7.5, C++11/C++14 should be enabled via
[devtoolset-6](https://www.softwarecollections.org/en/scls/rhscl/devtoolset-6/).

### Development Tools

This library is designed to work with Vitis 2021.1,
and a matching version of XRT should be installed.

## Benchmark Result

In `L2/benchmarks`, more details about the benchmarks, please kindly find them in [benchmark results](https://xilinx.github.io/Vitis_Libraries/blas/2021.1/user_guide/L2/L2_benchmark_spmv.html).

## Documentations
For more details of the sparse library, please refer to [sparse Library Documentation](https://xilinx.github.io/Vitis_Libraries/sparse/2021.1/index.html).

## License

Licensed using the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0).

    Copyright 2019-2021 Xilinx, Inc.
    
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
        http://www.apache.org/licenses/LICENSE-2.0
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
