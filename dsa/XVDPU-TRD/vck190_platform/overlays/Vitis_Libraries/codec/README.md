# Codec Library

Codec Library is an open-sourced library written in C/C++ accelerating image processing including 2 APIs, JPEG decoder and PIK encoder. It now covers a level of acceleration: the module level(L1) and the pre-defined kernel level(L2).

## Overview

The algorithms implemented by Codec Library include:

*  API ‘jpegDec’: This API supports the ‘Sequential DCT-based mode’ of ISO/IEC 10918-1 standard. It is a high-performance implementation based-on Xilinx HLS design methodology. It can process 1 Huffman token and create up to 8 DCT coefficients within one cycle. It is also an easy-to-use decoder as it can directly parse the JPEG file header without help of software functions. 
*  API ‘pikEnc’: This API is based on Google’s PIK, which was ‘chosen as the base framework for JPEG XL’. The pikEnc used the ‘fast mode’ of PIK encoder which can provide better encoding efficiency than most of other still image encoding methods. The pikEnc is based on Xilinx HLS design methodology and optimized for FPGA architecture. It can proved higher throughput and lower latency compared to software-based solutions.

## Benchmark Result

In `L2/demos`, thest Kernels are built into xclbin targeting U200. We achieved a good performance against several dataset, e.g. lena.png with latency of "value". For more details about the benchmarks, please kindly find them in [benchmark results](https://xilinx.github.io/Vitis_Libraries/codec/2021.1/benchmark.html).


## Documentations

For more details of the Codec library, please refer to [xf_codec Library Documentation](https://xilinx.github.io/Vitis_Libraries/codec/2021.1/index.html).

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


