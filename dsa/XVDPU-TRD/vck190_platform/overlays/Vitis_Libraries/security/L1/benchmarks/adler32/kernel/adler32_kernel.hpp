/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ADLER32_KERNEL_HPP_
#define __ADLER32_KERNEL_HPP_

#include <ap_int.h>
#include <hls_stream.h>
#include "xf_security/adler32.hpp"

#define W 16
#define K (512 / 8 / W)

extern "C" void Adler32Kernel(
    int num, int size, ap_uint<32>* len, ap_uint<32>* adler, ap_uint<512>* inData, ap_uint<32>* adler32Result);

#endif // __ADLER32_KERNEL_HPP_
