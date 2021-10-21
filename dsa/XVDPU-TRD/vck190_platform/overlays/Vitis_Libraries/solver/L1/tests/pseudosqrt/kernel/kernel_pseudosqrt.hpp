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
#ifndef __KERNEL_PSEUDOSQRT__
#define __KERNEL_PSEUDOSQRT__
#include <hls_stream.h>
#include <ap_int.h>

#define unrollNm1 4
#define matSize 16
typedef double DT;
const int DTLen = 8 * sizeof(DT);
const int TO = 2;

extern "C" void kernel_pseudosqrt_0(int nrows,
                                    hls::stream<ap_uint<DTLen * TO> >& matIn,
                                    hls::stream<ap_uint<DTLen * TO> >& matOut);
#endif
