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

#ifndef _XF_WJ_KERNEL_HPP_
#define _XF_WJ_KERNEL_HPP_

#include <ap_int.h>
#include <hls_stream.h>
#include "write_json.hpp"

#define W_IN 64
#define W_OUT 256

extern "C" void WJ_kernel(ap_uint<W_IN>* cfgBuff,
                          ap_uint<W_IN>* msgBuff,
                          ap_uint<16>* msgLenBuff,
                          ap_uint<32>* msgPosBuff,
                          ap_uint<256>* geoBuff,
                          ap_uint<64>* geoLenBuff,
                          ap_uint<32>* geoPosBuff,
                          ap_uint<W_OUT>* outJson);
#endif
