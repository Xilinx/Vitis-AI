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

#ifndef _XF_PREDICATE_KERNEL_HPP_
#define _XF_PREDICATE_KERNEL_HPP_

#include <ap_int.h>
#include <hls_stream.h>
#include "xf_data_analytics/text/two_gram_predicate.hpp"

typedef ap_uint<16> uint16;
typedef ap_uint<32> uint32;
typedef ap_uint<64> uint64;
typedef ap_uint<512> uint512;
typedef ap_uint<128> AXI_DT;

struct ConfigParam {
    int docSize;
    int fldSize;
};

extern "C" void TGP_Kernel(ConfigParam config,
                           uint8_t* field,
                           uint32_t* offset,
                           double* idfValue,
                           uint64_t* tfAddr,
                           AXI_DT* tfValue,
                           AXI_DT* tfValue2,
                           AXI_DT* tfValue3,
                           AXI_DT* tfValue4,
                           uint32_t* indexId);

#endif
