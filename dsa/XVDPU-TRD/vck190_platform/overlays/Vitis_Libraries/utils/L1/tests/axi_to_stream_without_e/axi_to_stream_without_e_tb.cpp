/*
 * Copyright 2020 Xilinx, Inc.
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
#include "xf_utils_hw/axi_to_stream.hpp"

#define ELEM_WIDTH (32)
#define AXI_WIDTH (128)
#define BURST_LENTH (32)

#define ELEM_NUM (1 << 15)
#define BUF_DEPTH (ELEM_NUM / (AXI_WIDTH / ELEM_WIDTH))

typedef ap_int<ELEM_WIDTH> DType;

// ------------------------------------------------------------
// top functions for aligned data
void top_axi_to_stream(ap_uint<AXI_WIDTH> rbuf[BUF_DEPTH], const int num, hls::stream<DType>& ostrm) {
    // clang-format off
    ;
#pragma HLS INTERFACE m_axi port = rbuf offset = slave bundle = gmem_in1 \
    latency = 125 num_read_outstanding = 32 max_read_burst_length = 32
    // clang-format on

    xf::common::utils_hw::axiToStream<BURST_LENTH>(rbuf, num, ostrm);
}

#ifndef __SYNTHESIS__
#include <iostream>
#include <cstdlib>

int main() {
    int nerror = 0;

    // popolate the buffer
    ap_uint<AXI_WIDTH>* buf = (ap_uint<AXI_WIDTH>*)malloc(sizeof(ap_uint<AXI_WIDTH>) * BUF_DEPTH);

    DType* eptr = (DType*)buf;
    for (int i = 0; i < ELEM_NUM; i++) {
        eptr[i] = i;
    }

    // conduct the test
    hls::stream<DType> d_strm("d_strm");

    const int k_num = ELEM_NUM - 50;

    top_axi_to_stream(buf, k_num, d_strm);

    std::cout << "num elem total: " << ELEM_NUM << std::endl;
    std::cout << "num elem requested: " << k_num << std::endl;
    std::cout << "num elem read: " << d_strm.size() << std::endl;

    // verify the result
    for (int j = 0; j < k_num; ++j) {
        DType t = d_strm.read();
        DType r = eptr[j];
        if (t != r) {
            ++nerror;
            if (nerror < 20) {
                std::cout << "ERROR: read: " << t << ", ref: " << r << std::endl;
            }
        }
    }

    if (nerror) {
        std::cout << "\nFAIL: nerror= " << nerror << " errors found.\n";
    } else {
        std::cout << "\nPASS: no error found.\n";
    }

    free(buf);
    return nerror;
}

#endif
