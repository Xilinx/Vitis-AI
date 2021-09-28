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
#include "xf_utils_hw/axi_to_stream.hpp"

#define BUS_WIDTH (32)
#define AXI_WIDTH (128)
#define BURST_LENTH (32)

#define DATA_LEN (4 << 15)
#define BUF_DEPTH (DATA_LEN / (AXI_WIDTH / 8))

typedef ap_int<BUS_WIDTH> DType;

// ------------------------------------------------------------
// top functions for aligned data
void top_axi_to_char_stream(
    ap_uint<AXI_WIDTH> rbuf[BUF_DEPTH], hls::stream<DType>& ostrm, hls::stream<bool>& e_ostrm, int len, int offset) {
    // clang-format off
    ;
#pragma HLS INTERFACE m_axi port = rbuf offset = slave bundle = gmem_in1 \
    latency = 125 num_read_outstanding = 32 max_read_burst_length = 32
    // clang-format on

    xf::common::utils_hw::axiToCharStream<BURST_LENTH>(rbuf, ostrm, e_ostrm, len, offset);
}

#ifndef __SYNTHESIS__
#include <iostream>
#include <cstdlib>

int main() {
    int nerror = 0;

    // popolate the buffer
    ap_uint<AXI_WIDTH>* buf = (ap_uint<AXI_WIDTH>*)malloc(sizeof(ap_uint<AXI_WIDTH>) * BUF_DEPTH);

    char* cptr = (char*)buf;
    for (int i = 0; i < DATA_LEN; i++) {
        cptr[i] = (i % 15) + 1;
    }

    // conduct the test
    hls::stream<DType> d_strm("d_strm");
    hls::stream<bool> e_strm("e_strm");

    const int k_offset = 5; // by 5 chars.
    const int k_len = DATA_LEN - 50 - k_offset;

    top_axi_to_char_stream(buf, d_strm, e_strm, k_len, k_offset);

    std::cout << "len total: " << DATA_LEN << std::endl;
    std::cout << "len requested: " << k_len << std::endl;
    std::cout << "num bus word read: " << d_strm.size() << ", bus word size: " << sizeof(DType) << std::endl;
    std::cout << "bus byte total: " << sizeof(DType) * d_strm.size() << std::endl;

    // verify the result
    int j = 0;
    DType dptr = (DType*)(cptr + k_offset);
    while (!e_strm.read()) {
        DType t = d_strm.read();
        DType r;
        for (int c = 0; c < sizeof(DType); ++c) {
            r.range(7 + c * 8, c * 8) = *(cptr + k_offset + c + sizeof(DType) * j);
        }
        if (sizeof(DType) * j + sizeof(DType) < k_len) {
            // no garbase data
            if (t != r) {
                ++nerror;
                if (nerror < 20) {
                    std::cout << "ERROR: read: " << std::hex << t << ", ref: " << r << std::dec << std::endl;
                }
            }
        } else {
            int garbage_byte = sizeof(DType) - (k_len % sizeof(DType));
            if (t.range(BUS_WIDTH - 1 - 8 * garbage_byte, 0) != r.range(BUS_WIDTH - 1 - 8 * garbage_byte, 0)) {
                ++nerror;
                std::cout << "INFO: last word garbage_byte: " << garbage_byte << std::endl;
                std::cout << "ERROR: last word read: " << std::hex << t << ", ref: " << r << std::dec << std::endl;
            }
        }
        ++j;
    }
    if (d_strm.size() != 0) {
        ++nerror;
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
