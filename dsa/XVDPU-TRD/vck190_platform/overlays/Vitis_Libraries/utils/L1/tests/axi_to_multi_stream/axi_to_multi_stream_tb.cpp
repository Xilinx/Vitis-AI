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

#include "xf_utils_hw/axi_to_multi_stream.hpp"

#define AXI_SZ (64)
#define AXI_WIDTH (AXI_SZ * 8)
#define BURST_LENTH (32)

#define DT0_SZ (4)
#define DT1_SZ (4)
#define DT2_SZ (8)
typedef ap_int<8 * DT0_SZ> DT0;
typedef ap_int<8 * DT1_SZ> DT1;
typedef ap_int<8 * DT2_SZ> DT2;

#define ELEM_NUM (999)
#define ELEM_SPACE (1024)

#define BUF_DEPTH (ELEM_SPACE * (DT0_SZ + DT1_SZ + DT2_SZ) / AXI_SZ)

template <int _II, typename _TStrm>
void forward_data(hls::stream<_TStrm>& ostrm, hls::stream<bool>& e_ostrm, hls::stream<_TStrm>& r_strm) {
    bool e = e_ostrm.read();
    while (!e) {
#pragma HLS pipeline II = _II
        _TStrm dat = ostrm.read();
        r_strm.write(dat);
        e = e_ostrm.read();
    }
}

// top functions for co-sim
// 3 consumers of different rate are connected to test against deadlock.
void top_func(ap_uint<AXI_WIDTH> rbuf[BUF_DEPTH],
              hls::stream<DT0>& r_strm0,
              hls::stream<DT1>& r_strm1,
              hls::stream<DT2>& r_strm2) {
    // clang-format off
    ;
    int len[3] = {100 * DT0_SZ, 200 * DT2_SZ, 300 * DT2_SZ};
#pragma HLS ARRAY_PARTITION variable = len complete
    int offset[3] = {0 + 1, DT0_SZ * ELEM_SPACE + 2, DT0_SZ * ELEM_SPACE + DT1_SZ * ELEM_SPACE + 3};
#pragma HLS ARRAY_PARTITION variable = offset complete
#pragma HLS INTERFACE m_axi port = rbuf offset = slave bundle = gmem_in1 latency = 125 \
    num_read_outstanding = 32 max_read_burst_length = 32
#pragma HLS ARRAY_PARTITION variable = len complete
#pragma HLS ARRAY_PARTITION variable = offset complete
    ;
    // clang-format on

    for (int i = 0; i < 1; i++) {
#pragma HLS DATAFLOW

        hls::stream<bool> e_ostrm0;
        hls::stream<DT0> ostrm0;
        hls::stream<DT1> ostrm1;
        hls::stream<bool> e_ostrm1;
        hls::stream<DT2> ostrm2;
        hls::stream<bool> e_ostrm2;

        // const int NONBLOCK_DEPTH = 256;
        const int NONBLOCK_DEPTH = 8;

#pragma HLS RESOURCE variable = ostrm0 core = FIFO_LUTRAM
#pragma HLS STREAM variable = ostrm0 depth = NONBLOCK_DEPTH
#pragma HLS RESOURCE variable = e_ostrm0 core = FIFO_LUTRAM
#pragma HLS STREAM variable = e_ostrm0 depth = NONBLOCK_DEPTH
#pragma HLS RESOURCE variable = ostrm1 core = FIFO_LUTRAM
#pragma HLS STREAM variable = ostrm1 depth = NONBLOCK_DEPTH
#pragma HLS RESOURCE variable = e_ostrm1 core = FIFO_LUTRAM
#pragma HLS STREAM variable = e_ostrm1 depth = NONBLOCK_DEPTH
#pragma HLS RESOURCE variable = ostrm2 core = FIFO_LUTRAM
#pragma HLS STREAM variable = ostrm2 depth = NONBLOCK_DEPTH
#pragma HLS RESOURCE variable = e_ostrm2 core = FIFO_LUTRAM
#pragma HLS STREAM variable = e_ostrm2 depth = NONBLOCK_DEPTH

        xf::common::utils_hw::axiToMultiStream<BURST_LENTH, AXI_WIDTH, DT0, DT1, DT2>(
            rbuf, ostrm0, e_ostrm0, ostrm1, e_ostrm1, ostrm2, e_ostrm2, len, offset);

        // Stream Consumers
        forward_data<1>(ostrm0, e_ostrm0, r_strm0);
        forward_data<2>(ostrm1, e_ostrm1, r_strm1);
        forward_data<4>(ostrm2, e_ostrm2, r_strm2);
    }
}

#ifndef __SYNTHESIS__

#include <iostream>
#include <cstdlib>

int main() {
    int nerror = 0;
    int len[3] = {100 * DT0_SZ, 200 * DT2_SZ, 300 * DT2_SZ};
    int offset[3] = {0 + 1, DT0_SZ * ELEM_SPACE + 2, DT0_SZ * ELEM_SPACE + DT1_SZ * ELEM_SPACE + 3};

    // popolate the buffer
    char* buf = (char*)malloc(AXI_SZ * BUF_DEPTH);
    memset(buf, 0, AXI_SZ * BUF_DEPTH);

    DT0* const d0ptr = (DT0*)(buf + offset[0]);
    for (int i = 0; i < ELEM_NUM; ++i) {
        d0ptr[i] = i;
    }
    DT1* const d1ptr = (DT1*)(buf + offset[1]);
    for (int i = 0; i < ELEM_NUM; ++i) {
        d1ptr[i] = i * 2;
    }
    DT2* const d2ptr = (DT2*)(buf + offset[2]);
    for (int i = 0; i < ELEM_NUM; ++i) {
        d2ptr[i] = i * 3;
    }

    // conduct the test
    hls::stream<DT0> ds0;
    hls::stream<DT1> ds1;
    hls::stream<DT2> ds2;

    top_func((ap_uint<AXI_WIDTH>*)buf, ds0, ds1, ds2);

    // verify the result
    int j = 0;
    while (!ds0.empty()) {
        DT0 t = ds0.read();
        DT0 r = d0ptr[j];
        if (t != r) {
            ++nerror;
            if (j < 20) {
                std::cout << "ERROR: read: " << t << ", ref: " << r << std::endl;
            }
        }
        j++;
    }
    if (j != len[0] / DT0_SZ) {
        ++nerror;
        std::cout << "ERROR: stream 0 gets: " << j << ", requested: " << len[0] << std::endl;
    }

    j = 0;
    while (!ds1.empty()) {
        DT1 t = ds1.read();
        DT1 r = d1ptr[j];
        if (t != r) {
            ++nerror;
            if (j < 20) {
                std::cout << "ERROR: read: " << t << ", ref: " << r << std::endl;
            }
        }
        j++;
    }
    if (j != len[1] / DT1_SZ) {
        ++nerror;
        std::cout << "ERROR: stream 1 gets: " << j << ", requested: " << len[1] << std::endl;
    }

    j = 0;
    while (!ds2.empty()) {
        DT2 t = ds2.read();
        DT2 r = d2ptr[j];
        if (t != r) {
            ++nerror;
            if (j < 20) {
                std::cout << "ERROR: read: " << t << ", ref: " << r << std::endl;
            }
        }
        j++;
    }
    if (j != len[2] / DT2_SZ) {
        ++nerror;
        std::cout << "ERROR: stream 2 gets: " << j << ", requested: " << len[2] << std::endl;
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
