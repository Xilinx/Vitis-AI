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

#ifndef _XF_MEAN_STDDEV_HPP_
#define _XF_MEAN_STDDEV_HPP_

#ifndef __cplusplus
#error C++ is needed to include this header
#endif

#define POW32 2147483648

#include "hls_stream.h"
#include "common/xf_common.hpp"
#include "core/xf_math.h"

namespace xf {
namespace cv {

template <int TYPE, int ROWS, int COLS, int PLANES, int NPC>
void xFStddevkernel(xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _src_mat1,
                    unsigned short _mean[XF_CHANNELS(TYPE, NPC)],
                    unsigned short _dst_stddev[XF_CHANNELS(TYPE, NPC)],
                    uint16_t height,
                    uint16_t width) {
// clang-format off
    #pragma HLS inline
    // clang-format on
    ap_uint<4> j;
    ap_uint<45> tmp_var_vals[(1 << XF_BITSHIFT(NPC)) * PLANES]; //={0};
    ap_uint<64> var[PLANES];                                    //={0};
    uint32_t tmp_sum_vals[(1 << XF_BITSHIFT(NPC)) * PLANES];    //={0};
    uint64_t sum[PLANES];                                       //={0};

// ap_uint<8> val[(1<<XF_BITSHIFT(NPC))*PLANES];

// clang-format off
    #pragma HLS ARRAY_PARTITION variable=tmp_var_vals complete dim=0
    #pragma HLS ARRAY_PARTITION variable=tmp_sum_vals complete dim=0
    #pragma HLS ARRAY_PARTITION variable=sum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=var complete dim=0
    // clang-format on
    //#pragma HLS ARRAY_PARTITION variable=val complete dim=0

    for (j = 0; j < ((1 << XF_BITSHIFT(NPC)) * PLANES); j++) {
// clang-format off
        #pragma HLS UNROLL
        // clang-format on
        tmp_var_vals[j] = 0;
        tmp_sum_vals[j] = 0;
    }
    for (j = 0; j < PLANES; j++) {
        sum[j] = 0;
        var[j] = 0;
    }
    int p = 0, read_index = 0;
    ap_uint<13> row, col;
Row_Loop1:
    for (row = 0; row < height; row++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    // clang-format on

    Col_Loop1:
        for (col = 0; col < (width >> XF_BITSHIFT(NPC)); col++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=COLS/NPC max=COLS/NPC
            #pragma HLS pipeline II=1
            #pragma HLS LOOP_FLATTEN OFF
            // clang-format on

            XF_TNAME(TYPE, NPC) in_buf;
            in_buf = _src_mat1.read(read_index++);

        Extract1:
            for (int p = 0; p < XF_NPIXPERCYCLE(NPC) * PLANES; p++) {
// clang-format off
                #pragma HLS unroll
                #pragma HLS DEPENDENCE variable=tmp_var_vals intra false
                // clang-format on

                ap_uint<8> val = in_buf.range(p * 8 + 7, p * 8);
                tmp_sum_vals[p] = tmp_sum_vals[p] + val;
                unsigned short int temp = ((unsigned short)val * (unsigned short)val);

                tmp_var_vals[p] += temp;
            }
        }
    }

    for (int c = 0; c < PLANES; c++) {
        for (j = 0; j < (1 << XF_BITSHIFT(NPC)); j++) {
// clang-format off
            #pragma HLS UNROLL
            // clang-format on
            sum[c] = (sum[c] + tmp_sum_vals[j * PLANES + c]);
            var[c] = (ap_uint<64>)((ap_uint<64>)var[c] + (ap_uint<64>)tmp_var_vals[j * PLANES + c]);
        }
    }

    ap_uint<16 *PLANES> mean_acc = 0, stddev_acc = 0;

    for (int c = 0; c < PLANES; c++) {
// clang-format off
        #pragma HLS UNROLL
        // clang-format on
        unsigned int tempmean = 0;

        tempmean = (unsigned short)((ap_uint<64>)(256 * (ap_uint<64>)sum[c]) / (width * height));
        mean_acc.range(c * 16 + 15, c * 16) = tempmean;

        /* Variance Computation */

        uint32_t temp = (ap_uint<32>)((ap_uint<64>)(65536 * (ap_uint<64>)var[c]) / (width * height));

        uint32_t Varstddev = temp - (tempmean * tempmean);

        uint32_t t1 = (uint32_t)((Varstddev >> 16) << 16);

        stddev_acc.range(c * 16 + 15, c * 16) = (unsigned short)xf::cv::Sqrt(t1); // StdDev;//(StdDev >> 4);
    }

    for (int i = 0; i < PLANES; ++i) {
        _mean[i] = mean_acc.range(i * 16 + 15, i * 16);
        _dst_stddev[i] = stddev_acc.range(i * 16 + 15, i * 16);
    }
}

template <int SRC_T, int ROWS, int COLS, int NPC = 1>
void meanStdDev(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src,
                unsigned short _mean[XF_CHANNELS(SRC_T, NPC)],
                unsigned short _stddev[XF_CHANNELS(SRC_T, NPC)]) {
// clang-format off
    #pragma HLS inline off
// clang-format on
//#pragma HLS dataflow

#ifndef __SYNTHESIS__
    assert((SRC_T == XF_8UC1 || SRC_T == XF_8UC3 || SRC_T == XF_8UC4) &&
           "Input image type should be XF_8UC1, XF_8UC3 or XF_8UC4");

    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && " NPC must be XF_NPPC1, XF_NPPC8");

    assert(((_src.rows <= ROWS) && (_src.cols <= COLS)) && "ROWS and COLS should be greater than input image");
#endif
    xFStddevkernel<SRC_T, ROWS, COLS, XF_CHANNELS(SRC_T, NPC), NPC>(_src, _mean, _stddev, _src.rows, _src.cols);
}
} // namespace cv
} // namespace xf

#endif // _XF_MEAN_STDDEV_HPP_
