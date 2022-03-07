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

#ifndef _XF_REDUCE_HPP_
#define _XF_REDUCE_HPP_

#ifndef __cplusplus
#error C++ is needed to include this header
#endif

typedef unsigned short uint16_t;
typedef unsigned char uchar;

#include "hls_stream.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
namespace xf {
namespace cv {

template <int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int ONE_D_HEIGHT,
          int ONE_D_WIDTH,
          int DEPTH,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int COLS_TRIP,
          int REDUCE_OP>
void xFreduceKernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                    xf::cv::Mat<DST_T, ONE_D_HEIGHT, ONE_D_WIDTH, 1>& _dst_mat,
                    unsigned char dim,
                    unsigned short height,
                    unsigned short width) {
    XF_SNAME(WORDWIDTH_SRC) val_src;
    XF_SNAME(WORDWIDTH_DST) val_dst;
    unsigned long long int p = 0, q = 0;
    unsigned char max = 0;

    short int depth = XF_DTPIXELDEPTH(SRC_T, NPC) / XF_CHANNELS(SRC_T, NPC);

    XF_SNAME(WORDWIDTH_DST) internal_res;

    XF_SNAME(WORDWIDTH_DST) line_buf[(COLS >> XF_BITSHIFT(NPC))];
// clang-format off
    #pragma HLS RESOURCE variable=line_buf core=RAM_S2P_BRAM
    // clang-format on

    if (dim == 0) {
        for (int i = 0; i < (width >> XF_BITSHIFT(NPC)); i++) {
// clang-format off
            #pragma HLS pipeline
            // clang-format on
            line_buf[i] = _src_mat.read(i);
        }
    }

    ap_uint<13> i, j, k, planes;

    unsigned int var;
    if (dim == 0) {
        var = 1;
    } else {
        var = 0;
    }
rowLoop:
    for (i = var; i < height; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        #pragma HLS LOOP_FLATTEN off
        // clang-format on
        if (REDUCE_OP == REDUCE_MIN) {
            internal_res = -1;
            max = 255;
        } else {
            internal_res = 0;
            max = 0;
        }
    colLoop:
        for (j = 0; j < width; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=COLS_TRIP max=COLS_TRIP
            #pragma HLS pipeline
            // clang-format on

            val_src =
                (XF_SNAME(WORDWIDTH_SRC))(_src_mat.read(i * width + j)); // reading the source stream _src into val_src
            if (dim == 0) {
                internal_res = line_buf[j];
            }

            switch (REDUCE_OP) {
                case REDUCE_SUM:
                    internal_res = internal_res + val_src;
                    break;
                case REDUCE_AVG:
                    internal_res = internal_res + val_src;
                    break;
                case REDUCE_MAX:
                    internal_res =
                        ((XF_SNAME(WORDWIDTH_SRC))internal_res > val_src ? (XF_SNAME(WORDWIDTH_SRC))internal_res
                                                                         : val_src);
                    break;
                case REDUCE_MIN:
                    internal_res =
                        ((XF_SNAME(WORDWIDTH_SRC))internal_res < val_src ? (XF_SNAME(WORDWIDTH_SRC))internal_res
                                                                         : val_src);
                    break;
            }
            if (dim == 1 && j == width - 1) {
                if (REDUCE_OP == REDUCE_AVG) {
                    val_dst = internal_res / width;
                } else {
                    val_dst = internal_res;
                }
            }
            if (dim == 0) {
                val_dst = internal_res;
                line_buf[j] = val_dst;
            }
        }

        if (dim == 1) {
            _dst_mat.write(q, val_dst);
            q++;
        }
    }
    if (dim == 0) {
        for (unsigned int out = 0; out < ((width >> XF_BITSHIFT(NPC))); out++) {
            if ((REDUCE_OP == REDUCE_SUM)) {
                _dst_mat.write(q, line_buf[out]);
                q++;
            } else if (REDUCE_OP == REDUCE_AVG) {
                _dst_mat.write(q, line_buf[out] / height);
                q++;
            } else {
                _dst_mat.write(q, line_buf[out]);
                q = q + 1;
            }
        }
    }
}

template <int REDUCE_OP, int SRC_T, int DST_T, int ROWS, int COLS, int ONE_D_HEIGHT, int ONE_D_WIDTH, int NPC = 1>
void reduce(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
            xf::cv::Mat<DST_T, ONE_D_HEIGHT, ONE_D_WIDTH, 1>& _dst_mat,
            unsigned char dim) {
    unsigned short width = _src_mat.cols >> XF_BITSHIFT(NPC);
    unsigned short height = _src_mat.rows;

#ifndef __SYNTHESIS__
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && "NPC must be XF_NPPC1, XF_NPPC8");
    assert(((height <= ROWS) && (width <= COLS)) && "ROWS and COLS should be greater than input image");
#endif

// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    xFreduceKernel<SRC_T, DST_T, ROWS, COLS, ONE_D_HEIGHT, ONE_D_WIDTH, XF_DEPTH(SRC_T, NPC), NPC,
                   XF_WORDWIDTH(SRC_T, NPC), XF_WORDWIDTH(DST_T, NPC), (COLS >> XF_BITSHIFT(NPC)), REDUCE_OP>(
        _src_mat, _dst_mat, dim, height, width);
}
} // namespace cv
} // namespace xf

#endif
