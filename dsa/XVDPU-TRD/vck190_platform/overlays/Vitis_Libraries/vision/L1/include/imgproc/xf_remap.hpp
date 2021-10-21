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

#ifndef _XF_REMAP_HPP_
#define _XF_REMAP_HPP_

#ifndef __cplusplus
#error C++ is needed to include this header.
#endif

#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "hls_stream.h"
#include <algorithm>

#define XF_RESIZE_INTER_TAB_SIZE 32
#define XF_RESIZE_INTER_BITS 5

namespace xf {
namespace cv {

template <int SRC_T, int DST_T, int PLANES, int MAP_T, int WIN_ROW, int ROWS, int COLS, int NPC, bool USE_URAM>
void xFRemapNNI(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src,
                xf::cv::Mat<DST_T, ROWS, COLS, NPC>& dst,
                xf::cv::Mat<MAP_T, ROWS, COLS, NPC>& mapx,
                xf::cv::Mat<MAP_T, ROWS, COLS, NPC>& mapy,
                uint16_t rows,
                uint16_t cols) {
    XF_TNAME(DST_T, NPC) buf[WIN_ROW][COLS];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=buf complete dim=1
    // clang-format on

    XF_TNAME(SRC_T, NPC) s;
    int read_pointer_src = 0, read_pointer_map = 0, write_pointer = 0;

    ap_uint<64> bufUram[PLANES][WIN_ROW][(COLS + 7) / 8];
// clang-format off
    #pragma HLS resource variable=bufUram core=RAM_T2P_URAM latency=2
// clang-format on
//#pragma HLS dependence variable=bufUram inter false
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=bufUram complete dim=2
    #pragma HLS ARRAY_PARTITION variable=bufUram complete dim=1
    // clang-format on

    XF_TNAME(SRC_T, NPC) sx8[8];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=sx8 complete dim=1
    // clang-format on

    XF_TNAME(DST_T, NPC) d;

#ifndef __SYNTHESIS__
    assert(rows <= ROWS);
    assert(cols <= COLS);
#endif

    int ishift = WIN_ROW / 2;
    int r[WIN_ROW] = {};
    const int row_tripcount = ROWS + WIN_ROW;

loop_height:
    for (int i = 0; i < rows + ishift; i++) {
// clang-format off
        #pragma HLS LOOP_FLATTEN OFF
        #pragma HLS LOOP_TRIPCOUNT min=1 max=row_tripcount
    // clang-format on

    loop_width:
        for (int j = 0; j < cols; j++) {
// clang-format off
            #pragma HLS PIPELINE II=1
            #pragma HLS dependence variable=buf     inter false
            #pragma HLS dependence variable=bufUram inter false
            #pragma HLS dependence variable=r       inter false
            #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS
            // clang-format on

            if (i < rows && j < cols) {
                s = src.read(read_pointer_src++);

                if (USE_URAM) {
                    sx8[j % 8] = s;
                    for (int pl = 0, bit = 0; pl < PLANES; pl++, bit += 8) {
// clang-format off
                        #pragma HLS UNROLL
                        // clang-format on
                        for (int k = 0; k < 8; k++) {
// clang-format off
                            #pragma HLS UNROLL
                            // clang-format on
                            bufUram[pl][i % WIN_ROW][j / 8](k * 8 + 7, k * 8) = sx8[k](bit + 7, bit);
                        }
                    }
                }
            }

            if (!USE_URAM) buf[i % WIN_ROW][j] = s;
            r[i % WIN_ROW] = i;

            if (i >= ishift) {
                float mx_fl = mapx.read_float(read_pointer_map);
                float my_fl = mapy.read_float(read_pointer_map++);

                int x = (int)(mx_fl + 0.5f);
                int y = (int)(my_fl + 0.5f);

                bool in_range = (y >= 0 && my_fl <= (rows - 1) && r[y % WIN_ROW] == y && x >= 0 && mx_fl <= (cols - 1));
                if (in_range)
                    if (USE_URAM) {
                        XF_TNAME(DST_T, NPC) dx9[8];
// clang-format off
                        #pragma HLS ARRAY_PARTITION variable=dx9 complete dim=1
                        // clang-format on
                        for (int pl = 0, bit = 0; pl < PLANES; pl++, bit += 8) {
                            ap_uint<72> tempvalue[PLANES]; //
                            tempvalue[pl] = bufUram[pl][y % WIN_ROW][x / 8];
                            for (int k = 0; k < 8; k++) {
                                dx9[k](bit + 7, bit) = tempvalue[pl].range(k * 8 + 7, k * 8);
                            }
                        }
                        d = dx9[x % 8];
                    } else
                        d = buf[y % WIN_ROW][x];
                else
                    d = 0;

                dst.write(write_pointer++, d);
            }
        }
    }
}

#define TWO_POW_16 65536
template <int SRC_T, int DST_T, int PLANES, int MAP_T, int WIN_ROW, int ROWS, int COLS, int NPC, bool USE_URAM>
void xFRemapLI(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src,
               xf::cv::Mat<DST_T, ROWS, COLS, NPC>& dst,
               xf::cv::Mat<MAP_T, ROWS, COLS, NPC>& mapx,
               xf::cv::Mat<MAP_T, ROWS, COLS, NPC>& mapy,
               uint16_t rows,
               uint16_t cols) {
    // Add one to always get zero for boundary interpolation. Maybe need
    // initialization here?
    XF_TNAME(DST_T, NPC)
    buf[WIN_ROW / 2 + 1][2][COLS / 2 + 1][2]; // AK,ZoTech: static added for
                                              // initialization, otherwise X are
                                              // generated in co-sim.
                                              // clang-format off
    #pragma HLS array_partition complete variable=buf dim=2
    #pragma HLS array_partition complete variable=buf dim=4
                                              // clang-format on
    XF_TNAME(SRC_T, NPC) s;

    // URAM storage garnularity is 3x3-pel block in 2x2-pixel picture grid, it
    // fits to one URAM word
    ap_uint<72> bufUram[PLANES][(WIN_ROW + 1) / 2][(COLS + 1) / 2];
// clang-format off
    #pragma HLS resource variable=bufUram core=RAM_T2P_URAM latency=2
    #pragma HLS array_partition complete variable=bufUram dim=1
    // clang-format on

    ap_uint<24> lineBuf[PLANES][(COLS + 1) / 2];
// clang-format off
    #pragma HLS resource variable=lineBuf core=RAM_S2P_BRAM latency=1
    #pragma HLS array_partition complete variable=lineBuf dim=1
    // clang-format on

    XF_TNAME(MAP_T, NPC) mx;
    XF_TNAME(MAP_T, NPC) my;

    int read_pointer_src = 0, read_pointer_map = 0, write_pointer = 0;

#ifndef __SYNTHESIS__
    assert(rows <= ROWS);
    assert(cols <= COLS);
#endif

    int ishift = WIN_ROW / 2;
    int r1[WIN_ROW] = {};
    int r2[WIN_ROW] = {};
    const int row_tripcount = ROWS + WIN_ROW;

    bool store_col = 1;
    bool store_row = 1;

    ap_uint<16> temppix[PLANES];     //     = 0;
    ap_uint<24> pixval[PLANES];      //      = 0;
    ap_uint<48> pixval_2[PLANES];    //    = 0;
    ap_uint<24> prev_pixval[PLANES]; // = 0;
    ap_uint<72> tempbuf[PLANES];

    for (int pl = 0; pl < PLANES; pl++) {
// clang-format off
        #pragma HLS UNROLL
        // clang-format on
        temppix[pl] = 0;
        pixval[pl] = 0;
        pixval_2[pl] = 0;
        prev_pixval[pl] = 0;
        tempbuf[pl] = 0;
    }

loop_height:
    for (int i = 0; i < rows + ishift; i++) {
// clang-format off
        #pragma HLS LOOP_FLATTEN OFF
        #pragma HLS LOOP_TRIPCOUNT min=1 max=row_tripcount
        // clang-format on

        // Initialize for every row
        store_col = 1;

    loop_width:
        for (int j = 0; j < cols + 1; j++) {
// clang-format off
            #pragma HLS PIPELINE II=1
            #pragma HLS dependence variable=buf     inter false
            #pragma HLS dependence variable=bufUram inter false
            #pragma HLS dependence variable=bufUram intra false
            #pragma HLS dependence variable=r1      inter false
            #pragma HLS dependence variable=r2      inter false
            #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS+2
            // clang-format on

            if (i < rows && j < cols) {
                s = src.read(read_pointer_src++);
            } else {
                s = 0;
            }

            if (USE_URAM && i < rows) {
                for (int pl = 0, bit = 0; pl < PLANES; pl++, bit += 8) {
// clang-format off
                    #pragma HLS UNROLL
                    // clang-format on
                    if (store_col && (j != 0)) {
                        pixval[pl].range(15, 0) = temppix[pl];
                        pixval[pl].range(23, 16) = s.range(bit + 7, bit);

                        if (store_row) {
                            // Store every 3rd row in a buffer
                            lineBuf[pl][(j / 2) - 1] = pixval[pl];
                        } else {
                            // Read the stored row and fill in
                            prev_pixval[pl] = lineBuf[pl][(j / 2) - 1];
                        }

                        if (i != 0) {
                            if (store_row) {
                                bufUram[pl][((i - 1) / 2) % (WIN_ROW / 2)][(j / 2) - 1].range(71, 48) = pixval[pl];
                            } else {
                                pixval_2[pl].range(23, 0) = prev_pixval[pl];
                                pixval_2[pl].range(47, 24) = pixval[pl];
                                bufUram[pl][((i - 1) / 2) % (WIN_ROW / 2)][(j / 2) - 1].range(47, 0) = pixval_2[pl];
                            }
                        }
                    }

                    if (store_col) {
                        temppix[pl].range(7, 0) = s.range(bit + 7, bit);
                    } else {
                        temppix[pl].range(15, 8) = s.range(bit + 7, bit);
                    }
                }

                store_col = !(store_col);
            }

            if (!USE_URAM) {
                if ((i % WIN_ROW) % 2) {
                    buf[(i % WIN_ROW) / 2][(i % WIN_ROW) % 2][j / 2][j % 2] = s; //.range(bit+7,bit);
                } else {
                    buf[(i % WIN_ROW) / 2][(i % WIN_ROW) % 2][j / 2][j % 2] = s; //.range(bit+7,bit);
                }
            }

            r1[i % WIN_ROW] = i;
            r2[i % WIN_ROW] = i;

            if (i >= ishift && j < cols) {
                float x_fl = mapx.read_float(read_pointer_map);
                float y_fl = mapy.read_float(read_pointer_map++);

                int x_fix = (int)((float)x_fl * (float)XF_RESIZE_INTER_TAB_SIZE); // mapx data in
                                                                                  // A16.XF_RESIZE_INTER_TAB_SIZE
                                                                                  // format
                int y_fix = (int)((float)y_fl * (float)XF_RESIZE_INTER_TAB_SIZE); // mapy data in
                                                                                  // A16.XF_RESIZE_INTER_TAB_SIZE
                                                                                  // format

                int x = x_fix >> XF_RESIZE_INTER_BITS;
                int y = y_fix >> XF_RESIZE_INTER_BITS;
                int x_frac = x_fix & (XF_RESIZE_INTER_TAB_SIZE - 1);
                int y_frac = y_fix & (XF_RESIZE_INTER_TAB_SIZE - 1);
                int ynext = y + 1;

                ap_ufixed<XF_RESIZE_INTER_BITS, 0> iu, iv;
                iu(XF_RESIZE_INTER_BITS - 1, 0) = x_frac;
                iv(XF_RESIZE_INTER_BITS - 1, 0) = y_frac;

                // Note that the range here is larger than expected by 1 horizontal and
                // 1 vertical pixel, to allow
                // Interpolating at the edge of the image
                bool in_range = (y >= 0 && y_fl <= (rows - 1) && r1[y % WIN_ROW] == y && r2[ynext % WIN_ROW] == ynext &&
                                 x >= 0 && x_fl <= (cols - 1));

                int xa0, xa1, ya0, ya1;
// The buffer is essentially cyclic partitioned, but we have
// to do this manually because HLS can't figure it out.
// The code below is wierd, but it is this code expanded.
//  if ((y % WIN_ROW) % 2) {
//                     // Case 1, where y hits in bank 1 and ynext in bank 0
//                     ya0 = (ynext%WIN_ROW)/2;
//                     ya1 = (y%WIN_ROW)/2;
//                 } else {
//                     // The simpler case, where y hits in bank 0 and ynext
//                     hits in bank 1
//                     ya0 = (y%WIN_ROW)/2;
//                     ya1 = (ynext%WIN_ROW)/2;
//                 }
// Both cases reduce to this, if WIN_ROW is a multiple of two.
#ifndef __SYNTHESIS__
                assert(((WIN_ROW & 1) == 0) && "WIN_ROW must be a multiple of two");
#endif
                xa0 = x / 2 + x % 2;
                xa1 = x / 2;
                ya0 = (y / 2 + y % 2) % (WIN_ROW / 2);
                ya1 = (y / 2) % (WIN_ROW / 2);

                XF_TNAME(DST_T, NPC) d;

                for (int ch = 0; ch < PLANES; ch++) {
                    XF_CTUNAME(DST_T, NPC) d00, d01, d10, d11;

                    if (in_range) {
                        if (USE_URAM) {
                            XF_TNAME(DST_T, NPC) d3x3[9];
// clang-format off
                           #pragma HLS ARRAY_PARTITION variable=d3x3 complete
                            // clang-format on

                            tempbuf[ch] = bufUram[ch][ya1][xa1];

                            for (int k = 0; k < 9; k++) {
                                d3x3[k] = tempbuf[ch].range(k * 8 + 7, k * 8);
                            }

                            d00 = d3x3[(y % 2) * 3 + x % 2];
                            d01 = d3x3[(y % 2) * 3 + x % 2 + 1];
                            d10 = d3x3[(y % 2 + 1) * 3 + x % 2];
                            d11 = d3x3[(y % 2 + 1) * 3 + x % 2 + 1];
                        } else {
                            d00 = buf[ya0][0][xa0][0].range((ch + 1) * 8 - 1, ch * 8);
                            d01 = buf[ya0][0][xa1][1].range((ch + 1) * 8 - 1, ch * 8);
                            d10 = buf[ya1][1][xa0][0].range((ch + 1) * 8 - 1, ch * 8);
                            d11 = buf[ya1][1][xa1][1].range((ch + 1) * 8 - 1, ch * 8);

                            if (x % 2) {
                                // std::swap(d00,d01);
                                int t = d00;
                                d00 = d01;
                                d01 = t;

                                int t2 = d10;
                                d10 = d11;
                                d11 = d10;
                                // std::swap(d10,d11);
                            }
                            if (y % 2) {
                                int t = d00;
                                d00 = d10;
                                d10 = t;

                                int t2 = d01;
                                d01 = d11;
                                d11 = d01;
                                // std::swap(d00,d10);
                                // std::swap(d01,d11);
                            }
                            // if(x == (cols-1))
                            //{
                            //	d01=0;d11=0;
                            //}
                        }
                    }
                    ap_ufixed<2 * XF_RESIZE_INTER_BITS + 1, 1> k01 = (1 - iv) * (iu); // iu-iu*iv
                    ap_ufixed<2 * XF_RESIZE_INTER_BITS + 1, 1> k10 = (iv) * (1 - iu); // iv-iu*iv
                    ap_ufixed<2 * XF_RESIZE_INTER_BITS + 1, 1> k11 = (iv) * (iu);     // iu*iv
                    ap_ufixed<2 * XF_RESIZE_INTER_BITS + 1, 1> k00 =
                        1 - iv - k01; //(1-iv)*(1-iu) = 1-iu-iv+iu*iv = 1-iv-k01
#ifndef __SYNTHESIS__
                    assert(k00 + k01 + k10 + k11 == 1);
#endif

                    if (in_range)
                        d.range((ch + 1) * 8 - 1, ch * 8) = d00 * k00 + d01 * k01 + d10 * k10 + d11 * k11;
                    else
                        d.range((ch + 1) * 8 - 1, ch * 8) = 0;
                }
                dst.write(write_pointer++, d);
            }
        }

        store_row = !(store_row);
    }
}

template <int WIN_ROWS,
          int INTERPOLATION_TYPE,
          int SRC_T,
          int MAP_T,
          int DST_T,
          int ROWS,
          int COLS,
          int NPC,
          bool USE_URAM = false>
void remap(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
           xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _remapped_mat,
           xf::cv::Mat<MAP_T, ROWS, COLS, NPC>& _mapx_mat,
           xf::cv::Mat<MAP_T, ROWS, COLS, NPC>& _mapy_mat) {
// clang-format off
    #pragma HLS inline off
    #pragma HLS dataflow
// clang-format on

#ifndef __SYNTHESIS__
    assert((MAP_T == XF_32FC1) && "The MAP_T must be XF_32FC1");
    assert(((SRC_T == XF_8UC1) || (SRC_T == XF_8UC3)) && "The SRC_T must be XF_8UC1 or XF_8UC3");
    assert(((DST_T == XF_8UC1) || (SRC_T == XF_8UC3)) && "The DST_T must be XF_8UC1 or XF_8UC3");
    assert((SRC_T == DST_T) && "Source Mat type and Destination Mat type must be the same");
    assert((NPC == XF_NPPC1) && "The NPC must be XF_NPPC1");
#endif

    int depth_est = WIN_ROWS * _src_mat.cols;

    uint16_t rows = _src_mat.rows;
    uint16_t cols = _src_mat.cols;

    if (INTERPOLATION_TYPE == XF_INTERPOLATION_NN) {
        xFRemapNNI<SRC_T, DST_T, XF_CHANNELS(SRC_T, NPC), MAP_T, WIN_ROWS, ROWS, COLS, NPC, USE_URAM>(
            _src_mat, _remapped_mat, _mapx_mat, _mapy_mat, rows, cols);
    } else if (INTERPOLATION_TYPE == XF_INTERPOLATION_BILINEAR) {
        xFRemapLI<SRC_T, DST_T, XF_CHANNELS(SRC_T, NPC), MAP_T, WIN_ROWS, ROWS, COLS, NPC, USE_URAM>(
            _src_mat, _remapped_mat, _mapx_mat, _mapy_mat, rows, cols);
    } else {
#ifndef __SYNTHESIS__
        assert(((INTERPOLATION_TYPE == XF_INTERPOLATION_NN) || (INTERPOLATION_TYPE == XF_INTERPOLATION_BILINEAR)) &&
               "The INTERPOLATION_TYPE must be either XF_INTERPOLATION_NN or "
               "XF_INTERPOLATION_BILINEAR");
#endif
    }
}
} // namespace cv
} // namespace xf

#endif //_XF_REMAP_HPP_
