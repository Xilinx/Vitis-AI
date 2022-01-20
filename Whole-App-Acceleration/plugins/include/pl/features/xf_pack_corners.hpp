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

#ifndef _XF_PACK_CORNERS_HPP_
#define _XF_PACK_CORNERS_HPP_

#ifndef __cplusplus
#error C++ is needed to include this header
#endif
#include <stdio.h>

template <int NPC, int WORDWIDTH_DST, int MAXSIZE, int TC>
void auFillList(ap_uint<32> listcorners[MAXSIZE],
                XF_SNAME(WORDWIDTH_DST) tmp_cor_bufs[][MAXSIZE >> XF_BITSHIFT(NPC)],
                uint16_t* cor_cnts,
                uint32_t* nCorners) {
    int sz = 0;
    ap_uint<9> i;
    uint32_t EOL = 0;
    for (i = 0; i < (1 << XF_BITSHIFT(NPC)); i++) {
// clang-format off
        #pragma HLS unroll
        // clang-format on

        for (int crn_cnt = 0; crn_cnt < cor_cnts[i]; crn_cnt++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            // clang-format on
            XF_SNAME(WORDWIDTH_DST) val = tmp_cor_bufs[i][crn_cnt];
            listcorners[sz + crn_cnt] = val;
        }
        sz += cor_cnts[i];
    }
    // listcorners[sz]=EOL;
    *nCorners = sz;
}

template <int NPC, int WORDWIDTH_DST, int MAXSIZE, int TC>
void auFillList_points(ap_uint32_t* listcorners,
                       ap_uint<32> tmp_cor_bufs[][MAXSIZE >> XF_BITSHIFT(NPC)],
                       int* cor_cnts) {
    int sz = 0;
    for (int i = 0; i < (1 << XF_BITSHIFT(NPC)); i++) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        for (int crn_cnt = 0; crn_cnt < cor_cnts[i]; crn_cnt++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            // clang-format on
            listcorners[i * sz + crn_cnt] = (ap_uint32_t)(tmp_cor_bufs[i][crn_cnt]);
        }
        sz += cor_cnts[i];
    }
}

template <int NPC, int IN_WW, int MAXSIZE, int OUT_WW>
void auCheckForCorners(XF_SNAME(IN_WW) val,
                       uint16_t* corners_cnt,
                       XF_SNAME(OUT_WW) out_keypoints[][MAXSIZE >> XF_BITSHIFT(NPC)],
                       ap_uint<13> row,
                       ap_uint<13> col) {
    XF_SNAME(OUT_WW) tmp_store[(1 << XF_BITSHIFT(NPC))];

    for (ap_uint<9> i = 0; i < (1 << XF_BITSHIFT(NPC)); i++) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        ap_uint<9> shift = i << 3;
        ap_uint<8> v = val >> shift;
        uint16_t cnt = corners_cnt[i];

        if ((cnt < (MAXSIZE >> XF_BITSHIFT(NPC))) && (v != 0)) {
            XF_SNAME(OUT_WW) tmp = 0;

            tmp.range(15, 0) = col + i;
            tmp.range(31, 16) = row;
            out_keypoints[i][cnt] = tmp;
            cnt++;
        } else {
            tmp_store[i] = 0;
        }
        corners_cnt[i] = cnt;
    }
}

template <int ROWS, int COLS, int IN_DEPTH, int NPC, int IN_WW, int MAXPNTS, int OUT_WW, int SRC_TC>
void xFWriteCornersToList(hls::stream<XF_SNAME(IN_WW)>& _max_sup,
                          ap_uint<32> list[MAXPNTS],
                          uint32_t* nCorners,
                          uint16_t img_height,
                          uint16_t img_width) {
    img_width = img_width >> XF_BITSHIFT(NPC);
    if (NPC == XF_NPPC1) {
        int cnt = 0;
        uint32_t EOL = 0;
        ap_uint<13> row, col;
        for (row = 0; row < img_height; row++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=ROWS max = ROWS
            #pragma HLS LOOP_FLATTEN off
            // clang-format on
            for (col = 0; col < img_width; col++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=SRC_TC max = SRC_TC
                #pragma HLS pipeline
                // clang-format on
                XF_SNAME(IN_WW) val = _max_sup.read();

                if ((cnt < (MAXPNTS)) && (val != 0)) {
                    XF_SNAME(OUT_WW) tmp = 0;

                    tmp.range(15, 0) = col;
                    tmp.range(31, 16) = row;
                    list[cnt] = tmp; //.write(tmp);
                    cnt++;
                }
            }
        }
        // list[cnt]=EOL;
        *nCorners = cnt;
    } else {
        XF_SNAME(OUT_WW) tmp_corbufs[(1 << XF_BITSHIFT(NPC))][(MAXPNTS >> XF_BITSHIFT(NPC))];
        uint16_t corners_cnt[(1 << XF_BITSHIFT(NPC))];
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=corners_cnt complete dim=0
        #pragma HLS ARRAY_PARTITION variable=tmp_corbufs complete dim=1
        // clang-format on

        for (ap_uint<9> i = 0; i < (1 << XF_BITSHIFT(NPC)); i++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            corners_cnt[i] = 0;
        }
        ap_uint<13> row, col;
        for (row = 0; row < img_height; row++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=ROWS max = ROWS
            #pragma HLS LOOP_FLATTEN off
            // clang-format on
            for (col = 0; col < img_width; col++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=SRC_TC max = SRC_TC
                #pragma HLS pipeline
                // clang-format on
                XF_SNAME(IN_WW) val = _max_sup.read();
                auCheckForCorners<NPC, IN_WW, MAXPNTS, OUT_WW>(val, corners_cnt, tmp_corbufs, row,
                                                               (col << XF_BITSHIFT(NPC)));
            }
        }

        auFillList<NPC, OUT_WW, MAXPNTS, (MAXPNTS >> XF_BITSHIFT(NPC))>(list, tmp_corbufs, corners_cnt, nCorners);
    }
}

#endif // _XF_MAX_SUPPRESSION_H_
