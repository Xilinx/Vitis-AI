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

#ifndef __XF_CORNER_UPDATE_HPP__
#define __XF_CORNER_UPDATE_HPP__
#include "ap_int.h"
#include "hls_stream.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
namespace xf {
namespace cv {
template <unsigned int MAXCORNERSNO, unsigned int TYPE, unsigned int ROWS, unsigned int COLS, unsigned int NPC>
void cornerUpdate(unsigned long* list_fix,
                  unsigned int* list,
                  uint32_t nCorners,
                  xf::cv::Mat<TYPE, ROWS, COLS, NPC>& flow_vectors,
                  ap_uint<1> harris_flag) {
    unsigned int* flowvectorsDataPtr = (unsigned int*)flow_vectors.data;
    unsigned int list_flag_tmp;
    unsigned int row_num_fix = 0;
    unsigned int col_num_fix = 0;
    unsigned short row_num;
    unsigned short col_num;

    // reading the packed flow vectors
    unsigned int flowuv_tl;
    unsigned int flowuv_tr;
    unsigned int flowuv_bl;
    unsigned int flowuv_br;

    for (int li = 0; li < nCorners; li++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAXCORNERSNO
        // clang-format on
        for (int lj = 0; lj < 6; lj++) {
// clang-format off
            #pragma HLS PIPELINE II=1
            // clang-format on
            if (lj == 0) {
                unsigned int point = list[li];
                unsigned long list_flag_data = list_fix[li];
                if (harris_flag) {
                    // unsigned int point = list[li];
                    // unsigned long list_flag_data = list_fix[li];
                    list_flag_tmp = ((unsigned long)list_flag_data & 0xFFFFFC0000000000) >> 42; //.range(63,42);
                    // row_num_fix and col_num_fix in Q16.0 format, but the format is converted to Q16.5
                    if (list_flag_tmp == 0) {
                        row_num_fix = ((point >> 16) & (0x0000FFFF)) << 5; //((ap_uint<21>)point.range(31,16))<<5;
                        col_num_fix = ((point) & (0x0000FFFF)) << 5;       //((ap_uint<21>)point.range(15,0))<<5;
                        list_flag_tmp = ((unsigned long)list_flag_data & 0xFFFFFC0000000000) >> 42; //.range(63,42);
                    }
                } else {
                    // unsigned long list_flag_data = list_fix[li];
                    // row_num_fix and col_num_fix in Q16.5 format
                    list_flag_tmp = ((unsigned long)list_flag_data & 0xFFFFFC0000000000) >> 42; //.range(63,42);
                    if (list_flag_tmp == 0) {
                        row_num_fix = ((list_flag_data >> 21) & (0x001FFFFF)); //(ap_uint<21>)point.range(41,21);
                        col_num_fix = ((list_flag_data) & (0x001FFFFF));       //((a(ap_uint<21>)point.range(20,0);
                    }
                }
                row_num = (unsigned short)(row_num_fix >> 5);
                col_num = (unsigned short)(col_num_fix >> 5);
            } else if (lj == 1) {
                flowuv_tl = flowvectorsDataPtr[row_num * (flow_vectors.cols) + col_num];
            } else if (lj == 2) {
                flowuv_tr = flowvectorsDataPtr[row_num * (flow_vectors.cols) + (col_num + 1)];
            } else if (lj == 3) {
                flowuv_bl = flowvectorsDataPtr[(row_num + 1) * (flow_vectors.cols) + col_num];
            } else if (lj == 4) {
                flowuv_br = flowvectorsDataPtr[(row_num + 1) * (flow_vectors.cols) + (col_num + 1)];
            } else if (lj == 5) {
                unsigned int rl_fix = ((row_num_fix >> 5) << 5);
                unsigned int ct_fix = ((col_num_fix >> 5) << 5);
                unsigned int rr_fix = rl_fix + 32;
                unsigned int cb_fix = ct_fix + 32;

                // Q0.5*Q0.5 -> Q0.10 >> 5 -> Q0.5
                unsigned short tl_w = ((rr_fix - row_num_fix) * (cb_fix - col_num_fix)) >> 5;
                unsigned short tr_w = ((row_num_fix - rl_fix) * (cb_fix - col_num_fix)) >> 5;
                unsigned short bl_w = ((rr_fix - row_num_fix) * (col_num_fix - ct_fix)) >> 5;
                unsigned short br_w = ((row_num_fix - rl_fix) * (col_num_fix - ct_fix)) >> 5;

                // extracting the flow vectors, format A10.6
                short flow_utl = (flowuv_tl >> 16);
                short flow_vtl = (0x0000FFFF & flowuv_tl);
                short flow_utr = (flowuv_tr >> 16);
                short flow_vtr = (0x0000FFFF & flowuv_tr);
                short flow_ubl = (flowuv_bl >> 16);
                short flow_vbl = (0x0000FFFF & flowuv_bl);
                short flow_ubr = (flowuv_br >> 16);
                short flow_vbr = (0x0000FFFF & flowuv_br);

                short flow_u = ((tl_w * flow_utl) + (tr_w * flow_utr) + (bl_w * flow_ubl) + (br_w * flow_ubr)) >> 6;
                short flow_v = ((tl_w * flow_vtl) + (tr_w * flow_vtr) + (bl_w * flow_vbl) + (br_w * flow_vbr)) >> 6;

                // add the flow vector to the corner data, rx and ry in Q16.5 format // TODO: check the
                // overflow/underflow
                unsigned int rx = (unsigned int)flow_u + (unsigned int)col_num_fix;
                unsigned int ry = (unsigned int)flow_v + (unsigned int)row_num_fix;

                unsigned long outpoint_fix = 0;
                unsigned int outpoint = 0;
                if ((ry < (flow_vectors.rows << 5)) && (ry >= 0) && (rx >= 0) && (rx < (flow_vectors.cols << 5)) &&
                    (list_flag_tmp == 0)) {
                    outpoint_fix =
                        ((unsigned long)ry << 21 & 0x000003FFFFE00000) | ((unsigned long)rx & 0x00000000001FFFFF);
                    // outpoint_fix.range(20,0) = (ap_uint<21>)rx;
                    // outpoint_fix.range(41,21) = (ap_uint<21>)ry;
                    // outpoint_fix.range(20,0) = (ap_uint<21>)rx;

                    outpoint = ((unsigned int)((ry + 16) >> 5) << 16 & 0xFFFF0000) |
                               ((unsigned int)((rx + 16) >> 5) & 0x0000FFFF);
                    // outpoint.range(31,16) = (ry+16)>>5; // rounding-off by adding 0.5 and flooring
                    // outpoint.range(15,0) = (rx+16)>>5;
                } else {
                    outpoint_fix = (unsigned long)1 << 42 & 0xFFFFFC0000000000;
                }
                list[li] = outpoint;
                list_fix[li] = outpoint_fix;
            }
        }
    }
}
} // namespace cv
} // namespace xf
#endif
