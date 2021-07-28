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

#ifndef _XF_PRE_PROCESS_
#define _XF_PRE_PROCESS_

#include "hls_stream.h"
#include "ap_int.h"

//#include "common/xf_common.hpp"

namespace xf {
namespace cv {

enum ops { mean_sub, scale_n_clip, clip, scale_n_bias, scale_n_bias_mean_sub, fused_op };

template <int INPUT_PTR_WIDTH_T,
          int OUTPUT_PTR_WIDTH_T,
          int T_CHANNELS_T,
          int CPW_T,
          int ROWS_T,
          int COLS_T,
          int NPC_T,
          bool PACK_MODE_T,
          int WX_T,
          int WA_T,
          int WB_T,
          int WY_T,
          int WO_T,
          int FX_T,
          int FA_T,
          int FB_T,
          int FY_T,
          int FO_T,
          bool SIGNED_IN_T,
          int OPMODE_T>
void xFpreProcessKernel(hls::stream<ap_uint<INPUT_PTR_WIDTH_T> >& srcStrm,
                        hls::stream<ap_uint<OUTPUT_PTR_WIDTH_T> >& dstStrm,
                        ap_fixed<WA_T, FA_T, AP_RND> alpha_reg[T_CHANNELS_T],
                        ap_fixed<WB_T, FB_T, AP_RND> beta_reg[T_CHANNELS_T],
                        ap_fixed<WY_T, FY_T, AP_RND> gamma_reg[T_CHANNELS_T],
                        int th1,
                        int th2,
                        int loop_count) {
#if SIGNED_IN_T
    typedef ap_fixed<WX_T, FX_T, AP_RND> X_TYPE;
#else
    typedef ap_ufixed<WX_T, FX_T, AP_RND> X_TYPE;
#endif

    for (int k = 0; k < loop_count; k++) {
// clang-format off
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=608*608
        // clang-format on
        ap_uint<INPUT_PTR_WIDTH_T> x_pack = srcStrm.read();
        ap_uint<OUTPUT_PTR_WIDTH_T> out_pack;

        for (int i = 0; i < NPC_T; i++) {
// clang-format off
#pragma HLS UNROLL
            // clang-format on
            for (int j = 0; j < CPW_T; j++) {
// clang-format off
#pragma HLS UNROLL
                // clang-format on
                X_TYPE x = x_pack.range((j * WX_T) + (WX_T - 1) + (i * CPW_T * WX_T), (j * WX_T) + (i * CPW_T * WX_T));

                ap_fixed<WA_T, FA_T, AP_RND> a = alpha_reg[j];
                ap_fixed<WB_T, FB_T, AP_RND> b = beta_reg[j];
                ap_fixed<WY_T, FA_T + FB_T, AP_RND> y = gamma_reg[j];
                ap_fixed<WO_T, FO_T, AP_RND> out;

                switch (OPMODE_T) {
                    case mean_sub: {
                        out = x - a;
                    } break;

                    case scale_n_clip: {
                        ap_fixed<WX_T + WB_T, FX_T + FB_T, AP_RND> prod1 = x * b;

                        if (prod1 >= th1)
                            out = th1;
                        else if (prod1 <= -th2)
                            out = -th2;
                        else
                            out = prod1;
                    } break;

                    case clip: {
                        if (x >= th1)
                            out = th1;
                        else if (x <= -th2)
                            out = -th2;
                        else
                            out = x;
                    }

                    break;

                    case scale_n_bias: {
                        ap_fixed<WX_T + WB_T, FX_T + FB_T, AP_RND> prod2 = x * b;

                        out = prod2 + y;
                    } break;

                    case scale_n_bias_mean_sub: {
                        ap_fixed<WX_T + WB_T, FX_T + FB_T, AP_RND> prod3 = (x - a) * b;

                        out = prod3 + y;
                    } break;

                    case fused_op: {
                        ap_fixed<WX_T + WB_T, FX_T + FB_T, AP_RND> prod4 = (x - a) * b + y;

                        if (prod4 >= th1)
                            out = th1;
                        else if (prod4 <= -th2)
                            out = -th2;
                        else
                            out = prod4;
                    } break;
                }

                ap_uint<WO_T>* out_val;

                out_val = (ap_uint<WO_T>*)&out;

                out_pack.range((j * WO_T) + (WO_T - 1) + (i * CPW_T * WO_T), (j * WO_T) + (i * CPW_T * WO_T)) =
                    *out_val;
            }
        }
        dstStrm.write(out_pack);
    }
}

template <int INPUT_PTR_WIDTH_T, int T_CHANNELS_T, int CPW_T, int NPC_T, int WX_T>
void Arr2Strm(ap_uint<INPUT_PTR_WIDTH_T>* inp, hls::stream<ap_uint<INPUT_PTR_WIDTH_T> >& Strm, int rows, int cols) {
    int loop_count = (rows * cols * WX_T * CPW_T * (T_CHANNELS_T / CPW)) / (INPUT_PTR_WIDTH_T);
    for (int i = 0; i < loop_count; i++) {
// clang-format off
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=608*608
        // clang-format on
        Strm.write(inp[i]);
    }
}

template <int OUTPUT_PTR_WIDTH_T, int T_CHANNELS_T, int CPW_T, int NPC_T, int WX_T>
void Strm2Arr(hls::stream<ap_uint<OUTPUT_PTR_WIDTH_T> >& Strm, ap_uint<OUTPUT_PTR_WIDTH_T>* out, int rows, int cols) {
    int loop_count = (rows * cols * WX_T * CPW_T * (T_CHANNELS_T / CPW_T)) / (OUTPUT_PTR_WIDTH_T);
    for (int i = 0; i < loop_count; i++) {
// clang-format off
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=608*608
        // clang-format on
        out[i] = Strm.read();
    }
}

template <int CPW_T, int NPC_T, int WX_T, int ptr_width, int strm_width>
void InBitWidthConvert(hls::stream<ap_uint<ptr_width> >& srcStrm,
                       hls::stream<ap_uint<strm_width> >& dstStrm,
                       int loop_count) {
    // int loop_count = (rows*cols)/(NPC);

    int valid_bits = 0;
    const int N_size = WX_T * CPW_T * NPC_T;
    ap_uint<ptr_width> r;
    ap_uint<strm_width> out;

L1:
    for (int i = 0; i < loop_count; i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=608*608
#pragma HLS PIPELINE II=1
        // clang-format on
        if (valid_bits < N_size) {
            if (valid_bits != 0) {
                out.range(valid_bits - 1, 0) = r.range(ptr_width - 1, ptr_width - valid_bits);
            }
            r = srcStrm.read();
            out.range(N_size - 1, valid_bits) = r.range(N_size - valid_bits - 1, 0);
            valid_bits = ptr_width - (N_size - valid_bits);
        } else {
            out = r.range(ptr_width - valid_bits + N_size - 1, ptr_width - valid_bits);
            valid_bits -= N_size;
        }
        dstStrm.write(out);
    }
}

template <int CPW_T, int NPC_T, int WX_T, int ptr_width, int strm_width>
void OutBitWidthConvert(hls::stream<ap_uint<strm_width> >& srcStrm,
                        hls::stream<ap_uint<ptr_width> >& dstStrm,
                        int loop_count) {
    // int loop_count = (rows*cols)/(NPC_T);

    int bits_to_add = ptr_width;
    const int N_size = WX_T * CPW_T * NPC_T;
    ap_uint<ptr_width> r;
    ap_uint<strm_width> in;

L1:
    for (int i = 0; i < loop_count; i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=608*608
#pragma HLS PIPELINE II=1
        // clang-format on
        in = srcStrm.read();

        if (bits_to_add <= N_size) {
            r.range(ptr_width - 1, ptr_width - bits_to_add) = in.range(bits_to_add - 1, 0);
            dstStrm.write(r);

            if (bits_to_add != N_size) {
                r.range(N_size - bits_to_add - 1, 0) = in.range(N_size - 1, bits_to_add);
            }
            bits_to_add = ptr_width - (N_size - bits_to_add);
        } else {
            r.range(ptr_width - bits_to_add + N_size - 1, ptr_width - bits_to_add) = in;
            bits_to_add -= N_size;
        }
    }

    if (bits_to_add != ptr_width) {
        dstStrm.write(r);
    }
}

template <int INPUT_PTR_WIDTH_T,
          int OUTPUT_PTR_WIDTH_T,
          int T_CHANNELS_T,
          int CPW_T,
          int ROWS_T,
          int COLS_T,
          int NPC_T,
          bool PACK_MODE_T,
          int WX_T,
          int WA_T,
          int WB_T,
          int WY_T,
          int WO_T,
          int FX_T,
          int FA_T,
          int FB_T,
          int FY_T,
          int FO_T,
          bool SIGNED_IN_T,
          int OPMODE_T>
void preProcess(hls::stream<ap_uint<INPUT_PTR_WIDTH_T> >& srcStrm,
                ap_uint<OUTPUT_PTR_WIDTH_T>* out,
                float params[3 * T_CHANNELS_T],
                int rows,
                int cols,
                int th1,
                int th2) {
#pragma HLS INLINE OFF

    // hls::stream<ap_uint<INPUT_PTR_WIDTH_T> > srcStrm;
    hls::stream<ap_uint<OUTPUT_PTR_WIDTH_T> > dstStrm;

    const int strm_width_in = CPW_T * NPC_T * WX_T;
    const int strm_width_out = CPW_T * NPC_T * WO_T;

    hls::stream<ap_uint<strm_width_in> > srcStrmIn;
    hls::stream<ap_uint<strm_width_out> > dstStrmOut;

    ap_fixed<WA_T, FA_T, AP_RND> alpha_reg[T_CHANNELS_T];
    ap_fixed<WB_T, FB_T, AP_RND> beta_reg[T_CHANNELS_T];
    ap_fixed<WY_T, FY_T, AP_RND> gamma_reg[T_CHANNELS_T];
// clang-format off
#pragma HLS ARRAY_PARTITION variable=alpha_reg dim=0 complete
#pragma HLS ARRAY_PARTITION variable=beta_reg dim=0 complete
#pragma HLS ARRAY_PARTITION variable=gamma_reg dim=0 complete
    // clang-format on
    for (int i = 0; i < 3 * T_CHANNELS_T; i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=12
#pragma HLS PIPELINE II=1
        // clang-format on
        float temp = params[i];
        if (i < T_CHANNELS_T)
            alpha_reg[i] = temp;
        else if (i < 2 * T_CHANNELS_T)
            beta_reg[i - T_CHANNELS_T] = temp;
        else
            gamma_reg[i - 2 * T_CHANNELS_T] = temp;
    }

    // TODO send this to arry2strm
    int loop_count = (rows * cols) / (NPC_T);
// clang-format off
#pragma HLS DATAFLOW
    // clang-format on
    // Arr2Strm<INPUT_PTR_WIDTH_T, T_CHANNELS_T, CPW_T, NPC_T, WX_T>(inp, srcStrm, rows, cols);
    InBitWidthConvert<CPW_T, NPC_T, WX_T, INPUT_PTR_WIDTH_T, strm_width_in>(srcStrm, srcStrmIn, loop_count);
    xFpreProcessKernel<strm_width_in, strm_width_out, T_CHANNELS_T, CPW_T, ROWS_T, COLS_T, NPC_T, PACK_MODE_T, WX_T,
                       WA_T, WB_T, WY_T, WO_T, FX_T, FA_T, FB_T, FY_T, FO_T, SIGNED_IN_T, OPMODE_T>(
        srcStrmIn, dstStrmOut, alpha_reg, beta_reg, gamma_reg, th1, th2, loop_count);
    OutBitWidthConvert<CPW_T, NPC_T, WO_T, OUTPUT_PTR_WIDTH_T, strm_width_out>(dstStrmOut, dstStrm, loop_count);
    Strm2Arr<OUTPUT_PTR_WIDTH_T, T_CHANNELS_T, CPW_T, NPC_T, WO_T>(dstStrm, out, rows, cols);
}
}
}
#endif