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

#ifndef _XF_UTILITY_H_
#define _XF_UTILITY_H_

#include "ap_axi_sdata.h"
#include "common/xf_common.hpp"
#include <assert.h>
#include <string.h>

namespace xf {
namespace cv {

/**
 * Extract Pixels from a packed word into an array from the index pos.
 * The number of pixels to be extracted is determined by the NPC.
 */

template <int NPC, int WORDWIDTH, int PIXELDEPTH>
void xfPackPixels(
    XF_PTNAME(PIXELDEPTH) * tmp_buf, XF_SNAME(WORDWIDTH) & val, uint16_t pos, int16_t loopIter, uint16_t& shift) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    ap_uint<8> STEP = XF_PIXELDEPTH(PIXELDEPTH);

    for (ap_int<9> i = 0; i < loopIter; i++) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        XF_PTUNAME(PIXELDEPTH) tmp = tmp_buf[pos];
        val = val | (((XF_SNAME(WORDWIDTH))tmp) << (shift * STEP));
        pos++;
        shift++;
    }
}

template <int NPC, int WORDWIDTH, int PIXELDEPTH>
void xfExtractPixels(XF_PTNAME(PIXELDEPTH) * tmp_buf, XF_SNAME(WORDWIDTH) & val1, int pos) {
// clang-format off
    #pragma HLS inline off
    // clang-format on
    XF_SNAME(WORDWIDTH) v = val1;

    int shift = 0;
    int STEP = XF_PIXELDEPTH(PIXELDEPTH);
Extract_pixels_loop:
    for (int i = 0; i < (1 << (XF_BITSHIFT(NPC))); i++) {
// clang-format off
        #pragma HLS UNROLL
        // clang-format on
        tmp_buf[pos + i] = v.range(shift + STEP - 1, shift);
        shift = shift + STEP;
    }
}

template <int NPC, int WORDWIDTH_SRC, int DEPTH_SRC>
void xfExtractData(XF_PTNAME(DEPTH_SRC) * src_buf1,
                   XF_PTNAME(DEPTH_SRC) * src_buf2,
                   XF_PTNAME(DEPTH_SRC) * src_buf3,
                   XF_PTNAME(DEPTH_SRC) * src_buf4,
                   XF_PTNAME(DEPTH_SRC) * src_buf5,
                   XF_PTNAME(DEPTH_SRC) * src_buf6,
                   XF_PTNAME(DEPTH_SRC) * src_buf7,
                   XF_SNAME(WORDWIDTH_SRC) buf0,
                   XF_SNAME(WORDWIDTH_SRC) buf1,
                   XF_SNAME(WORDWIDTH_SRC) buf2,
                   XF_SNAME(WORDWIDTH_SRC) buf3,
                   XF_SNAME(WORDWIDTH_SRC) buf4,
                   XF_SNAME(WORDWIDTH_SRC) buf5,
                   XF_SNAME(WORDWIDTH_SRC) buf6) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    xfExtractPixels<NPC, WORDWIDTH_SRC, DEPTH_SRC>(&src_buf1[6], buf0, 0);
    xfExtractPixels<NPC, WORDWIDTH_SRC, DEPTH_SRC>(&src_buf2[6], buf1, 0);
    xfExtractPixels<NPC, WORDWIDTH_SRC, DEPTH_SRC>(&src_buf3[6], buf2, 0);
    xfExtractPixels<NPC, WORDWIDTH_SRC, DEPTH_SRC>(&src_buf4[6], buf3, 0);
    xfExtractPixels<NPC, WORDWIDTH_SRC, DEPTH_SRC>(&src_buf5[6], buf4, 0);
    xfExtractPixels<NPC, WORDWIDTH_SRC, DEPTH_SRC>(&src_buf6[6], buf5, 0);
    xfExtractPixels<NPC, WORDWIDTH_SRC, DEPTH_SRC>(&src_buf7[6], buf6, 0);
}

template <int NPC, int DEPTH_SRC>
void xfCopyData(XF_PTNAME(DEPTH_SRC) src_buf1[XF_NPIXPERCYCLE(NPC) + 6],
                XF_PTNAME(DEPTH_SRC) src_buf2[XF_NPIXPERCYCLE(NPC) + 6],
                XF_PTNAME(DEPTH_SRC) src_buf3[XF_NPIXPERCYCLE(NPC) + 6],
                XF_PTNAME(DEPTH_SRC) src_buf4[XF_NPIXPERCYCLE(NPC) + 6],
                XF_PTNAME(DEPTH_SRC) src_buf5[XF_NPIXPERCYCLE(NPC) + 6],
                XF_PTNAME(DEPTH_SRC) src_buf6[XF_NPIXPERCYCLE(NPC) + 6],
                XF_PTNAME(DEPTH_SRC) src_buf7[XF_NPIXPERCYCLE(NPC) + 6]) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    ap_uint<5> buf_size = (XF_NPIXPERCYCLE(NPC) + 6);
    ap_uint<4> i = 0;
    ap_uint<4> ind = buf_size - 6;

    for (i = 0; i < 6; i++, ind++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=6 max=6
        #pragma HLS unroll
        // clang-format on
        src_buf1[i] = src_buf1[ind];
        src_buf2[i] = src_buf2[ind];
        src_buf3[i] = src_buf3[ind];
        src_buf4[i] = src_buf4[ind];
        src_buf5[i] = src_buf5[ind];
        src_buf6[i] = src_buf6[ind];
        src_buf7[i] = src_buf7[ind];
    }
}

/**
 * CopyMemoryOut: Copies memory from BRAM to DDR
 */
template <int SIZE, int WORDWIDTH>
void xFCopyBlockMemoryOut1(XF_SNAME(WORDWIDTH) * _src, unsigned long long int* _dst, int nbytes) {
#if _XF_SYNTHESIS_
    memcpy((unsigned long long int*)_dst, (unsigned long long int*)_src, SIZE);
#else
    if (nbytes) memcpy((unsigned long long int*)_dst, (unsigned long long int*)_src, nbytes);
#endif
}

/**
 * CopyMemoryIn: Copies memory from DDR to BRAM if y_offset and x_offset is
 * provided
 */
template <int SIZE, int WORDWIDTH>
void xFCopyBlockMemoryIn1(unsigned long long int* _src, XF_SNAME(WORDWIDTH) * _dst, int nbytes) {
#if _XF_SYNTHESIS_
    memcpy((XF_SNAME(WORDWIDTH)*)_dst, (XF_SNAME(WORDWIDTH)*)_src, SIZE);
#else
    memcpy((XF_SNAME(WORDWIDTH)*)_dst, (XF_SNAME(WORDWIDTH)*)_src, nbytes);
#endif
}

/**
 * CopyMemoryIn: Copies memory from DDR to BRAM if y_offset and x_offset is
 * provided
 */
template <int SIZE, int WORDWIDTH>
void xFCopyBlockMemoryIn(XF_SNAME(WORDWIDTH) * _src, XF_SNAME(WORDWIDTH) * _dst, int nbytes) {
#if _XF_SYNTHESIS_
    memcpy((AU_TNAME(WORDWIDTH)*)_dst, (AU_TNAME(WORDWIDTH)*)_src, SIZE);
#else
    memcpy((XF_SNAME(WORDWIDTH)*)_dst, (XF_SNAME(WORDWIDTH)*)_src, nbytes);
#endif
}

/**
 * CopyMemoryOut: Copies memory from BRAM to DDR
 */
template <int SIZE, int WORDWIDTH>
void xFCopyBlockMemoryOut(XF_SNAME(WORDWIDTH) * _src, XF_SNAME(WORDWIDTH) * _dst, int nbytes) {
#if _XF_SYNTHESIS_
    memcpy((XF_SNAME(WORDWIDTH)*)_dst, (XF_SNAME(WORDWIDTH)*)_src, SIZE);
#else
    memcpy((XF_SNAME(WORDWIDTH)*)_dst, (XF_SNAME(WORDWIDTH)*)_src, nbytes);
#endif
}

template <int WORDWIDTH, int NPC, int IN_BH, int IN_BW>
void xFDuplicateStream(hls::stream<XF_SNAME(WORDWIDTH)>& in_strm,
                       hls::stream<XF_SNAME(WORDWIDTH)>& out_strm1,
                       hls::stream<XF_SNAME(WORDWIDTH)>& out_strm2,
                       int imwidth,
                       int imheight) {
    for (int i = 0; i < imheight; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=IN_BH max=IN_BH
        #pragma HLS LOOP_FLATTEN off
        // clang-format on
        for (int j = 0; j < (imwidth >> NPC); j++) {
// clang-format off
            #pragma HLS pipeline
            #pragma HLS LOOP_TRIPCOUNT min=IN_BW max=IN_BW
            // clang-format on
            XF_SNAME(WORDWIDTH) tmp = in_strm.read();
            out_strm1.write(tmp);
            out_strm2.write(tmp);
        }
    }
}

// ==============================================================================
// Class contains funcitons requried for accel file (top wrapper file)
// ==============================================================================
class accel_utils {
   public:
    // ==============================================================================
    // Read module(s) to handle data transfer from AXI/HLS stream to xfMat
    // ------------------------------------------------------------------------------

    template <int PTR_WIDTH, int ROWS, int COLS, int NPC, int COLOR_T, int CH_WIDTH, int TRIPCOUNT>
    void Array2hlsStrm(ap_uint<PTR_WIDTH>* srcPtr, hls::stream<ap_uint<PTR_WIDTH> >& dstStrm, int rows, int cols) {
        int pixel_width = COLOR_T * CH_WIDTH;
        int loop_count = (((rows * cols * pixel_width) + PTR_WIDTH - 1) / PTR_WIDTH);

        for (int i = 0; i < loop_count; i++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=TRIPCOUNT
            #pragma HLS PIPELINE
            // clang-format on
            dstStrm.write(srcPtr[i]);
        }
    }

    template <int PTR_WIDTH, int MAT_T, int ROWS, int COLS, int NPC, int TRIPCOUNT>
    void hlsStrm2xfMat(hls::stream<ap_uint<PTR_WIDTH> >& srcStrm,
                       xf::cv::Mat<MAT_T, ROWS, COLS, NPC>& dstMat,
                       int dstMat_cols_align_npc) {
        int rows = dstMat.rows;
        int cols = dstMat.cols;
        int loop_count = (rows * dstMat_cols_align_npc) / XF_NPIXPERCYCLE(NPC);
        int pad = dstMat_cols_align_npc - cols;
        int in_size_bits = XF_PIXELWIDTH(MAT_T, NPC) * rows * dstMat_cols_align_npc; // channels
        int ddr_read_cycles = (((in_size_bits) + (PTR_WIDTH)-1) / (PTR_WIDTH));
        int ddr_read_cnt = 0;

        int valid_bits = 0;
        const int N_size = XF_PIXELWIDTH(MAT_T, NPC) * XF_NPIXPERCYCLE(NPC);
        const int last_N_size = XF_PIXELWIDTH(MAT_T, NPC) * (XF_NPIXPERCYCLE(NPC) - pad);
        const int PTR_WIDTH_min_N = PTR_WIDTH - N_size;
        const int PTR_WIDTH_min_last_N = PTR_WIDTH - last_N_size;
        const int PTR_WIDTH_plus_N = PTR_WIDTH + N_size;
        const int PTR_WIDTH_plus_last_N = PTR_WIDTH + last_N_size;

        int K_size;
        ap_uint<PTR_WIDTH> r;
        XF_TNAME(MAT_T, NPC) out;
        int ncpr = dstMat_cols_align_npc / XF_NPIXPERCYCLE(NPC); // number of clock per row
        int clk_cnt = 0;                                         // clock counter. reset at the start of every row
        int strm_cnt_disply = 0;
    L1:
        for (int i = 0; i < loop_count; i++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=TRIPCOUNT
            #pragma HLS PIPELINE
            // clang-format on

            int PTR_WIDTH_min_Ksize;
            int PTR_WIDTH_plus_Ksize;

            if (clk_cnt == ncpr - 1) {
                clk_cnt = 0;
                K_size = last_N_size;
                PTR_WIDTH_min_Ksize = PTR_WIDTH_min_last_N;
                PTR_WIDTH_plus_Ksize = PTR_WIDTH_plus_last_N;
            } else {
                clk_cnt++;
                K_size = N_size;
                PTR_WIDTH_min_Ksize = PTR_WIDTH_min_N;
                PTR_WIDTH_plus_Ksize = PTR_WIDTH_plus_N;
            }

            int valid_bits_update;
            int valid_bits_tmp = valid_bits - K_size;
            XF_TNAME(MAT_T, NPC) out = 0;

            if (valid_bits < K_size) {
                if (valid_bits != 0) {
                    out.range(valid_bits - 1, 0) = r.range(PTR_WIDTH - 1, PTR_WIDTH - valid_bits);
                }
                if (ddr_read_cnt < ddr_read_cycles) {
                    r = srcStrm.read();
                    ddr_read_cnt++;
                } else {
                    r = 0;
                }
                out.range(K_size - 1, valid_bits) = r.range(K_size - valid_bits - 1, 0);
                valid_bits = PTR_WIDTH_min_Ksize + valid_bits;
            } else {
                out = r.range(PTR_WIDTH_plus_Ksize - valid_bits - 1, PTR_WIDTH - valid_bits);
                valid_bits = valid_bits - K_size;
            }

            dstMat.write(i, out);
        }
        int stop = 0;
    }

    template <int PTR_WIDTH, int MAT_T, int ROWS, int COLS, int NPC>
    void Array2xfMat(ap_uint<PTR_WIDTH>* srcPtr, xf::cv::Mat<MAT_T, ROWS, COLS, NPC>& dstMat, int stride = -1) {
#if !defined(__XF_USE_OLD_IMPL__)
        MMIterIn<PTR_WIDTH, MAT_T, ROWS, COLS, NPC>::Array2xfMat(srcPtr, dstMat, stride);
#else
// clang-format off
        #pragma HLS DATAFLOW
        // clang-format on
        assert((PTR_WIDTH >= XF_WORDDEPTH(XF_WORDWIDTH(MAT_T, NPC))) &&
               "The PTR_WIDTH must be always greater than or equal to the minimum "
               "width for the corresponding "
               "configuration");
        const int ch_width = XF_DTPIXELDEPTH(MAT_T, NPC);

        hls::stream<ap_uint<PTR_WIDTH> > strm;
        int rows = dstMat.rows;
        int cols = dstMat.cols;
        int dstMat_cols_align_npc = ((dstMat.cols + (NPC - 1)) >> XF_BITSHIFT(NPC)) << XF_BITSHIFT(NPC);
        Array2hlsStrm<PTR_WIDTH, ROWS, COLS, NPC, XF_CHANNELS(MAT_T, NPC), ch_width,
                      ((ROWS * COLS * XF_CHANNELS(MAT_T, NPC) * ch_width) / PTR_WIDTH)>(srcPtr, strm, rows, cols);
        hlsStrm2xfMat<PTR_WIDTH, MAT_T, ROWS, COLS, NPC, (ROWS * COLS) / NPC>(strm, dstMat, dstMat_cols_align_npc);
#endif
    }

    template <int PTR_WIDTH, int ROWS, int COLS, int NPC, int COLOR_T, int CH_WIDTH, int TRIPCOUNT>
    void axiStrm2hlsStrm(hls::stream<ap_axiu<PTR_WIDTH, 0, 0, 0> >& srcPtr,
                         hls::stream<ap_uint<PTR_WIDTH> >& dstStrm,
                         int rows,
                         int cols) {
        int pixel_width = COLOR_T * CH_WIDTH;
        int loop_count = (((rows * cols * pixel_width) + PTR_WIDTH - 1) / PTR_WIDTH);

        for (int i = 0; i < loop_count; i++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=TRIPCOUNT
            #pragma HLS PIPELINE
            // clang-format on
            ap_axiu<PTR_WIDTH, 0, 0, 0> v = srcPtr.read();
            dstStrm.write(v.data);
        }
    }

    template <int PTR_WIDTH, int MAT_T, int ROWS, int COLS, int NPC>
    void axiStrm2xfMat(hls::stream<ap_axiu<PTR_WIDTH, 0, 0, 0> >& srcPtr, xf::cv::Mat<MAT_T, ROWS, COLS, NPC>& dstMat) {
// clang-format off
        #pragma HLS DATAFLOW
        // clang-format on
        assert((PTR_WIDTH >= XF_WORDDEPTH(XF_WORDWIDTH(MAT_T, NPC))) &&
               "The PTR_WIDTH must be always greater than or equal to the minimum "
               "width for the corresponding "
               "configuration");
        const int ch_width = XF_DTPIXELDEPTH(MAT_T, NPC);

        hls::stream<ap_uint<PTR_WIDTH> > strm;
        int rows = dstMat.rows;
        int cols = dstMat.cols;
        int dstMat_cols_align_npc = ((dstMat.cols + (NPC - 1)) >> XF_BITSHIFT(NPC)) << XF_BITSHIFT(NPC);
        axiStrm2hlsStrm<PTR_WIDTH, ROWS, COLS, NPC, XF_CHANNELS(MAT_T, NPC), ch_width,
                        ((ROWS * COLS * XF_CHANNELS(MAT_T, NPC) * ch_width) / PTR_WIDTH)>(srcPtr, strm, rows, cols);
        hlsStrm2xfMat<PTR_WIDTH, MAT_T, ROWS, COLS, NPC, (ROWS * COLS) / NPC>(strm, dstMat, dstMat_cols_align_npc);
    }

    // ==============================================================================
    // Write module(s) to handle data transfer from xfMat to AXI/HLS stream
    // ------------------------------------------------------------------------------

    template <int PTR_WIDTH, int MAT_T, int ROWS, int COLS, int NPC, int TRIPCOUNT>
    void xfMat2hlsStrm(xf::cv::Mat<MAT_T, ROWS, COLS, NPC>& srcMat,
                       hls::stream<ap_uint<PTR_WIDTH> >& dstStrm,
                       int srcMat_cols_align_npc) {
        int rows = srcMat.rows;
        int cols = srcMat.cols;
        int loop_count = (rows * srcMat_cols_align_npc) / XF_NPIXPERCYCLE(NPC);
        int pad = srcMat_cols_align_npc - cols;
        int out_size_bits = XF_PIXELWIDTH(MAT_T, NPC) * rows * srcMat_cols_align_npc; // channels
        int ddr_write_cycles = (((out_size_bits) + (PTR_WIDTH)-1) / (PTR_WIDTH));
        int ddr_write_cnt = 0;

        int bits_to_add = PTR_WIDTH;
        const int N_size = XF_PIXELWIDTH(MAT_T, NPC) * XF_NPIXPERCYCLE(NPC);
        const int last_N_size = XF_PIXELWIDTH(MAT_T, NPC) * (XF_NPIXPERCYCLE(NPC) - pad);
        const int PTR_WIDTH_min_N = PTR_WIDTH - N_size;
        const int PTR_WIDTH_min_last_N = PTR_WIDTH - last_N_size;
        const int PTR_WIDTH_plus_N = PTR_WIDTH + N_size;
        const int PTR_WIDTH_plus_last_N = PTR_WIDTH + last_N_size;

        ap_uint<PTR_WIDTH> r;
        XF_TNAME(MAT_T, NPC) in;
        int ncpr = srcMat_cols_align_npc / XF_NPIXPERCYCLE(NPC); // number of clock per row
        int clk_cnt = 0;                                         // clock counter. reset at the start of every row

    L1:
        for (int i = 0; i < loop_count; i++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=TRIPCOUNT
            #pragma HLS PIPELINE
            // clang-format on
            int K_size;
            int PTR_WIDTH_min_Ksize;
            int PTR_WIDTH_plus_Ksize;
            if (clk_cnt == ncpr - 1) {
                clk_cnt = 0;
                K_size = last_N_size;
                PTR_WIDTH_min_Ksize = PTR_WIDTH_min_last_N;
                PTR_WIDTH_plus_Ksize = PTR_WIDTH_plus_last_N;
            } else {
                clk_cnt++;
                K_size = N_size;
                PTR_WIDTH_min_Ksize = PTR_WIDTH_min_N;
                PTR_WIDTH_plus_Ksize = PTR_WIDTH_plus_N;
            }

            in = srcMat.read(i);

            if (bits_to_add <= K_size) {
                r.range(PTR_WIDTH - 1, PTR_WIDTH - bits_to_add) = in.range(bits_to_add - 1, 0);
                dstStrm.write(r);

                if (bits_to_add != K_size) {
                    r.range(K_size - bits_to_add - 1, 0) = in.range(K_size - 1, bits_to_add);
                }
                bits_to_add = PTR_WIDTH_min_Ksize + bits_to_add;
            } else {
                r.range(PTR_WIDTH_plus_Ksize - bits_to_add - 1, PTR_WIDTH - bits_to_add) = in;
                bits_to_add -= K_size;
            }
        }

        if (bits_to_add != PTR_WIDTH) {
            dstStrm.write(r);
        }
    }

    template <int PTR_WIDTH, int ROWS, int COLS, int NPC, int COLOR_T, int CH_WIDTH, int TRIPCOUNT>
    void hlsStrm2Array(hls::stream<ap_uint<PTR_WIDTH> >& srcStrm, ap_uint<PTR_WIDTH>* dstPtr, int rows, int cols) {
        int pixel_width = COLOR_T * CH_WIDTH;
        int loop_count = (((rows * cols * pixel_width) + PTR_WIDTH - 1) / PTR_WIDTH);

        for (int i = 0; i < loop_count; i++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=TRIPCOUNT
            #pragma HLS PIPELINE
            // clang-format on
            dstPtr[i] = srcStrm.read();
        }
    }

    template <int PTR_WIDTH, int MAT_T, int ROWS, int COLS, int NPC, int FILLZERO = 1>
    void xfMat2Array(xf::cv::Mat<MAT_T, ROWS, COLS, NPC>& srcMat, ap_uint<PTR_WIDTH>* dstPtr, int stride = -1) {
#if !defined(__XF_USE_OLD_IMPL__)
        MMIterOut<PTR_WIDTH, MAT_T, ROWS, COLS, NPC, FILLZERO>::xfMat2Array(srcMat, dstPtr, stride);
#else
// clang-format off
        #pragma HLS DATAFLOW
        // clang-format on
        assert((PTR_WIDTH >= XF_WORDDEPTH(XF_WORDWIDTH(MAT_T, NPC))) &&
               "The PTR_WIDTH must be always greater than or equal to the minimum "
               "width for the corresponding "
               "configuration");
        const int ch_width = XF_DTPIXELDEPTH(MAT_T, NPC);

        hls::stream<ap_uint<PTR_WIDTH> > strm;
        int rows = srcMat.rows;
        int cols = srcMat.cols;
        int srcMat_cols_align_npc = ((srcMat.cols + (NPC - 1)) >> XF_BITSHIFT(NPC)) << XF_BITSHIFT(NPC);

        xfMat2hlsStrm<PTR_WIDTH, MAT_T, ROWS, COLS, NPC, ROWS*((COLS + NPC - 1) / NPC)>(srcMat, strm,
                                                                                        srcMat_cols_align_npc);
        hlsStrm2Array<PTR_WIDTH, ROWS, COLS, NPC, XF_CHANNELS(MAT_T, NPC), ch_width,
                      ((ROWS * COLS * XF_CHANNELS(MAT_T, NPC) * ch_width) / PTR_WIDTH)>(strm, dstPtr, rows, cols);
#endif
    }

    template <int PTR_WIDTH, int ROWS, int COLS, int NPC, int COLOR_T, int CH_WIDTH, int TRIPCOUNT>
    void hlsStrm2axiStrm(hls::stream<ap_uint<PTR_WIDTH> >& srcStrm,
                         hls::stream<ap_axiu<PTR_WIDTH, 0, 0, 0> >& dstPtr,
                         int rows,
                         int cols) {
        int pixel_width = COLOR_T * CH_WIDTH;
        int loop_count = (((rows * cols * pixel_width) + PTR_WIDTH - 1) / PTR_WIDTH);

        for (int i = 0; i < loop_count; i++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=TRIPCOUNT
            #pragma HLS PIPELINE
            // clang-format on
            ap_axiu<PTR_WIDTH, 0, 0, 0> v;
            v.data = srcStrm.read();
            dstPtr.write(v);
        }
    }

    template <int PTR_WIDTH, int MAT_T, int ROWS, int COLS, int NPC>
    void xfMat2axiStrm(xf::cv::Mat<MAT_T, ROWS, COLS, NPC>& srcMat, hls::stream<ap_axiu<PTR_WIDTH, 0, 0, 0> >& dstPtr) {
// clang-format off
        #pragma HLS DATAFLOW
        // clang-format on
        assert((PTR_WIDTH >= XF_WORDDEPTH(XF_WORDWIDTH(MAT_T, NPC))) &&
               "The PTR_WIDTH must be always greater than or equal to the minimum "
               "width for the corresponding "
               "configuration");
        const int ch_width = XF_DTPIXELDEPTH(MAT_T, NPC);

        hls::stream<ap_uint<PTR_WIDTH> > strm;
        int rows = srcMat.rows;
        int cols = srcMat.cols;
        int srcMat_cols_align_npc = ((srcMat.cols + (NPC - 1)) >> XF_BITSHIFT(NPC)) << XF_BITSHIFT(NPC);

        xfMat2hlsStrm<PTR_WIDTH, MAT_T, ROWS, COLS, NPC, ROWS*((COLS + NPC - 1) / NPC)>(srcMat, strm,
                                                                                        srcMat_cols_align_npc);
        hlsStrm2axiStrm<PTR_WIDTH, ROWS, COLS, NPC, XF_CHANNELS(MAT_T, NPC), ch_width,
                        ((ROWS * COLS * XF_CHANNELS(MAT_T, NPC) * ch_width) / PTR_WIDTH)>(strm, dstPtr, rows, cols);
    }
};

template <int PTR_WIDTH, int MAT_T, int ROWS, int COLS, int NPC, int FILLZERO = 1>
void xfMat2Array(xf::cv::Mat<MAT_T, ROWS, COLS, NPC>& srcMat, ap_uint<PTR_WIDTH>* dstPtr, int stride = -1) {
#if !defined(__XF_USE_OLD_IMPL__)
    MMIterOut<PTR_WIDTH, MAT_T, ROWS, COLS, NPC, FILLZERO>::xfMat2Array(srcMat, dstPtr, stride);
#else
    accel_utils au;
    au.xfMat2Array<PTR_WIDTH, MAT_T, ROWS, COLS, NPC>(srcMat, dstPtr);
#endif
}

template <int PTR_WIDTH, int MAT_T, int ROWS, int COLS, int NPC>
void Array2xfMat(ap_uint<PTR_WIDTH>* srcPtr, xf::cv::Mat<MAT_T, ROWS, COLS, NPC>& dstMat, int stride = -1) {
#if !defined(__XF_USE_OLD_IMPL__)
    MMIterIn<PTR_WIDTH, MAT_T, ROWS, COLS, NPC>::Array2xfMat(srcPtr, dstMat, stride);
#else
    accel_utils au;
    au.Array2xfMat<PTR_WIDTH, MAT_T, ROWS, COLS, NPC>(srcPtr, dstMat);
#endif
}

template <int PTR_WIDTH, int MAT_T, int ROWS, int COLS, int NPC>
void xfMat2axiStrm(xf::cv::Mat<MAT_T, ROWS, COLS, NPC>& srcMat, hls::stream<ap_axiu<PTR_WIDTH, 0, 0, 0> >& dstPtr) {
    accel_utils au;
    au.xfMat2axiStrm<PTR_WIDTH, MAT_T, ROWS, COLS, NPC>(srcMat, dstPtr);
}

template <int PTR_WIDTH, int MAT_T, int ROWS, int COLS, int NPC>
void axiStrm2xfMat(hls::stream<ap_axiu<PTR_WIDTH, 0, 0, 0> >& srcPtr, xf::cv::Mat<MAT_T, ROWS, COLS, NPC>& dstMat) {
    accel_utils au;
    au.axiStrm2xfMat<PTR_WIDTH, MAT_T, ROWS, COLS, NPC>(srcPtr, dstMat);
}

} // namespace cv
} // namespace xf

#endif //_XF_UTILITY_H_
