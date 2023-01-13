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

#ifndef _XF_HARRIS_UTILS_H_
#define _XF_HARRIS_UTILS_H_

#ifndef __cplusplus
#error C++ is needed to use this file!
#endif

/**
 *  xFDuplicate
 */
template <int SRC_T, int ROWS, int COLS, int DEPTH, int NPC, int WORDWIDTH, int TC>
void xFDuplicate(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                 xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst1_mat,
                 xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst2_mat,
                 uint16_t img_height,
                 uint16_t img_width) {
    img_width = img_width >> XF_BITSHIFT(NPC);

    ap_uint<13> row, col;
Row_Loop:
    for (row = 0; row < img_height; row++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        #pragma HLS LOOP_FLATTEN off
    // clang-format on
    Col_Loop:
        for (col = 0; col < img_width; col++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            #pragma HLS pipeline
            // clang-format on
            XF_SNAME(WORDWIDTH) tmp_src;
            tmp_src = _src_mat.read(row * img_width + col);
            _dst1_mat.write(row * img_width + col, tmp_src);
            _dst2_mat.write(row * img_width + col, tmp_src);
        }
    }
}

/**
 * xFSquare : Compute square of the input image
 */
template <int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int IN_DEPTH,
          int OUT_DEPTH,
          int NPC,
          int IN_WW,
          int OUT_WW,
          int TC,
          typename SCALE_T>
void xFSquare(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
              xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst_mat,
              SCALE_T scale,
              uint8_t filter_width,
              uint16_t img_height,
              uint16_t img_width) {
    img_width = img_width >> XF_BITSHIFT(NPC);

    ap_uint<13> row, col;
    XF_SNAME(IN_WW) tmp_src;
    XF_SNAME(OUT_WW) tmp_dst;
    uint16_t shift = 0;
    uint16_t npc = XF_NPIXPERCYCLE(NPC);

    XF_PTNAME(IN_DEPTH) src_buf[(1 << XF_BITSHIFT(NPC))];
    XF_PTNAME(OUT_DEPTH) dst_buf[(1 << XF_BITSHIFT(NPC))];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=src_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=dst_buf complete dim=1
// clang-format on
Row_Loop:
    for (row = 0; row < img_height; row++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        #pragma HLS LOOP_FLATTEN off
    // clang-format on
    Col_Loop:
        for (col = 0; col < img_width; col++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            #pragma HLS pipeline
            // clang-format on
            tmp_src = _src_mat.read(row * img_width + col);
            xfExtractPixels<NPC, IN_WW, IN_DEPTH>(src_buf, tmp_src, 0);

        Square_Loop:
            for (ap_uint<9> k = 0; k < (1 << XF_BITSHIFT(NPC)); k++) {
// clang-format off
                #pragma HLS UNROLL
                // clang-format on
                XF_PTNAME(IN_DEPTH) val;

                if (filter_width == XF_FILTER_7X7) {
                    int16_t val2;
                    uint16_t val1;
                    val2 = src_buf[k] >> 9;
                    if (val2 < 0)
                        val1 = -(val2);
                    else
                        val1 = val2;

                    dst_buf[k] = ((val1 * val1) >> scale);

                } else {
                    if (src_buf[k] < 0)
                        val = -(src_buf[k]);
                    else
                        val = src_buf[k];

                    dst_buf[k] = (val * val) >> scale;
                }
            }

            tmp_dst = 0;
            xfPackPixels<NPC, OUT_WW, OUT_DEPTH>(&dst_buf[0], tmp_dst, 0, npc, shift);
            shift = 0;
            _dst_mat.write(row * img_width + col, tmp_dst); // Write the data in to output stream
        }                                                   // Col_Loop
    }                                                       // Row_Loop
}
// xFSquare

/**
 *  xFMultiply : Compute dst = src1 * src2
 */
template <int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int IN_DEPTH,
          int OUT_DEPTH,
          int NPC,
          int IN_WW,
          int OUT_WW,
          int TC,
          typename SCALE_T>
void xFMultiply(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat1,
                xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat2,
                xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst_mat,
                SCALE_T scale,
                uint8_t filter_width,
                uint16_t img_height,
                uint16_t img_width) {
    img_width = img_width >> XF_BITSHIFT(NPC);

    ap_uint<13> row, col;
    XF_SNAME(IN_WW) tmp_src1, tmp_src2;
    XF_SNAME(OUT_WW) tmp_dst;
    XF_PTNAME(IN_DEPTH) src_buf1[(1 << XF_BITSHIFT(NPC))], src_buf2[(1 << XF_BITSHIFT(NPC))];
    XF_PTNAME(OUT_DEPTH) dst_buf[(1 << XF_BITSHIFT(NPC))];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=src_buf1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=src_buf2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=dst_buf complete dim=1
    // clang-format on

    uint16_t npc = XF_NPIXPERCYCLE(NPC);
    uint16_t shift = 0;

Row_Loop:
    for (row = 0; row < img_height; row++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        #pragma HLS LOOP_FLATTEN off
    // clang-format on
    Col_Loop:
        for (col = 0; col < img_width; col++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            #pragma HLS pipeline
            // clang-format on
            tmp_src1 = _src_mat1.read(row * img_width + col); // Read data from the source1
            tmp_src2 = _src_mat2.read(row * img_width + col); // Read data from the source2

            /* Extract data from source */
            xfExtractPixels<NPC, IN_WW, IN_DEPTH>(src_buf1, tmp_src1, 0);
            xfExtractPixels<NPC, IN_WW, IN_DEPTH>(src_buf2, tmp_src2, 0);

            for (ap_uint<9> k = 0; k < (1 << XF_BITSHIFT(NPC)); k++) {
// clang-format off
                #pragma HLS UNROLL
                // clang-format on
                XF_PTNAME(IN_DEPTH) val1 = src_buf1[k];
                XF_PTNAME(IN_DEPTH) val2 = src_buf2[k];
                // TODO:: I included only for filter 3x3
                if (filter_width == XF_FILTER_7X7) {
                    int16_t val11 = val1 >> 9;
                    int16_t val22 = val2 >> 9;
                    dst_buf[k] = (val11 * val22) >> scale;

                } else {
                    dst_buf[k] = (val1 * val2) >> scale;
                }
            }

            tmp_dst = 0;
            xfPackPixels<NPC, OUT_WW, OUT_DEPTH>(&dst_buf[0], tmp_dst, 0, npc, shift);
            shift = 0;

            _dst_mat.write(row * img_width + col, (tmp_dst)); // Write data into the output stream
        }                                                     // Col_Loop

    } // Row_Loop
}
// xFMultiply

/**
 * Thresholding function
 * Arguments: Input Stream, Output Stream, Threshold
 */
template <int SRC_T, int ROWS, int COLS, int DEPTH, int NPC, int WORDWIDTH, int TC>
void xFThreshold(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                 xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst_mat,
                 uint16_t threshold,
                 uint16_t img_height,
                 uint16_t img_width) {
    img_width = img_width >> XF_BITSHIFT(NPC);

    XF_SNAME(WORDWIDTH) tmp_src;
    int buf1;
    XF_PTNAME(DEPTH) thresh = threshold;
    int res[(1 << XF_BITSHIFT(NPC))];
    ap_uint<9> i, j;
    ap_uint<8> STEP = XF_PIXELDEPTH(XF_32UP);

Row_Loop:
    for (uint16_t row = 0; row < img_height; row++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        #pragma HLS LOOP_FLATTEN off
    // clang-format on
    Col_Loop:
        for (uint16_t col = 0; col < img_width; col++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            #pragma HLS pipeline
            // clang-format on
            tmp_src = _src_mat.read(row * img_width + col); // Read data from the input stream

        Threshold_Loop:
            for (i = 0, j = 0; i < (32 << XF_BITSHIFT(NPC)); i += 32) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                buf1 = tmp_src.range(i + 31, i);

                /*	Pack the data into result  */
                buf1 = (buf1 > thresh) ? buf1 : 0;
                res[j++] = (uint32_t)buf1;
            }

            uint16_t npc = XF_NPIXPERCYCLE(NPC);

            uint16_t shift = 0;
            tmp_src = 0;

            for (i = 0; i < npc; i++) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                uint32_t tmp = res[i];
                tmp_src = tmp_src | (((XF_SNAME(WORDWIDTH))tmp) << (shift * STEP));
                shift++;
            }
            shift = 0;

            _dst_mat.write(row * img_width + col, tmp_src); // Write data into output pixel
        }                                                   // Col_Loop

    } // Row_Loop
}
// xFThreshold

/**
 *  Compute score = det - k * trace^2
 *
 *  _src1_mat --> gx^2
 *  _src1_mat --> gx * gy
 *  _src1_mat --> gy^2
 *  _dst_mat --> Result
 */
template <int SRC_T, int DST_T, int ROWS, int COLS, int IN_DEPTH, int OUT_DEPTH, int NPC, int IN_WW, int OUT_WW, int TC>
void xFComputeScore(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src1_mat,
                    xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src2_mat,
                    xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src3_mat,
                    xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst_mat,
                    uint16_t img_height,
                    uint16_t img_width,
                    uint16_t thresold,
                    uint8_t _filter_type) {
    img_width = img_width >> XF_BITSHIFT(NPC);
    XF_SNAME(IN_WW) tmp_src1, tmp_src2, tmp_src3;

    XF_PTNAME(OUT_DEPTH) dst_buf[(1 << XF_BITSHIFT(NPC))];
    XF_SNAME(OUT_WW) tmp_dst;
    ap_uint<8> in_step = XF_PIXELDEPTH(IN_DEPTH);
    uint16_t in_sumloop = (XF_PIXELDEPTH(IN_DEPTH) << XF_BITSHIFT(NPC));
    uint16_t npc = XF_NPIXPERCYCLE(NPC);

    ap_int<32> tmp_res[2];
    ap_int<32> det_res;
    ap_int<17> trace_res;
// clang-format off
    #pragma HLS RESOURCE variable=trace_res core=DSP48 latency=2
    // clang-format on
    ap_int<50> trace_res1;
// clang-format off
    #pragma HLS RESOURCE variable=trace_res1 core=DSP48 latency=2
    // clang-format on
    ap_int<32> trace_res2;
// clang-format off
    #pragma HLS RESOURCE variable=trace_res2 core=DSP48 latency=2
    // clang-format on
    ap_uint<13> row, col;
    ap_uint<10> i, j;
    uint16_t shift = 0;

Row_Loop:
    for (row = 0; row < img_height; row++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        #pragma HLS LOOP_FLATTEN off
    // clang-format on
    Col_Loop:
        for (col = 0; col < img_width; col++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            #pragma HLS pipeline
            // clang-format on
            tmp_src1 = _src1_mat.read(row * img_width + col);
            tmp_src2 = _src2_mat.read(row * img_width + col);
            tmp_src3 = _src3_mat.read(row * img_width + col);

        Determinant_Loop:
            for (i = 0, j = 0; i < in_sumloop; i += in_step) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                XF_PTNAME(IN_DEPTH) val1, val2, val3;
                val1 = tmp_src1.range(i + (in_step - 1), i);
                val2 = tmp_src2.range(i + (in_step - 1), i);
                val3 = tmp_src3.range(i + (in_step - 1), i);
                // TODO:shift according to box filter
                if (_filter_type == XF_FILTER_7X7) {
                    val1 = val1 >> 0;
                    val2 = val2 >> 0;
                    val3 = val3 >> 0;
                } else {
                    val1 = val1 >> 2;
                    val2 = val2 >> 2;
                    val3 = val3 >> 2;
                }

                /* Compute determinant */

                tmp_res[0] = ((ap_int<32>)val1 * (ap_int<32>)val2);
                tmp_res[1] = ((ap_int<32>)val3 * (ap_int<32>)val3);

                det_res = tmp_res[0] - tmp_res[1];

                /*	Compute trace */
                trace_res = val1 + val2;

                /*	Compute det - k*trace^2	 */
                trace_res1 = trace_res * trace_res;
                trace_res2 = (trace_res1 * thresold) >> 16;

                if (_filter_type == XF_FILTER_7X7) {
                    dst_buf[j++] = (XF_PTNAME(OUT_DEPTH))((det_res - trace_res2) >> 8);
                } else {
                    dst_buf[j++] = (XF_PTNAME(OUT_DEPTH))(det_res - trace_res2);
                }
            }
            tmp_dst = 0;
            xfPackPixels<NPC, OUT_WW, OUT_DEPTH>(&dst_buf[0], tmp_dst, 0, npc, shift);
            shift = 0;
            _dst_mat.write(row * img_width + col, (tmp_dst)); // Write data into output pixel
        }
    }
}
// xFDeterminant

#endif // _XF_HARRIS_UTILS_H_
