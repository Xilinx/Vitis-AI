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

#ifndef _XF_HOG_DESCRIPTOR_GRADIENTS_HPP_
#define _XF_HOG_DESCRIPTOR_GRADIENTS_HPP_

// maximum of three numbers function
#define MAX_MAG_OF_3_IDX(a, b, c) ((a > b ? a : b) > c ? (a > b ? 0 : 1) : 2)

/*****************************************************************
 * 		               Gradient computation
 *****************************************************************
 * X-Gradient Computation
 *
 * -----------
 * |-1  0  1 |
 * -----------
 *
 * Y-Gradient Computation
 * -----
 * |-1 |
 * | 0 |
 * | 1 |
 * -----
 *
 **********************************************************************/
template <int DEPTH_SRC, int DEPTH_DST, int NOC>
XF_PTNAME(DEPTH_DST)
xFHOGgradientXY(XF_PTNAME(DEPTH_SRC) n1, XF_PTNAME(DEPTH_SRC) n2) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    XF_PTNAME(DEPTH_DST) grad;

    grad = n2 - n1;

    return grad;
}

/**********************************************************************
 * xFHOGgradientCompute : Applies the mask and Computes the
 * 			gradient values.
 **********************************************************************/
template <int NPC, int DEPTH_SRC, int DEPTH_DST, int NOC, typename filter_type, int filter_width>
void xFHOGgradientCompute(XF_PTNAME(DEPTH_DST) * GradientvaluesX,
                          XF_PTNAME(DEPTH_DST) * GradientvaluesY,
                          filter_type src_buf0[][filter_width],
                          filter_type src_buf1[][filter_width],
                          filter_type src_buf2[][filter_width]) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    if (NOC == XF_GRAY) {
    Compute_Grad_Loop_Gray:
        for (uchar_t j = 0; j < XF_NPIXPERCYCLE(NPC); j++) {
// clang-format off
            #pragma HLS UNROLL
            // clang-format on

            // x-gradient computation
            GradientvaluesX[j] =
                xFHOGgradientXY<DEPTH_SRC, DEPTH_DST, NOC>(src_buf1[NOC - 1][j], src_buf1[NOC - 1][j + 2]);

            // y-gradient computation
            GradientvaluesY[j] =
                xFHOGgradientXY<DEPTH_SRC, DEPTH_DST, NOC>(src_buf0[NOC - 1][j + 1], src_buf2[NOC - 1][j + 1]);
        }
    } else {
        // Temporary array to hold the gradient data for each channel separately
        XF_PTNAME(DEPTH_DST) tmp_x[NOC], tmp_y[NOC];
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=tmp_x complete dim=1
        #pragma HLS ARRAY_PARTITION variable=tmp_y complete dim=1
    // clang-format on

    Compute_Grad_Loop_rgb:
        for (uchar_t j = 0; j < XF_NPIXPERCYCLE(NPC); j++) {
// clang-format off
            #pragma HLS UNROLL
            // clang-format on

            // x-gradient computation
            tmp_x[NOC - 3] = xFHOGgradientXY<DEPTH_SRC, DEPTH_DST, NOC>(src_buf1[NOC - 3][j], src_buf1[NOC - 3][j + 2]);
            tmp_x[NOC - 2] = xFHOGgradientXY<DEPTH_SRC, DEPTH_DST, NOC>(src_buf1[NOC - 2][j], src_buf1[NOC - 2][j + 2]);
            tmp_x[NOC - 1] = xFHOGgradientXY<DEPTH_SRC, DEPTH_DST, NOC>(src_buf1[NOC - 1][j], src_buf1[NOC - 1][j + 2]);

            // y-gradient computation
            tmp_y[NOC - 3] =
                xFHOGgradientXY<DEPTH_SRC, DEPTH_DST, NOC>(src_buf0[NOC - 3][j + 1], src_buf2[NOC - 3][j + 1]);
            tmp_y[NOC - 2] =
                xFHOGgradientXY<DEPTH_SRC, DEPTH_DST, NOC>(src_buf0[NOC - 2][j + 1], src_buf2[NOC - 2][j + 1]);
            tmp_y[NOC - 1] =
                xFHOGgradientXY<DEPTH_SRC, DEPTH_DST, NOC>(src_buf0[NOC - 1][j + 1], src_buf2[NOC - 1][j + 1]);

            // finding the maximum magnitude of RGB planes
            int mag_r = ((tmp_x[NOC - 3] * tmp_x[NOC - 3]) + (tmp_y[NOC - 3] * tmp_y[NOC - 3]));
            int mag_g = ((tmp_x[NOC - 2] * tmp_x[NOC - 2]) + (tmp_y[NOC - 2] * tmp_y[NOC - 2]));
            int mag_b = ((tmp_x[NOC - 1] * tmp_x[NOC - 1]) + (tmp_y[NOC - 1] * tmp_y[NOC - 1]));

            // gradient of higher magnitude plane is written to output array
            GradientvaluesX[j] = tmp_x[MAX_MAG_OF_3_IDX(mag_r, mag_g, mag_b)];
            GradientvaluesY[j] = tmp_y[MAX_MAG_OF_3_IDX(mag_r, mag_g, mag_b)];
        }
    }
}

/**************************************************************************************
 * xFHOGcomputeColGrad : Computes HoG gradients for the column input data
 **************************************************************************************/
template <int ROWS,
          int COLS,
          int DEPTH_SRC,
          int DEPTH_DST,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int NOS_SRC,
          int TC,
          int PIX_COUNT>
void xFHOGcomputeColGrad(hls::stream<XF_SNAME(WORDWIDTH_SRC)> _src_strm[NOS_SRC],
                         hls::stream<XF_SNAME(WORDWIDTH_DST)>& _gradx_strm,
                         hls::stream<XF_SNAME(WORDWIDTH_DST)>& _grady_strm,
                         XF_SNAME(WORDWIDTH_SRC) buf[NOS_SRC][3][(COLS >> XF_BITSHIFT(NPC))],
                         XF_PTNAME(DEPTH_SRC) src_buf0[NOS_SRC][XF_NPIXPERCYCLE(NPC) + 2],
                         XF_PTNAME(DEPTH_SRC) src_buf1[NOS_SRC][XF_NPIXPERCYCLE(NPC) + 2],
                         XF_PTNAME(DEPTH_SRC) src_buf2[NOS_SRC][XF_NPIXPERCYCLE(NPC) + 2],
                         XF_PTNAME(DEPTH_DST) GradientValuesX[XF_NPIXPERCYCLE(NPC)],
                         XF_PTNAME(DEPTH_DST) GradientValuesY[XF_NPIXPERCYCLE(NPC)],
                         XF_SNAME(WORDWIDTH_DST) & P0,
                         XF_SNAME(WORDWIDTH_DST) & P1,
                         uint16_t img_width,
                         ap_uint<13> row_ind,
                         ap_uint<2> tp,
                         ap_uint<2> mid,
                         ap_uint<2> bottom,
                         bool flag) {
    uchar_t step = XF_PIXELDEPTH(DEPTH_DST);
    uint16_t max_loop = XF_WORDDEPTH(WORDWIDTH_DST);
    uchar_t buf_size = XF_NPIXPERCYCLE(NPC) + 2;
    uint16_t col = 0, i = 0, j = 0;
    ap_uint<3> p;
// clang-format off
    #pragma HLS INLINE off
// clang-format on
// column loop up to the end of the row
Col_Loop:
    for (col = 0; col < (img_width); col++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
        #pragma HLS PIPELINE
    // clang-format on

    // reading the data from the stream
    Plane_Loop3:
        for (p = 0; p < NOS_SRC; p++) {
// clang-format off
            #pragma HLS UNROLL
            // clang-format on
            XF_SNAME(WORDWIDTH_SRC) in_data = 0;
            if (flag) {
                in_data = _src_strm[p].read();
                buf[p][row_ind][col] = in_data;
            }

            // extracting the data from the input buffer to the process buffer
            xfExtractPixels<NPC, WORDWIDTH_SRC, DEPTH_SRC>(&src_buf0[p][2], buf[p][tp][col], 0);
            xfExtractPixels<NPC, WORDWIDTH_SRC, DEPTH_SRC>(&src_buf1[p][2], buf[p][mid][col], 0);
            xfExtractPixels<NPC, WORDWIDTH_SRC, DEPTH_SRC>(&src_buf2[p][2], in_data, 0);
        }
        // function to compute the gradients
        xFHOGgradientCompute<NPC, DEPTH_SRC, DEPTH_DST, NOS_SRC>(GradientValuesX, GradientValuesY, src_buf0, src_buf1,
                                                                 src_buf2);

        if (col == 0) {
            j = 1;
        data_pack_loop1:
            for (i = 0; i < (max_loop - step); i = i + step) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=PIX_COUNT max=PIX_COUNT
                #pragma HLS UNROLL
                // clang-format on
                P0.range(i + (step - 1), i) = GradientValuesX[j];
                P1.range(i + (step - 1), i) = GradientValuesY[j++];
            }
        } else {
            P0.range((max_loop - 1), (max_loop - step)) = GradientValuesX[0];
            P1.range((max_loop - 1), (max_loop - step)) = GradientValuesY[0];
            _gradx_strm.write(P0);
            _grady_strm.write(P1);

            j = 1;
        data_pack_loop2:
            for (i = 0; i < (max_loop - step); i = i + step) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=PIX_COUNT max=PIX_COUNT
                #pragma HLS UNROLL
                // clang-format on
                P0.range(i + (step - 1), i) = GradientValuesX[j];
                P1.range(i + (step - 1), i) = GradientValuesY[j++];
            }
        }

    // copy the last two pixel data to the next iteration
    Plane_Loop4:
        for (p = 0; p < NOS_SRC; p++) {
// clang-format off
            #pragma HLS UNROLL
            // clang-format on
            src_buf0[p][0] = src_buf0[p][buf_size - 2];
            src_buf0[p][1] = src_buf0[p][buf_size - 1];

            src_buf1[p][0] = src_buf1[p][buf_size - 2];
            src_buf1[p][1] = src_buf1[p][buf_size - 1];

            src_buf2[p][0] = src_buf2[p][buf_size - 2];
            src_buf2[p][1] = src_buf2[p][buf_size - 1];
        }
    } // Col_Loop
}

template <int ROWS,
          int COLS,
          int DEPTH_SRC,
          int DEPTH_DST,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int NOS_SRC,
          int TC,
          int PIX_COUNT,
          bool USE_URAM>
void xFHOGgradientsKernel(hls::stream<XF_SNAME(WORDWIDTH_SRC)> _src_strm[NOS_SRC],
                          hls::stream<XF_SNAME(WORDWIDTH_DST)>& _gradx_strm,
                          hls::stream<XF_SNAME(WORDWIDTH_DST)>& _grady_strm,
                          uint16_t _height,
                          uint16_t _width) {
    // row_index for circular buffer organization
    uint16_t row_ind;
    ap_uint<3> p;
    ap_uint<2> tp, mid, bottom;
    uchar_t step = XF_PIXELDEPTH(DEPTH_DST);
    uint16_t max_loop = XF_WORDDEPTH(WORDWIDTH_DST);
    uchar_t buf_size = XF_NPIXPERCYCLE(NPC) + 2;
    uint16_t col, j, row, i;

    // output gradient buffers; gradient-x and gradient-y
    XF_PTNAME(DEPTH_DST) GradientValuesX[XF_NPIXPERCYCLE(NPC)];
    XF_PTNAME(DEPTH_DST) GradientValuesY[XF_NPIXPERCYCLE(NPC)];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=GradientValuesX complete dim=1
    #pragma HLS ARRAY_PARTITION variable=GradientValuesY complete dim=1
    // clang-format on

    // temporary buffer to hold the input data for computation
    XF_PTNAME(DEPTH_SRC)
    src_buf0[NOS_SRC][XF_NPIXPERCYCLE(NPC) + 2], src_buf1[NOS_SRC][XF_NPIXPERCYCLE(NPC) + 2],
        src_buf2[NOS_SRC][XF_NPIXPERCYCLE(NPC) + 2];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=src_buf0 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=src_buf1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=src_buf2 complete dim=0
    // clang-format on

    // used to temporarily hold the output data before pushing into the stream
    XF_SNAME(WORDWIDTH_DST) P0, P1;

    // Line buffer to hold image data
    XF_SNAME(WORDWIDTH_SRC) buf[NOS_SRC][3][(COLS >> XF_BITSHIFT(NPC))];

    if (USE_URAM) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=buf complete dim=1
        #pragma HLS RESOURCE variable=buf core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=buf cyclic factor=3 dim=2
        // clang-format on
    } else {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=buf complete dim=1
        #pragma HLS ARRAY_PARTITION variable=buf complete dim=2
        // clang-format on
    }

    row_ind = 1;

// reading the complete first line to the input buffer
Clear_Row_Read_Buf_Loop:
    for (col = 0; col < (_width); col++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
        #pragma HLS PIPELINE
    // clang-format on

    Plane_Loop1:
        for (p = 0; p < NOS_SRC; p++) {
            buf[p][0][col] = 0;
            buf[p][row_ind][col] = _src_strm[p].read(); // Read data
        }
    }
    row_ind++;

// process loop up to the end of the image
Row_Loop:
    for (row = 1; row < (_height); row++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        // clang-format on

        // updating the row index for the circular buffer organization
        if (row_ind == 2) {
            tp = 0;
            mid = 1;
            bottom = 2;
        } else if (row_ind == 0) {
            tp = 1;
            mid = 2;
            bottom = 0;
        } else if (row_ind == 1) {
            tp = 2;
            mid = 0;
            bottom = 1;
        }

    // padding the left border with zero
    Plane_Loop2:
        for (p = 0; p < NOS_SRC; p++) {
// clang-format off
            #pragma HLS UNROLL
            // clang-format on
            src_buf0[p][0] = src_buf0[p][1] = 0;
            src_buf1[p][0] = src_buf1[p][1] = 0;
            src_buf2[p][0] = src_buf2[p][1] = 0;
        }

        P0 = P1 = 0;

        // compute the gradient for the data in the Source buffer
        xFHOGcomputeColGrad<ROWS, COLS, DEPTH_SRC, DEPTH_DST, NPC, WORDWIDTH_SRC, WORDWIDTH_DST, NOS_SRC, TC,
                            PIX_COUNT>(_src_strm, _gradx_strm, _grady_strm, buf, src_buf0, src_buf1, src_buf2,
                                       GradientValuesX, GradientValuesY, P0, P1, _width, row_ind, tp, mid, bottom,
                                       true);

        if (row) {
        // copy the last two pixel data to the next iteration
        Plane_Loop4:
            for (p = 0; p < NOS_SRC; p++) {
// clang-format off
                #pragma HLS UNROLL
                // clang-format on
                src_buf0[p][2] = 0;
                src_buf1[p][2] = 0;
                src_buf2[p][2] = 0;
            }

            xFHOGgradientCompute<NPC, DEPTH_SRC, DEPTH_DST, NOS_SRC>(GradientValuesX, GradientValuesY, src_buf0,
                                                                     src_buf1, src_buf2);

            P0.range((max_loop - 1), (max_loop - step)) = GradientValuesX[0];
            P1.range((max_loop - 1), (max_loop - step)) = GradientValuesY[0];
            _gradx_strm.write(P0);
            _grady_strm.write(P1);
        }
        row_ind++;
        if (row_ind == 3) {
            row_ind = 0;
        }
    } // Row_Loop

    // compute indexes for the input buffer
    if (row_ind == 3) {
        row_ind = 0;
    }
    if (row_ind == 2) {
        tp = 0;
        mid = 1;
        bottom = 2;
    } else if (row_ind == 0) {
        tp = 1;
        mid = 2;
        bottom = 0;
    } else if (row_ind == 1) {
        tp = 2;
        mid = 0;
        bottom = 1;
    }

Plane_Loop6:
    for (p = 0; p < NOS_SRC; p++) {
// clang-format off
        #pragma HLS UNROLL
        // clang-format on

        src_buf0[p][0] = src_buf0[p][1] = 0;
        src_buf1[p][0] = src_buf1[p][1] = 0;
        src_buf2[p][0] = src_buf2[p][1] = 0;
    }

Clear_Row_Loop1:
    for (col = 0; col < (_width); col++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
        #pragma HLS PIPELINE
    // clang-format on

    Plane_Loop7:
        for (p = 0; p < NOS_SRC; p++) {
// clang-format off
            #pragma HLS UNROLL
            // clang-format on
            buf[p][bottom][col] = 0;
        }
    }

    // compute the gradient for the data in the Source buffer
    xFHOGcomputeColGrad<ROWS, COLS, DEPTH_SRC, DEPTH_DST, NPC, WORDWIDTH_SRC, WORDWIDTH_DST, NOS_SRC, TC, PIX_COUNT>(
        _src_strm, _gradx_strm, _grady_strm, buf, src_buf0, src_buf1, src_buf2, GradientValuesX, GradientValuesY, P0,
        P1, _width, row_ind, tp, mid, bottom, false);

Plane_Loop5:
    for (p = 0; p < NOS_SRC; p++) {
// clang-format off
        #pragma HLS UNROLL
        // clang-format on
        src_buf0[p][2] = 0;
        src_buf1[p][2] = 0;
        src_buf2[p][2] = 0;
    }

    xFHOGgradientCompute<NPC, DEPTH_SRC, DEPTH_DST, NOS_SRC>(GradientValuesX, GradientValuesY, src_buf0, src_buf1,
                                                             src_buf2);

    P0.range((max_loop - 1), (max_loop - step)) = GradientValuesX[0];
    P1.range((max_loop - 1), (max_loop - step)) = GradientValuesY[0];
    _gradx_strm.write(P0);
    _grady_strm.write(P1);
}
// xFHOGGradientComputation

/**************************************************************************
 * xFHOGgradients : Wrapper function which calls the kernel function
 * 				depending upon the configurations.
 **************************************************************************/
template <int ROWS,
          int COLS,
          int DEPTH_SRC,
          int DEPTH_DST,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int NOS_SRC,
          bool USE_URAM>
void xFHOGgradients(hls::stream<XF_SNAME(WORDWIDTH_SRC)> _src[NOS_SRC],
                    hls::stream<XF_SNAME(WORDWIDTH_DST)>& _gradx,
                    hls::stream<XF_SNAME(WORDWIDTH_DST)>& _grady,
                    int _border_type,
                    uint16_t _height,
                    uint16_t _width) {
#ifndef __SYNTHESIS__
    assert(((DEPTH_SRC == XF_8UP)) && " Input image must be of type XF_8UP");

    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && "NPC must be XF_NPPC1 or XF_NPPC8");

    assert(((WORDWIDTH_SRC == XF_8UW) || (WORDWIDTH_SRC == XF_64UW)) && "WORDWIDTH must be XF_8UW or XF_64UW");

    assert(((DEPTH_DST == XF_9SP)) && " Input image must be of type XF_9SP");

    assert(((WORDWIDTH_DST == XF_9UW) || (WORDWIDTH_DST == XF_72UW)) && "WORDWIDTH must be XF_9UW or XF_72UW");

    assert((_border_type == XF_BORDER_CONSTANT) && "Border type must be XF_BORDER_CONSTANT ");

    assert(((NOS_SRC == XF_GRAY) || (NOS_SRC == XF_RGB)) && "input_image_type must be either XF_GRAY or XF_RGB");
#endif

    xFHOGgradientsKernel<ROWS, COLS, DEPTH_SRC, DEPTH_DST, NPC, WORDWIDTH_SRC, WORDWIDTH_DST, NOS_SRC,
                         (COLS >> XF_BITSHIFT(NPC)), (XF_NPIXPERCYCLE(NPC)), USE_URAM>(_src, _gradx, _grady, _height,
                                                                                       _width);
}
// xFHOGgradients

#endif // _XF_HOG_DESCRIPTOR_GRADIENTS_HPP_
