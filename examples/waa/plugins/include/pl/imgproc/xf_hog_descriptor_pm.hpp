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

#ifndef _XF_HOG_DESCRIPTOR_PM_HPP_
#define _XF_HOG_DESCRIPTOR_PM_HPP_

#ifndef __cplusplus
#error C++ is needed to include this header
#endif

#include "core/xf_math.h"

// to convert the radians value to degrees
#define XF_NORM_FACTOR 58671 // (180/PI)  in  Q6.10

/***********************************************************************************************
 * xFHOGPhaseMagnitudeKernel : The Gradient Phase and Gradient magnitude Computation Kernel.
 *		 This kernel takes two gradients in XF_9SP depth and computes the angles and magnitude
 *		 for each pixel and store this in a XF_9UP images.
 *
 *  The Input arguments are _gradx_stream, _grady_stream
 *  _gradx_stream --> Gradient X data from the gradient computation function of
 *  		depth XF_9SP.
 *  _grady_stream --> Gradient Y data from the gradient computation function of
 *  		depth XF_9SP.
 *  _phase_stream --> phase computed image of depth XF_16UP.
 *  _mag_stream   --> magnitude computed image of depth XF_16UP.
 *
 *  Depending on NPC, 1 or 8 pixels are read and gradient values are calculated.
 **********************************************************************************************/
template <int ROWS,
          int COLS,
          int DEPTH_SRC,
          int DEPTH_DST,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int COLS_TRIP>
void xFHOGPhaseMagnitudeKernel(hls::stream<XF_SNAME(WORDWIDTH_SRC)>& _gradx_stream,
                               hls::stream<XF_SNAME(WORDWIDTH_SRC)>& _grady_stream,
                               hls::stream<XF_SNAME(WORDWIDTH_DST)>& _phase_stream,
                               hls::stream<XF_SNAME(WORDWIDTH_DST)>& _mag_stream,
                               uint16_t height,
                               uint16_t width) {
    // Data in the packed format
    XF_SNAME(WORDWIDTH_SRC) grad_x_packed_val, grad_y_packed_val;
    XF_SNAME(WORDWIDTH_DST) phase_packed_val, mag_packed_val;

    // declaring the loop variables
    uint16_t i, j;
    uint16_t k, l;

    // VARIABLES FOR PHASE KERNEL
    // Fixed point format of x and y, x = QM1.N1, y = QM2.N2 // Q9.0 format input
    int M1, N1, M2, N2;
    M1 = XF_PIXELDEPTH(DEPTH_SRC);
    N1 = 0;
    M2 = M1;
    N2 = N1;

rowLoop:
    for (i = 0; i < height; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        #pragma HLS LOOP_FLATTEN OFF
    // clang-format on

    colLoop:
        for (j = 0; j < width; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=COLS_TRIP max=COLS_TRIP
            #pragma HLS PIPELINE
            // clang-format on

            grad_x_packed_val = (XF_SNAME(WORDWIDTH_SRC))(_gradx_stream.read());
            grad_y_packed_val = (XF_SNAME(WORDWIDTH_SRC))(_grady_stream.read());

            uchar_t step_src = XF_PIXELDEPTH(DEPTH_SRC);
            uchar_t step_dst = XF_PIXELDEPTH(DEPTH_DST);
            uint16_t proc_loop_src = XF_PIXELDEPTH(DEPTH_SRC) << XF_BITSHIFT(NPC);
            uint16_t proc_loop_dst = XF_PIXELDEPTH(DEPTH_DST) << XF_BITSHIFT(NPC);

        procLoop:
            for (k = 0, l = 0; k < proc_loop_src, l < proc_loop_dst; k += step_src, l += step_dst) {
// clang-format off
                #pragma HLS UNROLL
                // clang-format on

                XF_PTNAME(DEPTH_SRC)
                g_x = grad_x_packed_val.range(k + (step_src - 1), k); // Get bits from certain range of positions.
                XF_PTNAME(DEPTH_SRC)
                g_y = grad_y_packed_val.range(k + (step_src - 1), k); // Get bits from certain range of positions.

                int16_t g_x_16_p, g_x_16_m, g_y_16_p, g_y_16_m;
                g_x_16_p = g_x_16_m = g_x;
                g_y_16_p = g_y_16_m = g_y;

                /////////////   PHASE COMPUTATION  /////////////
                // output will be in the radians
                int16_t ret = 0;
                int result_temp = 0;

                ret = xf::cv::Atan2LUT8(g_x_16_p, g_y_16_p, M1, N1, M2, N2);

                if ((ret < 0) || ((ret == 0) && (g_y_16_p < 0)))
                    result_temp = ret + XF_PI_FIXED + XF_PI_FIXED;
                else
                    result_temp = ret;

                // converting the radians value to degree
                // instead of removing complete 22 fractional bits, we shift only 15 times, for we have some precision
                // for HoG gradient computation
                XF_PTNAME(DEPTH_DST) result_phase = ((XF_NORM_FACTOR * result_temp) >> 15); // Q9.7 format
                ////////////////////////////////////////////////////

                /////////////   MAGNITUDE COMPUTATION  /////////////
                // absolute difference of the input data
                __HOG_ABS(g_x_16_m);
                __HOG_ABS(g_y_16_m);

                ap_uint17_t gx_sq = (uchar_t)g_x_16_m * (uchar_t)g_x_16_m;
                ap_uint17_t gy_sq = (uchar_t)g_y_16_m * (uchar_t)g_y_16_m;

                // perform square root for the result_tmp
                ap_uint<17> sum_sq_grad = gx_sq + gy_sq;
                ap_ufixed<31, 31, AP_TRN, AP_SAT> tmp_mag = ((ap_uint<31>)sum_sq_grad) << 14;
                XF_PTNAME(DEPTH_DST) result_mag = xFSqrtHOG<16>(tmp_mag); // Q9.7 format
                ////////////////////////////////////////////////////

                // packing the output pixel data
                phase_packed_val.range(l + (step_dst - 1), l) = result_phase;
                mag_packed_val.range(l + (step_dst - 1), l) = result_mag;
            } // end of proc loop

            _phase_stream.write(phase_packed_val);
            _mag_stream.write(mag_packed_val);
        } // end of col loop
    }     // end of row loop
} // end of function

/*******************************************************************
 * xFPhaseMagnitude: This function acts as a wrapper function and
 *		 calls the phaseMagnitude Kernel function.
 *******************************************************************/
template <int ROWS, int COLS, int DEPTH_SRC, int DEPTH_DST, int NPC, int WORDWIDTH_SRC, int WORDWIDTH_DST>
void xFHOGPhaseMagnitude(hls::stream<XF_SNAME(WORDWIDTH_SRC)>& _grad_x,
                         hls::stream<XF_SNAME(WORDWIDTH_SRC)>& _grad_y,
                         hls::stream<XF_SNAME(WORDWIDTH_DST)>& _phase_stream,
                         hls::stream<XF_SNAME(WORDWIDTH_DST)>& _mag_stream,
                         uint16_t height,
                         uint16_t width) {
#ifndef __SYNTHESIS__
    assert((DEPTH_SRC == XF_9SP) && "DEPTH_SRC must be XF_9SP");
    assert((DEPTH_DST == XF_16UP) && "DEPTH_DST must be of type XF_16UP");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8) || (NPC == XF_NPPC16)) &&
           "NPC must be XF_NPPC1, XF_NPPC8 or XF_NPPC16");
    assert(((WORDWIDTH_SRC == XF_9UW) || (WORDWIDTH_SRC == XF_72UW) || (WORDWIDTH_SRC == XF_144UW)) &&
           "WORDWIDTH_SRC must be XF_9UW, XF_72UW or XF_144UW");
    assert(((WORDWIDTH_DST == XF_16UW) || (WORDWIDTH_DST == XF_128UW) || (WORDWIDTH_DST == XF_256UW)) &&
           "WORDWIDTH_DST must be XF_16UW, XF_128UW or XF_256UW");
#endif

    xFHOGPhaseMagnitudeKernel<ROWS, COLS, DEPTH_SRC, DEPTH_DST, NPC, WORDWIDTH_SRC, WORDWIDTH_DST,
                              (COLS >> XF_BITSHIFT(NPC))>(_grad_x, _grad_y, _phase_stream, _mag_stream, height, width);
}

#endif // _XF_HOG_DESCRIPTOR_PM_HPP_
