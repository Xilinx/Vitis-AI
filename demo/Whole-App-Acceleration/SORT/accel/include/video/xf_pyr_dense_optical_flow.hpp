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

#ifndef __XF_PYR_DENSE_OPTICAL_FLOW__
#define __XF_PYR_DENSE_OPTICAL_FLOW__

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "xf_pyr_dense_optical_flow_config_types.h"
#include "xf_pyr_dense_optical_flow_scale.hpp"
#include "xf_pyr_dense_optical_flow_median_blur.hpp"
#include "xf_pyr_dense_optical_flow_find_gradients.hpp"
#include "xf_pyr_dense_optical_flow_oflow_process.hpp"
#include "math.h"
template <unsigned short MAXHEIGHT, unsigned short MAXWIDTH, int FLOW_WIDTH, int FLOW_INT>
void stitch_stream_fixed_int(hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> >& in_stream1,
                             hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> >& in_stream2,
                             xf::cv::Mat<XF_32UC1, MAXHEIGHT, MAXWIDTH, XF_NPPC1>& sitched_stream,
                             unsigned int rows,
                             unsigned int cols,
                             unsigned int level) {
// clang-format off
    #pragma HLS inline off
// clang-format on

#if DEBUG
    char name[200];
    sprintf(name, "postscaleU_hw%d.txt", level);
    FILE* fpU = fopen(name, "w");
    sprintf(name, "postscaleV_hw%d.txt", level);
    FILE* fpV = fopen(name, "w");
#endif
    for (ap_uint<16> i = 0; i < rows; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAXHEIGHT
        // clang-format on
        for (ap_uint<16> j = 0; j < cols; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=MAXWIDTH
            #pragma HLS pipeline ii=1
            #pragma HLS LOOP_FLATTEN OFF
            // clang-format on
            ap_fixed<FLOW_WIDTH, FLOW_INT> reg1 = 0;
            ap_fixed<FLOW_WIDTH, FLOW_INT> reg2 = 0;

            reg1 = in_stream1.read();
            reg2 = in_stream2.read();

            short* shortconv1 = (short*)&reg1;
            short* shortconv2 = (short*)&reg2;
#if DEBUG
            fprintf(fpU, "%f ", float(reg1));
            fprintf(fpV, "%f ", float(reg2));
#endif
            int convert = (*shortconv2);
            unsigned int tempstore = convert & 0x0000FFFF;
            tempstore = (*shortconv1 << 16) | tempstore;

            sitched_stream.write(i * cols + j, tempstore);
        } // end j loop
#if DEBUG
        fprintf(fpU, "\n");
        fprintf(fpV, "\n");
#endif
    } // end i loop
#if DEBUG
    fclose(fpU);
    fclose(fpV);
#endif
} // end split_stream()
template <unsigned short MAXHEIGHT, unsigned short MAXWIDTH, int FLOW_WIDTH, int FLOW_INT>
void split_stream_int_fixed(xf::cv::Mat<XF_32UC1, MAXHEIGHT, MAXWIDTH, XF_NPPC1>& instream,
                            hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> >& out_stream1,
                            hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> >& out_stream2,
                            unsigned int rows,
                            unsigned int cols,
                            unsigned int level) {
// clang-format off
    #pragma HLS inline off
// clang-format on

#if DEBUG
    char name[200];
    sprintf(name, "prescaleU_hw%d.txt", level);
    FILE* fpU = fopen(name, "w");
    sprintf(name, "prescaleV_hw%d.txt", level);
    FILE* fpV = fopen(name, "w");
#endif
    for (ap_uint<16> i = 0; i < rows; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAXHEIGHT
        // clang-format on
        for (ap_uint<16> j = 0; j < cols; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=MAXWIDTH
            #pragma HLS pipeline ii=1
            #pragma HLS LOOP_FLATTEN OFF
            // clang-format on

            unsigned int tempcopy = instream.read(i * cols + j);

            short splittemp1 = (tempcopy >> 16);
            short splittemp2 = (0x0000FFFF & tempcopy);

            ap_fixed<FLOW_WIDTH, FLOW_INT>* uflow = (ap_fixed<FLOW_WIDTH, FLOW_INT>*)&splittemp1;
            ap_fixed<FLOW_WIDTH, FLOW_INT>* vflow = (ap_fixed<FLOW_WIDTH, FLOW_INT>*)&splittemp2;

            ap_fixed<FLOW_WIDTH, FLOW_INT> u = *uflow;
            ap_fixed<FLOW_WIDTH, FLOW_INT> v = *vflow;

            out_stream1.write(u);
            out_stream2.write(v);
#if DEBUG
            fprintf(fpU, "%12.8f ", float(u));
            fprintf(fpV, "%12.8f ", float(v));
#endif
        } // end j loop
#if DEBUG
        fprintf(fpU, "\n");
        fprintf(fpV, "\n");
#endif
    } // end i loop
#if DEBUG
    fclose(fpU);
    fclose(fpV);
#endif
} // end split_stream()

template <unsigned short MAXHEIGHT,
          unsigned short MAXWIDTH,
          int SIXIY_WIDTH,
          int SIXIY_INT,
          int SIXYIT_WIDTH,
          int SIXYIT_INT,
          int FLOW_WIDTH,
          int FLOW_INT,
          int DET_WIDTH,
          int DET_INT,
          int DIVBY_WIDTH,
          int DIVBY_INT,
          int FLCMP_WIDTH,
          int FLCMP_INT,
          int WINSIZE>
void find_flow(hls::stream<ap_fixed<SIXIY_WIDTH, SIXIY_INT> >& strmSigmaIx2,
               hls::stream<ap_fixed<SIXIY_WIDTH, SIXIY_INT> >& strmSigmaIy2,
               hls::stream<ap_fixed<SIXIY_WIDTH, SIXIY_INT> >& strmSigmaIxIy,
               hls::stream<ap_fixed<SIXYIT_WIDTH, SIXYIT_INT> >& strmSigmaItIx,
               hls::stream<ap_fixed<SIXYIT_WIDTH, SIXYIT_INT> >& strmSigmaItIy,
               hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> >& streamflowU_in,
               hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> >& streamflowV_in,
               hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> >& strmFlowU,
               hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> >& strmFlowV,
               hls::stream<bool>& flagU,
               hls::stream<bool>& flagV,
               unsigned int rows,
               unsigned int cols,
               unsigned int level,
               bool scale_up_flag,
               ap_uint<1> init_flag) {
// clang-format off
    #pragma HLS inline off
// clang-format on

#if DEBUG
    char filename0[200];
    char filename1[200];
    char filename2[200];
    sprintf(filename0, "flU_hw%d.txt", level);
    sprintf(filename1, "flV_hw%d.txt", level);
    sprintf(filename2, "det_hw%d.txt", level);
    FILE* fpdet = fopen(filename2, "w");
    FILE* fpglxup = fopen(filename0, "w");
    FILE* fpglyup = fopen(filename1, "w");
#endif
    int count = 0;
    for (ap_uint<16> i = 0; i < rows; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAXHEIGHT
        // clang-format on
        for (ap_uint<16> j = 0; j < cols; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=MAXWIDTH
            #pragma HLS pipeline ii=1
            #pragma HLS LOOP_FLATTEN OFF
            // clang-format on
            ap_fixed<FLOW_WIDTH, FLOW_INT> flowU, flowV;
            ap_fixed<SIXIY_WIDTH, SIXIY_INT> sigmaIx2 = strmSigmaIx2.read();
            ap_fixed<SIXIY_WIDTH, SIXIY_INT> sigmaIy2 = strmSigmaIy2.read();
            ap_fixed<SIXIY_WIDTH, SIXIY_INT> sigmaIxIy = strmSigmaIxIy.read();
            ap_fixed<SIXYIT_WIDTH, SIXYIT_INT> sigmaItIx = strmSigmaItIx.read();
            ap_fixed<SIXYIT_WIDTH, SIXYIT_INT> sigmaItIy = strmSigmaItIy.read();

            ap_fixed<((SIXIY_WIDTH + 1) << 1) + 3, ((SIXIY_INT + 1) << 1) + 3> S12sq = sigmaIxIy * sigmaIxIy;
            ap_fixed<DET_WIDTH, DET_INT> det = (sigmaIx2 * sigmaIy2 - S12sq);
            ap_fixed<SIXIY_WIDTH + 1, SIXIY_INT + 1> S1122 = (sigmaIx2 + sigmaIy2);
            ap_fixed<(SIXIY_WIDTH + 1) << 1, (SIXIY_INT + 1) << 1> S1122sq = S1122 * S1122;
            S12sq = (S12sq << 2) + S1122sq; // multiply by 4
            static half div_by_eig = 1 / (2.0 * WINSIZE * WINSIZE);
            float S1122_h = S1122;
            float S12sq_h = S12sq;
            float eig_comp = S1122_h - sqrt(S12sq_h);
            // float  eig_comp =  (((A11 + A22) - sqrt( ((A11 + A22)*(A11 + A22))
            // + 4.0*A12*A12))/(2.0*WINSIZE*WINSIZE));
            float eig_comp2 = eig_comp * div_by_eig;
            float eig_comp3 = (eig_comp2 < 0) ? -eig_comp2 : eig_comp2;

            bool tflagu;
            bool tflagv;
            if ((det == 0) || (eig_comp3 < 0.025)) {
                flowU = (ap_fixed<FLCMP_WIDTH, FLCMP_INT>)0;
                flowV = (ap_fixed<FLCMP_WIDTH, FLCMP_INT>)0;
                count++;
                tflagu = 0;
                tflagv = 0;
            } else {
                ap_fixed<DIVBY_WIDTH, DIVBY_INT> divideBy;
                ap_fixed<FLCMP_WIDTH, FLCMP_INT> tempU;
                ap_fixed<FLCMP_WIDTH, FLCMP_INT> tempV;
                divideBy = (ap_fixed<DIVBY_WIDTH, DIVBY_INT>)(1.0) / ((ap_fixed<DET_WIDTH, DET_INT>)det);
                tempU = ((ap_fixed<SIXYIT_WIDTH + SIXYIT_WIDTH, SIXYIT_INT + SIXYIT_INT>)sigmaIy2 * sigmaItIx -
                         (ap_fixed<SIXYIT_WIDTH + SIXYIT_WIDTH, SIXYIT_INT + SIXYIT_INT>)sigmaIxIy * sigmaItIy) *
                        (divideBy);
                tempV = ((ap_fixed<SIXYIT_WIDTH + SIXYIT_WIDTH, SIXYIT_INT + SIXYIT_INT>)sigmaIx2 * sigmaItIy -
                         (ap_fixed<SIXYIT_WIDTH + SIXYIT_WIDTH, SIXYIT_INT + SIXYIT_INT>)sigmaIxIy * sigmaItIx) *
                        (divideBy);
                flowU = ap_fixed<FLOW_WIDTH, FLOW_INT>(tempU);
                flowV = ap_fixed<FLOW_WIDTH, FLOW_INT>(tempV);
                tflagu = 1;
                tflagv = 1;
            }
            if (init_flag == (ap_uint<1>)0) {
                flowU += ap_fixed<FLOW_WIDTH, FLOW_INT>(streamflowU_in.read());
                flowV += ap_fixed<FLOW_WIDTH, FLOW_INT>(streamflowV_in.read());
            } else {
                ap_fixed<FLOW_WIDTH, FLOW_INT> flow_dummyU = ap_fixed<FLOW_WIDTH, FLOW_INT>(streamflowU_in.read());
                ap_fixed<FLOW_WIDTH, FLOW_INT> flow_dummyV = ap_fixed<FLOW_WIDTH, FLOW_INT>(streamflowV_in.read());
            }
            flagU.write(tflagu);
            flagV.write(tflagv);
#if DEBUG
            fprintf(fpdet, "%12.4f ", float(det));
            fprintf(fpglxup, "%12.8f ", float(flowU));
            fprintf(fpglyup, "%12.8f ", float(flowV));
#endif
            strmFlowU.write((ap_fixed<FLOW_WIDTH, FLOW_INT>)flowU);
            strmFlowV.write((ap_fixed<FLOW_WIDTH, FLOW_INT>)flowV);
        }
#if DEBUG
        fprintf(fpdet, "\n");
        fprintf(fpglxup, "\n");
        fprintf(fpglyup, "\n");
#endif
    }

#if DEBUG
    fclose(fpdet);
    fclose(fpglxup);
    fclose(fpglyup);
#endif
} // end find_flow()

template <unsigned short MAXHEIGHT,
          unsigned short MAXWIDTH,
          int NUM_PYR_LEVELS,
          int NUM_LINES,
          int WINSIZE,
          int FLOW_WIDTH,
          int FLOW_INT,
          bool USE_URAM>
void xFLKOpticalFlowDenseKernel(xf::cv::Mat<XF_8UC1, MAXHEIGHT, MAXWIDTH, XF_NPPC1>& currImg,
                                xf::cv::Mat<XF_8UC1, MAXHEIGHT, MAXWIDTH, XF_NPPC1>& nextImg,
                                xf::cv::Mat<XF_32UC1, MAXHEIGHT, MAXWIDTH, XF_NPPC1>& strmFlowin,
                                xf::cv::Mat<XF_32UC1, MAXHEIGHT, MAXWIDTH, XF_NPPC1>& strmFlow,
                                const unsigned int rows,
                                const unsigned int cols,
                                const unsigned int prev_rows,
                                const unsigned int prev_cols,
                                const int level,
                                const bool scale_up_flag,
                                float scale_in,
                                ap_uint<1> init_flag) {
    const int WINDOW_SIZE = WINDOW_SIZE_FL;
    const int RMAPPX_WIDTH = TYPE_RMAPPX_WIDTH;
    const int RMAPPX_INT = TYPE_RMAPPX_INT;
    const int SCALE_WIDTH = TYPE_SCALE_WIDTH;
    const int SCALE_INT = TYPE_SCALE_INT;
    const int IT_WIDTH = TYPE_IT_WIDTH;
    const int IT_INT = TYPE_IT_INT;
    const int SIXIY_WIDTH = TYPE_SIXIY_WIDTH;
    const int SIXIY_INT = TYPE_SIXIY_INT;
    const int SIXYIT_WIDTH = TYPE_SIXYIT_WIDTH;
    const int SIXYIT_INT = TYPE_SIXYIT_INT;
    const int DET_WIDTH = TYPE_DET_WIDTH;
    const int DET_INT = TYPE_DET_INT;
    const int DIVBY_WIDTH = TYPE_DIVBY_WIDTH;
    const int DIVBY_INT = TYPE_DIVBY_INT;
    const int FLCMP_WIDTH = TYPE_FLCMP_WIDTH;
    const int FLCMP_INT = TYPE_FLCMP_INT;
    const int SCCMP_WIDTH = FLOW_WIDTH + SCALE_WIDTH + 12;
    const int SCCMP_INT = FLOW_INT + 12;
    const int ITCMP_WIDTH = FLOW_WIDTH + 12 + 4;
    const int ITCMP_INT = FLOW_INT + 12;

// clang-format off
    #pragma HLS dataflow
    // clang-format on
    hls::stream<ap_int<9> > strmIx("Ix"), strmIy("Iy");
    hls::stream<ap_fixed<SIXIY_WIDTH, SIXIY_INT> > sigmaIx2("sigmaIx2"), sigmaIy2("sigmaIy2");
    hls::stream<ap_fixed<SIXYIT_WIDTH, SIXYIT_INT> > sigmaIxIt("sigmaIxIt"), sigmaIyIt("sigmaIyIt");
    hls::stream<ap_fixed<SIXIY_WIDTH, SIXIY_INT> > sigmaIxIy("sigmaIxIy");
    hls::stream<ap_fixed<IT_WIDTH, IT_INT> > strmIt_float("It");
    hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> > strmFlowU_fil("U_median_in"), strmFlowV_fil("V_median_in");
    hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> > strmFlowU_fil_out("U_median_out"), strmFlowV_fil_out("V_median_out");
    hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> > strmFlowU_in1("U_in1"), strmFlowV_in1("V_in1");
    hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> > strmFlowU_split("Flow_stream_splitU"),
        strmFlowV_split("Flow_stream_splitV");
    hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> > strmFlowU_scaled("U_in_scaled"), strmFlowV_scaled("V_in_scaled");
    hls::stream<bool> flagU("compute flowU flag"), flagV("compute flowV flag");
// Ix, Iy, and It will be consumed at the same time and without any d
// Giving them a 64 depth buffer just in case.
// clang-format off
    #pragma HLS STREAM variable=&strmIx            depth=64   dim=1
    #pragma HLS STREAM variable=&strmIy            depth=64   dim=1
    #pragma HLS STREAM variable=&strmIt_float      depth=64   dim=1
    #pragma HLS STREAM variable=&flagU             depth=5000 dim=1
    #pragma HLS STREAM variable=&flagV             depth=5000 dim=1
// clang-format on

// Flow U and V _in1 will be consumed at most 17*Width cycles after the _scaled.
// 1920*17= 32640 (17 is arrived at by trial and experiment)
// This ideally has to be taken care of by the data flow module.
// clang-format off
    #pragma HLS STREAM variable=&strmFlowU_in1     depth=32640 dim=1
    #pragma HLS STREAM variable=&strmFlowV_in1     depth=32640 dim=1
// clang-format on
#ifndef __SYNTHESIS__
    assert(rows <= MAXHEIGHT);
    assert(cols <= MAXWIDTH);
#endif
    // splitting the input flow streams to U and V to scale them up whenever scale up is enabled
    split_stream_int_fixed<MAXHEIGHT, MAXWIDTH, FLOW_WIDTH, FLOW_INT>(strmFlowin, strmFlowU_split, strmFlowV_split,
                                                                      prev_rows, prev_cols, level);

    // scaling up U and V streams whenever scaleup is enabled
    scale_up<MAXHEIGHT, MAXWIDTH, FLOW_WIDTH, FLOW_INT, SCCMP_WIDTH, SCCMP_INT, RMAPPX_WIDTH, RMAPPX_INT, SCALE_WIDTH,
             SCALE_INT, USE_URAM>(strmFlowU_split, strmFlowU_scaled, prev_rows, prev_cols, rows, cols, 2, scale_up_flag,
                                  scale_in);
    scale_up<MAXHEIGHT, MAXWIDTH, FLOW_WIDTH, FLOW_INT, SCCMP_WIDTH, SCCMP_INT, RMAPPX_WIDTH, RMAPPX_INT, SCALE_WIDTH,
             SCALE_INT, USE_URAM>(strmFlowV_split, strmFlowV_scaled, prev_rows, prev_cols, rows, cols, 2, scale_up_flag,
                                  scale_in);

    // Finding the Temporal and space gradients for the input set of images
    findGradients<MAXHEIGHT, MAXWIDTH, NUM_PYR_LEVELS, NUM_LINES, WINSIZE, IT_WIDTH, IT_INT, ITCMP_WIDTH, ITCMP_INT,
                  FLOW_WIDTH, FLOW_INT, RMAPPX_WIDTH, RMAPPX_INT, USE_URAM>(
        currImg, nextImg, strmIt_float, strmIx, strmIy, rows, cols, strmFlowU_scaled, strmFlowV_scaled, strmFlowU_in1,
        strmFlowV_in1, level);

    // finding the hessian matrix
    find_G_and_b_matrix<MAXHEIGHT, MAXWIDTH, WINSIZE, IT_WIDTH, IT_INT, SIXIY_WIDTH, SIXIY_INT, SIXYIT_WIDTH,
                        SIXYIT_INT, USE_URAM>(strmIx, strmIy, strmIt_float, sigmaIx2, sigmaIy2, sigmaIxIy, sigmaIxIt,
                                              sigmaIyIt, rows, cols, level);

    // computing the the optical flow

    find_flow<MAXHEIGHT, MAXWIDTH, SIXIY_WIDTH, SIXIY_INT, SIXYIT_WIDTH, SIXYIT_INT, FLOW_WIDTH, FLOW_INT, DET_WIDTH,
              DET_INT, DIVBY_WIDTH, DIVBY_INT, FLCMP_WIDTH, FLCMP_INT, WINSIZE>(
        sigmaIx2, sigmaIy2, sigmaIxIy, sigmaIxIt, sigmaIyIt, strmFlowU_in1, strmFlowV_in1, strmFlowU_fil, strmFlowV_fil,
        flagU, flagV, rows, cols, level, scale_up_flag, init_flag);

    // filtering the flow vectors using median blur
    auMedianBlur<MAXHEIGHT, MAXWIDTH, 0, 0, 0, 0, WINDOW_SIZE, WINDOW_SIZE * WINDOW_SIZE, FLOW_WIDTH, FLOW_INT,
                 USE_URAM>(strmFlowU_fil, strmFlowU_fil_out, flagU, WINDOW_SIZE, 1, rows, cols);
    auMedianBlur<MAXHEIGHT, MAXWIDTH, 0, 0, 0, 0, WINDOW_SIZE, WINDOW_SIZE * WINDOW_SIZE, FLOW_WIDTH, FLOW_INT,
                 USE_URAM>(strmFlowV_fil, strmFlowV_fil_out, flagV, WINDOW_SIZE, 1, rows, cols);

    // stitching the U and V flow streams to a single flow stream
    stitch_stream_fixed_int<MAXHEIGHT, MAXWIDTH, FLOW_WIDTH, FLOW_INT>(strmFlowU_fil_out, strmFlowV_fil_out, strmFlow,
                                                                       rows, cols, level);
}
#endif
