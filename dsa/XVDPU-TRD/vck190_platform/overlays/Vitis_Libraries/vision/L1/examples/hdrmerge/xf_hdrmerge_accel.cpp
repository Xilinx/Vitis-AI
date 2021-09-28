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

#include "xf_hdrmerge_config.h"

/************************************************************************************
 * Function:    AXIVideo2BayerMat
 * Parameters:  Multiple bayerWindow.getval AXI Stream, User Stream, Image Resolution
 * Return:      None
 * Description: Read data from multiple pixel/clk AXI stream into user defined stream
 ************************************************************************************/
template <int TYPE, int ROWS, int COLS, int NPPC>
void AXIVideo2BayerMat(InVideoStrm_t_e_s& bayer_strm, xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& bayer_mat) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on
    InVideoStrmBus_t_e_s axi;

    const int m_pix_width = XF_PIXELWIDTH(TYPE, NPPC) * XF_NPIXPERCYCLE(NPPC);

    int rows = bayer_mat.rows;
    int cols = bayer_mat.cols >> XF_BITSHIFT(NPPC);
    int idx = 0;

    bool start = false;
    bool last = false;

loop_start_hunt:
    while (!start) {
// clang-format off
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount avg=0 max=0
        // clang-format on

        bayer_strm >> axi;
        start = axi.user.to_bool();
    }

loop_row_axi2mat:
    for (int i = 0; i < rows; i++) {
        last = false;
// clang-format off
#pragma HLS loop_tripcount avg=ROWS max=ROWS
    // clang-format on
    loop_col_zxi2mat:
        for (int j = 0; j < cols; j++) {
// clang-format off
#pragma HLS loop_flatten off
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount avg=COLS/NPPC max=COLS/NPPC
            // clang-format on

            if (start || last) {
                start = false;
            } else {
                bayer_strm >> axi;
            }

            last = axi.last.to_bool();

            bayer_mat.write(idx++, axi.data(m_pix_width - 1, 0));
        }

    loop_last_hunt:
        while (!last) {
// clang-format off
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount avg=0 max=0
            // clang-format on

            bayer_strm >> axi;
            last = axi.last.to_bool();
        }
    }

    return;
}

template <int TYPE, int ROWS, int COLS, int NPPC>
void GRAYMat2AXIvideo(xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& gray_mat, OutVideoStrm_t_e_s& gray_strm) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    OutVideoStrmBus_t_e_s axi;

    int rows = gray_mat.rows;
    int cols = gray_mat.cols >> XF_BITSHIFT(NPPC);
    int idx = 0;

    XF_TNAME(TYPE, NPPC) srcpixel;

    const int m_pix_width = XF_PIXELWIDTH(TYPE, NPPC) * XF_NPIXPERCYCLE(NPPC);

    int depth = XF_DTPIXELDEPTH(TYPE, NPIX);

    bool sof = true; // Indicates start of frame

loop_row_mat2axi:
    for (int i = 0; i < rows; i++) {
// clang-format off
#pragma HLS loop_tripcount avg=ROWS max=ROWS
    // clang-format on
    loop_col_mat2axi:
        for (int j = 0; j < cols; j++) {
// clang-format off
#pragma HLS loop_flatten off
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount avg=COLS/NPPC max=COLS/NPPC
            // clang-format on
            if (sof) {
                axi.user = 1;
            } else {
                axi.user = 0;
            }

            if (j == cols - 1) {
                axi.last = 1;
            } else {
                axi.last = 0;
            }

            axi.data = 0;

            srcpixel = gray_mat.read(idx++);

            for (int npc = 0; npc < NPPC; npc++) {
                for (int rs = 0; rs < 1; rs++) {
                    int start = (rs + npc) * depth;

                    axi.data(start + (depth - 1), start) = srcpixel.range(start + (depth - 1), start);
                }
            }

            axi.keep = -1;
            gray_strm << axi;

            sof = false;
        }
    }

    return;
}
void hdrmerge_accel(InVideoStrm_t_e_s& img_in1,
                    InVideoStrm_t_e_s& img_in2,
                    OutVideoStrm_t_e_s& img_out,
                    int rows,
                    int cols,
                    short wr_hls[NO_EXPS * NPIX * W_B_SIZE]) {
// clang-format off
	#pragma HLS INTERFACE axis port=&img_in1 register
	#pragma HLS INTERFACE axis port=&img_in2 register
	#pragma HLS INTERFACE axis port=&img_out register
	
    #pragma HLS INTERFACE s_axilite port=rows
	#pragma HLS INTERFACE s_axilite port=wr_hls
    #pragma HLS INTERFACE s_axilite port=cols
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPIX> imgInput1(rows, cols);
    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPIX> imgInput2(rows, cols);

    xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPIX> imgOutput(rows, cols);

// clang-format off
#pragma HLS DATAFLOW
    // clang-format on

    AXIVideo2BayerMat<IN_TYPE, HEIGHT, WIDTH, NPIX>(img_in1, imgInput1);
    AXIVideo2BayerMat<IN_TYPE, HEIGHT, WIDTH, NPIX>(img_in2, imgInput2);

    xf::cv::Hdrmerge_bayer<IN_TYPE, OUT_TYPE, HEIGHT, WIDTH, NPIX, NO_EXPS, W_B_SIZE>(imgInput1, imgInput2, imgOutput,
                                                                                      wr_hls);

    GRAYMat2AXIvideo<IN_TYPE, HEIGHT, WIDTH, NPIX>(imgOutput, img_out);
}
