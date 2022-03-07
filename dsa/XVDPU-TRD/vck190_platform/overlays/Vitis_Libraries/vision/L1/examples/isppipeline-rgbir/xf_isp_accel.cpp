/*
 * Copyright 2021 Xilinx, Inc.
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

#include "xf_isp_types.h"
#include <iostream>
#include <fstream>
#include <strstream>

static bool flag = 0;

static uint32_t hist0_awb[3][HIST_SIZE] = {0};
static uint32_t hist1_awb[3][HIST_SIZE] = {0};

static int igain_0[3] = {0};
static int igain_1[3] = {0};

/************************************************************************************
 * Function:    AXIVideo2BayerMat
 * Parameters:  Multiple bayerWindow.getval AXI Stream, User Stream, Image Resolution
 * Return:      None
 * Description: Read data from multiple pixel/clk AXI stream into user defined stream
 ************************************************************************************/
template <int TYPE, int ROWS, int COLS, int NPPC>
void AXIVideo2BayerMat(InVideoStrm_t& bayer_strm, xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& bayer_mat) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on
    InVideoStrmBus_t axi;

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
void ColorMat2AXIvideo(xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& color_mat, OutVideoStrm_t& color_strm) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    OutVideoStrmBus_t axi;

    int rows = color_mat.rows;
    int cols = color_mat.cols >> XF_BITSHIFT(NPPC);
    int idx = 0;

    XF_TNAME(TYPE, NPPC) srcpixel;

    const int m_pix_width = XF_PIXELWIDTH(TYPE, NPPC) * XF_NPIXPERCYCLE(NPPC);

    int depth = XF_DTPIXELDEPTH(XF_LTM_T, XF_NPPC);

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

            srcpixel = color_mat.read(idx++);

            for (int npc = 0; npc < NPPC; npc++) {
                for (int rs = 0; rs < 3; rs++) {
#if XF_AXI_GBR == 1
                    int kmap[3] = {1, 0, 2}; // GBR format
#else
                    int kmap[3] = {0, 1, 2}; // GBR format
#endif

                    int start = (rs + npc * 3) * depth;

                    int start_format = (kmap[rs] + npc * 3) * depth;

                    axi.data(start + (depth - 1), start) = srcpixel.range(start_format + (depth - 1), start_format);
                }
            }

            axi.keep = -1;
            color_strm << axi;

            sof = false;
        }
    }

    return;
}

template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC = 1>
void fifo_copy(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& demosaic_out,
               xf::cv::Mat<DST_T, ROWS, COLS, NPC>& ltm_in,
               unsigned short height,
               unsigned short width) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on
    ap_uint<13> row, col;
    int readindex = 0, writeindex = 0;

    ap_uint<13> img_width = width >> XF_BITSHIFT(NPC);

Row_Loop:
    for (row = 0; row < height; row++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
#pragma HLS LOOP_FLATTEN off
    // clang-format on
    Col_Loop:
        for (col = 0; col < img_width; col++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=COLS/NPC max=COLS/NPC
#pragma HLS pipeline
            // clang-format on
            XF_TNAME(SRC_T, NPC) tmp_src;
            tmp_src = demosaic_out.read(readindex++);
            ltm_in.write(writeindex++, tmp_src);
        }
    }
}
template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC = 1>
void fifo_awb(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& demosaic_out,
              xf::cv::Mat<DST_T, ROWS, COLS, NPC>& ltm_in,
              uint32_t hist0[3][HIST_SIZE],
              uint32_t hist1[3][HIST_SIZE],
              int gain0[3],
              int gain1[3],
              unsigned short height,
              unsigned short width,
              float thresh) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on	
	xf::cv::Mat<XF_DST_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> impop(height, width);

	float inputMin = 0.0f;
    float inputMax = (1 << (XF_DTPIXELDEPTH(XF_SRC_T, XF_NPPC))) - 1; // 65535.0f;
    float outputMin = 0.0f;
    float outputMax = (1 << (XF_DTPIXELDEPTH(XF_SRC_T, XF_NPPC))) - 1; // 65535.0f;
	
	// clang-format off
#pragma HLS DATAFLOW
    // clang-format on
    if (WB_TYPE) {
        xf::cv::AWBhistogram<XF_DST_T, XF_DST_T, XF_HEIGHT, XF_WIDTH, XF_NPPC, WB_TYPE, HIST_SIZE>(
            demosaic_out, impop, hist0, thresh, inputMin, inputMax, outputMin, outputMax);
        xf::cv::AWBNormalization<XF_DST_T, XF_DST_T, XF_HEIGHT, XF_WIDTH, XF_NPPC, WB_TYPE, HIST_SIZE>(
            impop, ltm_in, hist1, thresh, inputMin, inputMax, outputMin, outputMax);
    } else {
        xf::cv::AWBChannelGain<XF_DST_T, XF_DST_T, XF_HEIGHT, XF_WIDTH, XF_NPPC, 0>(demosaic_out, impop, thresh, gain0);
        xf::cv::AWBGainUpdate<XF_DST_T, XF_DST_T, XF_HEIGHT, XF_WIDTH, XF_NPPC, 0>(impop, ltm_in, thresh, gain1);
    }
}

template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC = 1>
void function_awb(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& demosaic_out,
                  xf::cv::Mat<DST_T, ROWS, COLS, NPC>& ltm_in,
                  uint32_t hist0[3][HIST_SIZE],
                  uint32_t hist1[3][HIST_SIZE],
                  int gain0[3],
                  int gain1[3],
                  unsigned short height,
                  unsigned short width,
                  unsigned char mode_reg,
                  float thresh) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    ap_uint<8> mode = (ap_uint<8>)mode_reg;
    ap_uint<1> mode_flg = mode.range(0, 0);

    if (mode_flg) {
        fifo_awb<XF_DST_T, XF_DST_T, XF_HEIGHT, XF_WIDTH, XF_NPPC>(demosaic_out, ltm_in, hist0, hist1, gain0, gain1,
                                                                   height, width, thresh);
    } else {
        fifo_copy<XF_DST_T, XF_DST_T, XF_HEIGHT, XF_WIDTH, XF_NPPC>(demosaic_out, ltm_in, height, width);
    }
}

template <int TYPE, int ROWS, int COLS, int NPPC>
void ColorMat2AXIvideo_yuv(xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& color_mat, OutVideoStrm_t& color_strm) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    OutVideoStrmBus_t axi;

    int rows = color_mat.rows;
    int cols = color_mat.cols >> XF_BITSHIFT(NPPC);
    int idx = 0;

    XF_TNAME(TYPE, NPPC) srcpixel;

    const int m_pix_width = XF_PIXELWIDTH(TYPE, NPPC) * XF_NPIXPERCYCLE(NPPC);

    int depth = XF_DTPIXELDEPTH(TYPE, XF_NPPC);

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

            srcpixel = color_mat.read(idx++);

            for (int npc = 0; npc < NPPC; npc++) {
                for (int rs = 0; rs < 1; rs++) {
                    int start = (rs + npc) * depth;

                    axi.data(start + (depth - 1), start) = srcpixel.range(start + (depth - 1), start);
                }
            }

            axi.keep = -1;
            color_strm << axi;

            sof = false;
        }
    }

    return;
}
void ISPpipeline(unsigned short height,
                 unsigned short width,
                 InVideoStrm_t& in_axis_video,
                 //                 OutVideoStrm_t& out_axis_video,
                 InVideoStrm_t& out_axis_video,
                 OutVideoStrmIR_t& ir_axis_video,
                 char R_IR_C1_wgts[25],
                 char R_IR_C2_wgts[25],
                 char B_at_R_wgts[25],
                 char IR_at_R_wgts[9],
                 char IR_at_B_wgts[9],
                 char sub_wgts[4],
                 uint32_t hist0[3][HIST_SIZE],
                 uint32_t hist1[3][HIST_SIZE],
                 int gain0[3],
                 int gain1[3],
                 uint16_t rgain,
                 uint16_t bgain,
                 unsigned char gamma_lut[256 * 3],
                 unsigned char mode_reg,
                 uint16_t pawb) {
#pragma HLS INLINE OFF

    xf::cv::Mat<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> imgInput(height, width);
    xf::cv::Mat<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> blklevel_out(height, width);
    xf::cv::Mat<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> blklevel_out_copy1(height, width);
    xf::cv::Mat<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC, 3 * XF_WIDTH> blklevel_out_copy2(height, width);
    xf::cv::Mat<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> halfir_out(height, width);
    xf::cv::Mat<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> fullir_out(height, width);
    xf::cv::Mat<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC, 3 * XF_WIDTH> rggbWithIR(height, width);
    xf::cv::Mat<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> fullir_out_copy1(height, width);
    xf::cv::Mat<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> fullir_out_copy2(height, width);
    xf::cv::Mat<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> rggb_out(height, width);
    xf::cv::Mat<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> bpc_out(height, width);
    xf::cv::Mat<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> gain_out(height, width);
    xf::cv::Mat<XF_DST_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> demosaic_out(height, width);
    xf::cv::Mat<XF_DST_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> demoOut_final(height, width);
    xf::cv::Mat<XF_DST_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> impop(height, width);
    xf::cv::Mat<XF_DST_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> ltm_in(height, width);
    xf::cv::Mat<XF_DST_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> lsc_out(height, width);
    xf::cv::Mat<XF_LTM_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> _dst(height, width);
    xf::cv::Mat<XF_LTM_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> aecin(height, width);
    xf::cv::Mat<XF_16UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC> _imgOutput(height, width);

// clang-format off
#pragma HLS DATAFLOW
    // clang-format on

    const int Q_VAL = 1 << (XF_DTPIXELDEPTH(XF_SRC_T, XF_NPPC));

    float thresh = (float)pawb / 256;
    float inputMax = (1 << (XF_DTPIXELDEPTH(XF_SRC_T, XF_NPPC))) - 1; // 65535.0f;

    float mul_fact = (inputMax / (inputMax - BLACK_LEVEL));

    AXIVideo2BayerMat<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC>(in_axis_video, imgInput);

    xf::cv::blackLevelCorrection<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC, 16, 15, 1>(imgInput, blklevel_out, BLACK_LEVEL,
                                                                                    mul_fact);
    xf::cv::duplicateMat(blklevel_out, blklevel_out_copy1, blklevel_out_copy2);
    xf::cv::RGBIR_Demosaic<FILTERSIZE1, FILTERSIZE2, XF_BAYER_PATTERN, XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC,
                           3 * XF_WIDTH, XF_BORDER_CONSTANT, XF_USE_URAM>(blklevel_out_copy1, R_IR_C1_wgts,
                                                                          R_IR_C2_wgts, B_at_R_wgts, IR_at_R_wgts,
                                                                          IR_at_B_wgts, rggbWithIR, halfir_out);
    xf::cv::IR_bilinear<XF_BAYER_PATTERN>(halfir_out, fullir_out);

    xf::cv::duplicateMat(fullir_out, fullir_out_copy1, fullir_out_copy2);
    xf::cv::weightedSub<XF_BAYER_PATTERN, XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC, 3 * XF_WIDTH>(
        sub_wgts, rggbWithIR, fullir_out_copy1, rggb_out);
    xf::cv::gaincontrol<XF_BAYER_PATTERN, XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC>(rggb_out, gain_out, rgain, bgain);
    xf::cv::demosaicing<XF_BAYER_PATTERN, XF_SRC_T, XF_DST_T, XF_HEIGHT, XF_WIDTH, XF_NPPC, 0>(gain_out, demosaic_out);

    xf::cv::copyRpixel<XF_BAYER_PATTERN, XF_SRC_T, XF_DST_T, XF_HEIGHT, XF_WIDTH, XF_NPPC, 3 * XF_WIDTH>(
        blklevel_out_copy2, demosaic_out, demoOut_final);

    function_awb<XF_DST_T, XF_DST_T, XF_HEIGHT, XF_WIDTH, XF_NPPC>(demoOut_final, ltm_in, hist0, hist1, gain0, gain1,
                                                                   height, width, mode_reg, thresh);

    xf::cv::colorcorrectionmatrix<XF_CCM_TYPE, XF_DST_T, XF_DST_T, XF_HEIGHT, XF_WIDTH, XF_NPPC>(ltm_in, lsc_out);

    if (XF_DST_T == XF_8UC3) {
        fifo_copy<XF_DST_T, XF_LTM_T, XF_HEIGHT, XF_WIDTH, XF_NPPC>(lsc_out, aecin, height, width);
    } else {
        xf::cv::xf_QuatizationDithering<XF_DST_T, XF_LTM_T, XF_HEIGHT, XF_WIDTH, 256, Q_VAL, XF_NPPC>(lsc_out, aecin);
    }
    xf::cv::gammacorrection<XF_LTM_T, XF_LTM_T, XF_HEIGHT, XF_WIDTH, XF_NPPC>(aecin, _dst, gamma_lut);
    // ColorMat2AXIvideo<XF_LTM_T, XF_HEIGHT, XF_WIDTH, XF_NPPC>(_dst, out_axis_video);
    xf::cv::rgb2yuyv<XF_LTM_T, XF_16UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC>(_dst, _imgOutput);
    xfMat2AXIvideo<AXI_WIDTH_IN, XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC>(_imgOutput, out_axis_video);
    xfMat2AXIvideo<AXI_WIDTH_IN, XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC>(fullir_out_copy2, ir_axis_video);
}

/*********************************************************************************
 * Function:    ISPPipeline_accel
 * Parameters:  Stream of input/output pixels, image resolution
 * Return:
 * Description:
 **********************************************************************************/

void ISPPipeline_accel(uint16_t height,
                       uint16_t width,
                       InVideoStrm_t& in_axis_video,
                       InVideoStrm_t& out_axis_video,
                       OutVideoStrmIR_t& ir_axis_video,
                       char R_IR_C1_wgts[25],
                       char R_IR_C2_wgts[25],
                       char B_at_R_wgts[25],
                       char IR_at_R_wgts[9],
                       char IR_at_B_wgts[9],
                       char sub_wgts[4],
                       uint16_t rgain,
                       uint16_t bgain,
                       unsigned char gamma_lut[256 * 3],
                       unsigned char mode_reg,
                       uint16_t pawb) {
// Create AXI Streaming Interfaces for the core
// clang-format off
#pragma HLS INTERFACE axis port=in_axis_video register
#pragma HLS INTERFACE axis port=out_axis_video register
#pragma HLS INTERFACE axis port=ir_axis_video register

#pragma HLS INTERFACE s_axilite port=R_IR_C1_wgts bundle=CTRL
#pragma HLS INTERFACE s_axilite port=R_IR_C2_wgts bundle=CTRL
#pragma HLS INTERFACE s_axilite port=B_at_R_wgts bundle=CTRL
#pragma HLS INTERFACE s_axilite port=IR_at_R_wgts bundle=CTRL
#pragma HLS INTERFACE s_axilite port=IR_at_B_wgts bundle=CTRL
#pragma HLS INTERFACE s_axilite port=sub_wgts bundle=CTRL

#pragma HLS INTERFACE s_axilite port=width bundle=CTRL
#pragma HLS INTERFACE s_axilite port=height bundle=CTRL
#pragma HLS INTERFACE s_axilite port=rgain bundle=CTRL
#pragma HLS INTERFACE s_axilite port=bgain bundle=CTRL
#pragma HLS INTERFACE s_axilite port=mode_reg bundle=CTRL
#pragma HLS INTERFACE s_axilite port=pawb bundle=CTRL
#pragma HLS INTERFACE s_axilite port=gamma_lut bundle=CTRL

#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
// clang-format on
// clang-format off
#pragma HLS ARRAY_PARTITION variable=hist0_awb complete dim=1
#pragma HLS ARRAY_PARTITION variable=hist1_awb complete dim=1
    // clang-format on

    if (!flag) {
        ISPpipeline(height, width, in_axis_video, out_axis_video, ir_axis_video, R_IR_C1_wgts, R_IR_C2_wgts,
                    B_at_R_wgts, IR_at_R_wgts, IR_at_B_wgts, sub_wgts, hist0_awb, hist1_awb, igain_0, igain_1, rgain,
                    bgain, gamma_lut, mode_reg, pawb);
        flag = 1;

    } else {
        ISPpipeline(height, width, in_axis_video, out_axis_video, ir_axis_video, R_IR_C1_wgts, R_IR_C2_wgts,
                    B_at_R_wgts, IR_at_R_wgts, IR_at_B_wgts, sub_wgts, hist1_awb, hist0_awb, igain_1, igain_0, rgain,
                    bgain, gamma_lut, mode_reg, pawb);
        flag = 0;
    }
}
