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

static bool flag = 0;

static uint32_t hist0_awb[3][HIST_SIZE] = {0};
static uint32_t hist1_awb[3][HIST_SIZE] = {0};
// static uint32_t histogram0[1][256] = {0};
// static uint32_t histogram1[1][256] = {0};
static int igain_0[3] = {0};
static int igain_1[3] = {0};

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

void ISPpipeline(ap_uint<INPUT_PTR_WIDTH>* img_inp1,
                 ap_uint<INPUT_PTR_WIDTH>* img_inp2,
                 ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                 unsigned short height,
                 unsigned short width,
                 uint32_t hist0[3][HIST_SIZE],
                 uint32_t hist1[3][HIST_SIZE],
                 int gain0[3],
                 int gain1[3],
                 uint16_t rgain,
                 uint16_t bgain,
                 unsigned char gamma_lut[256 * 3],
                 unsigned char mode_reg,
                 uint16_t pawb,
                 short* wr_hls) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on
    xf::cv::Mat<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> imgInputhdr1(height, width);
    xf::cv::Mat<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> imgInputhdr2(height, width);

    xf::cv::Mat<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> imgInput1(height, width);
    xf::cv::Mat<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> imgInput2(height, width);
    xf::cv::Mat<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> bpc_out(height, width);
    xf::cv::Mat<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> gain_out(height, width);
    xf::cv::Mat<XF_DST_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> demosaic_out(height, width);
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

    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC>(img_inp1, imgInputhdr1);
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC>(img_inp2, imgInputhdr2);

    xf::cv::Hdrmerge_bayer<XF_SRC_T, XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC, NO_EXPS, W_B_SIZE>(
        imgInputhdr1, imgInputhdr2, imgInput1, wr_hls);

    xf::cv::blackLevelCorrection<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC, 16, 15, 1>(imgInput1, imgInput2, BLACK_LEVEL,
                                                                                    mul_fact);
    // xf::cv::badpixelcorrection<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC, 0, 0>(imgInput2, bpc_out);
    xf::cv::gaincontrol<XF_BAYER_PATTERN, XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC>(imgInput2, gain_out, rgain, bgain);
    xf::cv::demosaicing<XF_BAYER_PATTERN, XF_SRC_T, XF_DST_T, XF_HEIGHT, XF_WIDTH, XF_NPPC, 0>(gain_out, demosaic_out);

    function_awb<XF_DST_T, XF_DST_T, XF_HEIGHT, XF_WIDTH, XF_NPPC>(demosaic_out, ltm_in, hist0, hist1, gain0, gain1,
                                                                   height, width, mode_reg, thresh);

    xf::cv::colorcorrectionmatrix<XF_CCM_TYPE, XF_DST_T, XF_DST_T, XF_HEIGHT, XF_WIDTH, XF_NPPC>(ltm_in, lsc_out);

    if (XF_DST_T == XF_8UC3) {
        fifo_copy<XF_DST_T, XF_LTM_T, XF_HEIGHT, XF_WIDTH, XF_NPPC>(lsc_out, aecin, height, width);
    } else {
        xf::cv::xf_QuatizationDithering<XF_DST_T, XF_LTM_T, XF_HEIGHT, XF_WIDTH, 256, Q_VAL, XF_NPPC>(lsc_out, aecin);
    }
    xf::cv::gammacorrection<XF_LTM_T, XF_LTM_T, XF_HEIGHT, XF_WIDTH, XF_NPPC>(aecin, _dst, gamma_lut);
    // ColorMat2AXIvideo<XF_LTM_T, XF_HEIGHT, XF_WIDTH, XF_NPPC>(_dst, m_axis_video);
    // xf::cv::rgb2yuyv<XF_LTM_T, XF_16UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC>(_dst, _imgOutput);

    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, XF_HEIGHT, XF_WIDTH, XF_NPPC>(_dst, img_out);
}
/*********************************************************************************
 * Function:    ISPPipeline_accel
 * Parameters:  input and output image pointers, image resolution
 * Return:
 * Description:
 **********************************************************************************/
extern "C" {
void ISPPipeline_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp1,
                       ap_uint<INPUT_PTR_WIDTH>* img_inp2,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                       int height,
                       int width,
                       uint16_t rgain,
                       uint16_t bgain,
                       unsigned char gamma_lut[256 * 3],
                       unsigned char mode_reg,
                       uint16_t pawb,
                       short* wr_hls) {
// clang-format off
#pragma HLS INTERFACE m_axi     port=img_inp1  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=img_inp2  offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi     port=wr_hls  offset=slave bundle=gmem4
// clang-format on

// clang-format off
#pragma HLS ARRAY_PARTITION variable=hist0_awb complete dim=1
#pragma HLS ARRAY_PARTITION variable=hist1_awb complete dim=1

    // clang-format on

    if (!flag) {
        ISPpipeline(img_inp1, img_inp2, img_out, height, width, hist0_awb, hist1_awb, igain_0, igain_1, rgain, bgain,
                    gamma_lut, mode_reg, pawb, wr_hls);
        flag = 1;

    } else {
        ISPpipeline(img_inp1, img_inp2, img_out, height, width, hist1_awb, hist0_awb, igain_1, igain_0, rgain, bgain,
                    gamma_lut, mode_reg, pawb, wr_hls);
        flag = 0;
    }
}
}
