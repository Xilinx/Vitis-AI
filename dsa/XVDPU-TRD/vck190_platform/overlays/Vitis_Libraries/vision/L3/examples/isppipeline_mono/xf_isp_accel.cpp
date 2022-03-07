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

// static uint32_t histogram0[1][256] = {0};
// static uint32_t histogram1[1][256] = {0};

#define CLAHE_T                                                                                                        \
    xf::cv::clahe::CLAHEImpl<XF_LTM_T, XF_HEIGHT, XF_WIDTH, XF_NPPC, CLIPLIMIT, TILES_Y_MAX, TILES_X_MAX, TILES_Y_MIN, \
                             TILES_X_MIN>

static constexpr int HIST_COUNTER_BITS = CLAHE_T::HIST_COUNTER_BITS;
static constexpr int CLIP_COUNTER_BITS = CLAHE_T::CLIP_COUNTER_BITS;

static ap_uint<HIST_COUNTER_BITS> _lut1[TILES_Y_MAX][TILES_X_MAX][(XF_NPIXPERCYCLE(XF_NPPC) << 1)]
                                       [1 << XF_DTPIXELDEPTH(XF_LTM_T, XF_NPPC)];
static ap_uint<HIST_COUNTER_BITS> _lut2[TILES_Y_MAX][TILES_X_MAX][(XF_NPIXPERCYCLE(XF_NPPC) << 1)]
                                       [1 << XF_DTPIXELDEPTH(XF_LTM_T, XF_NPPC)];
static ap_uint<CLIP_COUNTER_BITS> _clipCounter[TILES_Y_MAX][TILES_X_MAX];

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

void ISPpipeline(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                 ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                 unsigned short height,
                 unsigned short width,
                 uint16_t lgain,
                 unsigned char gamma_lut[256],
                 ap_uint<HIST_COUNTER_BITS> _lutw[TILES_Y_MAX][TILES_X_MAX][(XF_NPIXPERCYCLE(XF_NPPC) << 1)]
                                                 [1 << XF_DTPIXELDEPTH(XF_LTM_T, XF_NPPC)],
                 ap_uint<HIST_COUNTER_BITS> _lutr[TILES_Y_MAX][TILES_X_MAX][(XF_NPIXPERCYCLE(XF_NPPC) << 1)]
                                                 [1 << XF_DTPIXELDEPTH(XF_LTM_T, XF_NPPC)],
                 ap_uint<CLIP_COUNTER_BITS> _clipCounter[TILES_Y_MAX][TILES_X_MAX],
                 int clip,
                 int tilesY,
                 int tilesX) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on
    xf::cv::Mat<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> imgInput1(height, width);
    xf::cv::Mat<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> imgInput2(height, width);
    xf::cv::Mat<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> dpc_out(height, width);
    xf::cv::Mat<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> gain_out(height, width);
    xf::cv::Mat<XF_DST_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> impop(height, width);
    xf::cv::Mat<XF_LTM_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> _dst(height, width);
    xf::cv::Mat<XF_LTM_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> aecin(height, width);
    xf::cv::Mat<XF_LTM_T, XF_HEIGHT, XF_WIDTH, XF_NPPC> _imgOutput(height, width);

// clang-format off
#pragma HLS DATAFLOW
    // clang-format on

    CLAHE_T obj;

    const int Q_VAL = 1 << (XF_DTPIXELDEPTH(XF_SRC_T, XF_NPPC));

    float inputMax = (1 << (XF_DTPIXELDEPTH(XF_SRC_T, XF_NPPC))) - 1; // 65535.0f;

    float mul_fact = (inputMax / (inputMax - BLACK_LEVEL));

    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC>(img_inp, imgInput1);
    xf::cv::blackLevelCorrection<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC, 16, 15, 1>(imgInput1, imgInput2, BLACK_LEVEL,
                                                                                    mul_fact);

    xf::cv::medianBlur<WINDOW_SIZE, XF_BORDER_REPLICATE, XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC>(imgInput2, dpc_out);
    xf::cv::gaincontrol_mono<XF_SRC_T, XF_HEIGHT, XF_WIDTH, XF_NPPC>(dpc_out, gain_out, lgain);

    if (XF_DST_T == XF_8UC1) {
        fifo_copy<XF_DST_T, XF_LTM_T, XF_HEIGHT, XF_WIDTH, XF_NPPC>(gain_out, aecin, height, width);
    } else {
        xf::cv::xf_QuatizationDithering<XF_DST_T, XF_LTM_T, XF_HEIGHT, XF_WIDTH, 256, Q_VAL, XF_NPPC>(gain_out, aecin);
    }

    obj.process(_dst, aecin, _lutw, _lutr, _clipCounter, height, width, clip, tilesY, tilesX);

    xf::cv::gammacorrection<XF_LTM_T, XF_LTM_T, XF_HEIGHT, XF_WIDTH, XF_NPPC>(_dst, _imgOutput, gamma_lut);

    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_LTM_T, XF_HEIGHT, XF_WIDTH, XF_NPPC>(_imgOutput, img_out);
}
/*********************************************************************************
 * Function:    ISPPipeline_accel
 * Parameters:  input and output image pointers, image resolution
 * Return:
 * Description:
 **********************************************************************************/
extern "C" {
void ISPPipeline_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                       int height,
                       int width,
                       uint16_t lgain,
                       unsigned char gamma_lut[256],
                       int clip,
                       int tilesY,
                       int tilesX) {
// clang-format off
#pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi      port=gamma_lut offset=slave  bundle=gmem3 depth=256

#pragma HLS INTERFACE s_axilite  port=clip
#pragma HLS INTERFACE s_axilite  port=tilesY
#pragma HLS INTERFACE s_axilite  port=tilesX
#pragma HLS INTERFACE s_axilite  port=return

#pragma HLS ARRAY_PARTITION variable=_lut1 dim=3 complete
#pragma HLS ARRAY_PARTITION variable=_lut2 dim=3 complete
    // clang-format on

    if (!flag) {
        ISPpipeline(img_inp, img_out, height, width, lgain, gamma_lut, _lut1, _lut2, _clipCounter, clip, tilesX,
                    tilesY);
        flag = 1;

    } else {
        ISPpipeline(img_inp, img_out, height, width, lgain, gamma_lut, _lut2, _lut1, _clipCounter, clip, tilesX,
                    tilesY);
        flag = 0;
    }
}
}
