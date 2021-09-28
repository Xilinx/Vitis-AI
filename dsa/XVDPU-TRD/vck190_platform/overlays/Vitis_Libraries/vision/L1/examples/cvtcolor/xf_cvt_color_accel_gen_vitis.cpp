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

#include "xf_cvt_color_config_gen_vitis.h"

#if RGBA2IYUV
void cvtcolor_rgba2iyuv(ap_uint<32 * NPC1>* imgInput,
                        ap_uint<8 * NPC1>* imgOutput0,
                        ap_uint<8 * NPC1>* imgOutput1,
                        ap_uint<8 * NPC1>* imgOutput2) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC4, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_1 = (((HEIGHT / 4)) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_2 = (((HEIGHT / 4)) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput0  offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE m_axi      port=imgOutput1  offset=slave  bundle=gmem_out1 depth=__XF_DEPTH_OUT_1
    #pragma HLS INTERFACE m_axi      port=imgOutput2  offset=slave  bundle=gmem_out2 depth=__XF_DEPTH_OUT_2
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC4, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, (HEIGHT / 4), WIDTH, NPC1> _imgOutput1((HEIGHT / 4), WIDTH);
    xf::cv::Mat<XF_8UC1, (HEIGHT / 4), WIDTH, NPC1> _imgOutput2((HEIGHT / 4), WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC4, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::rgba2iyuv<XF_8UC4, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput0, _imgOutput1, _imgOutput2);

    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput0, imgOutput0);
    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, (HEIGHT / 4), WIDTH, NPC1>(_imgOutput1, imgOutput1);
    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, (HEIGHT / 4), WIDTH, NPC1>(_imgOutput2, imgOutput2);
}
#endif
#if RGBA2NV12
void cvtcolor_rgba2nv12(ap_uint<32 * NPC1>* imgInput, ap_uint<8 * NPC1>* imgOutput0, ap_uint<16 * NPC2>* imgOutput1) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC4, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput0  offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE m_axi      port=imgOutput1  offset=slave  bundle=gmem_out1 depth=__XF_DEPTH_OUT_1
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC4, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgOutput1((HEIGHT / 2), (WIDTH / 2));

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC4, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::rgba2nv12<XF_8UC4, XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput, _imgOutput0, _imgOutput1);

    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput0, imgOutput0);
    xf::cv::xfMat2Array<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(_imgOutput1, imgOutput1);
}
#endif
#if RGBA2NV21
void cvtcolor_rgba2nv21(ap_uint<32 * NPC1>* imgInput, ap_uint<8 * NPC1>* imgOutput0, ap_uint<16 * NPC2>* imgOutput1) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC4, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput0  offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE m_axi      port=imgOutput1  offset=slave  bundle=gmem_out1 depth=__XF_DEPTH_OUT_1
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC4, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgOutput1((HEIGHT / 2), (WIDTH / 2));

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC4, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::rgba2nv21<XF_8UC4, XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput, _imgOutput0, _imgOutput1);

    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput0, imgOutput0);
    xf::cv::xfMat2Array<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(_imgOutput1, imgOutput1);
}
#endif
#if RGBA2YUV4
void cvtcolor_rgba2yuv4(ap_uint<32 * NPC1>* imgInput,
                        ap_uint<8 * NPC1>* imgOutput0,
                        ap_uint<8 * NPC1>* imgOutput1,
                        ap_uint<8 * NPC1>* imgOutput2) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC4, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_1 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_2 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput0  offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE m_axi      port=imgOutput1  offset=slave  bundle=gmem_out1 depth=__XF_DEPTH_OUT_1
    #pragma HLS INTERFACE m_axi      port=imgOutput2  offset=slave  bundle=gmem_out2 depth=__XF_DEPTH_OUT_2
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC4, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput1(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput2(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC4, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::rgba2yuv4<XF_8UC4, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput0, _imgOutput1, _imgOutput2);

    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput0, imgOutput0);
    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput1, imgOutput1);
    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput2, imgOutput2);
}
#endif
#if RGB2IYUV
void cvtcolor_rgb2iyuv(ap_uint<32 * NPC1>* imgInput,
                       ap_uint<8 * NPC1>* imgOutput0,
                       ap_uint<8 * NPC1>* imgOutput1,
                       ap_uint<8 * NPC1>* imgOutput2) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_1 = (((HEIGHT / 4)) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_2 = (((HEIGHT / 4)) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput0  offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE m_axi      port=imgOutput1  offset=slave  bundle=gmem_out1 depth=__XF_DEPTH_OUT_1
    #pragma HLS INTERFACE m_axi      port=imgOutput2  offset=slave  bundle=gmem_out2 depth=__XF_DEPTH_OUT_2
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, (HEIGHT / 4), WIDTH, NPC1> _imgOutput1((HEIGHT / 4), WIDTH);
    xf::cv::Mat<XF_8UC1, (HEIGHT / 4), WIDTH, NPC1> _imgOutput2((HEIGHT / 4), WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::rgb2iyuv<XF_8UC3, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput0, _imgOutput1, _imgOutput2);

    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput0, imgOutput0);
    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, (HEIGHT / 4), WIDTH, NPC1>(_imgOutput1, imgOutput1);
    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, (HEIGHT / 4), WIDTH, NPC1>(_imgOutput2, imgOutput2);
}
#endif
#if RGB2NV12
void cvtcolor_rgb2nv12(ap_uint<32 * NPC1>* imgInput, ap_uint<8 * NPC1>* imgOutput0, ap_uint<16 * NPC2>* imgOutput1) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput0  offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE m_axi      port=imgOutput1  offset=slave  bundle=gmem_out1 depth=__XF_DEPTH_OUT_1
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgOutput1((HEIGHT / 2), (WIDTH / 2));

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::rgb2nv12<XF_8UC3, XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput, _imgOutput0, _imgOutput1);

    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput0, imgOutput0);
    xf::cv::xfMat2Array<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(_imgOutput1, imgOutput1);
}
#endif
#if RGB2NV21
void cvtcolor_rgb2nv21(ap_uint<32 * NPC1>* imgInput, ap_uint<8 * NPC1>* imgOutput0, ap_uint<16 * NPC2>* imgOutput1) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput0  offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE m_axi      port=imgOutput1  offset=slave  bundle=gmem_out1 depth=__XF_DEPTH_OUT_1
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgOutput1((HEIGHT / 2), (WIDTH / 2));

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::rgb2nv21<XF_8UC3, XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput, _imgOutput0, _imgOutput1);

    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput0, imgOutput0);
    xf::cv::xfMat2Array<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(_imgOutput1, imgOutput1);
}
#endif
#if RGB2YUV4
void cvtcolor_rgb2yuv4(ap_uint<32 * NPC1>* imgInput,
                       ap_uint<8 * NPC1>* imgOutput0,
                       ap_uint<8 * NPC1>* imgOutput1,
                       ap_uint<8 * NPC1>* imgOutput2) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_1 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_2 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput0  offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE m_axi      port=imgOutput1  offset=slave  bundle=gmem_out1 depth=__XF_DEPTH_OUT_1
    #pragma HLS INTERFACE m_axi      port=imgOutput2  offset=slave  bundle=gmem_out2 depth=__XF_DEPTH_OUT_2
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput1(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput2(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::rgb2yuv4<XF_8UC3, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput0, _imgOutput1, _imgOutput2);

    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput0, imgOutput0);
    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput1, imgOutput1);
    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput2, imgOutput2);
}
#endif
#if RGB2UYVY
void cvtcolor_rgb2uyvy(ap_uint<32 * NPC1>* imgInput, ap_uint<16 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_16UC1, NPC1))) / (16 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::rgb2uyvy<XF_8UC3, XF_16UC1, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<16 * NPC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if RGB2YUYV
void cvtcolor_rgb2yuyv(ap_uint<32 * NPC1>* imgInput, ap_uint<16 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_16UC1, NPC1))) / (16 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::rgb2yuyv<XF_8UC3, XF_16UC1, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<16 * NPC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if RGB2BGR
void cvtcolor_rgb2bgr(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::rgb2bgr<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if BGR2UYVY
void cvtcolor_bgr2uyvy(ap_uint<32 * NPC1>* imgInput, ap_uint<16 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_16UC1, NPC1))) / (16 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::bgr2uyvy<XF_8UC3, XF_16UC1, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<16 * NPC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if BGR2YUYV
void cvtcolor_bgr2yuyv(ap_uint<32 * NPC1>* imgInput, ap_uint<16 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_16UC1, NPC1))) / (16 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::bgr2yuyv<XF_8UC3, XF_16UC1, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<16 * NPC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if BGR2RGB
void cvtcolor_bgr2rgb(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::bgr2rgb<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if BGR2NV12
void cvtcolor_bgr2nv12(ap_uint<32 * NPC1>* imgInput, ap_uint<8 * NPC1>* imgOutput0, ap_uint<16 * NPC2>* imgOutput1) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput0  offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE m_axi      port=imgOutput1  offset=slave  bundle=gmem_out1 depth=__XF_DEPTH_OUT_1
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgOutput1((HEIGHT / 2), (WIDTH / 2));

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::bgr2nv12<XF_8UC3, XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput, _imgOutput0, _imgOutput1);

    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput0, imgOutput0);
    xf::cv::xfMat2Array<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(_imgOutput1, imgOutput1);
}
#endif
#if BGR2NV21
void cvtcolor_bgr2nv21(ap_uint<32 * NPC1>* imgInput, ap_uint<8 * NPC1>* imgOutput0, ap_uint<16 * NPC2>* imgOutput1) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput0  offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE m_axi      port=imgOutput1  offset=slave  bundle=gmem_out1 depth=__XF_DEPTH_OUT_1
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgOutput1((HEIGHT / 2), (WIDTH / 2));

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::bgr2nv21<XF_8UC3, XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput, _imgOutput0, _imgOutput1);

    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput0, imgOutput0);
    xf::cv::xfMat2Array<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(_imgOutput1, imgOutput1);
}
#endif
#if IYUV2NV12
void cvtcolor_iyuv2nv12(ap_uint<8 * NPC1>* imgInput0,
                        ap_uint<8 * NPC1>* imgInput1,
                        ap_uint<8 * NPC1>* imgInput2,
                        ap_uint<8 * NPC1>* imgOutput0,
                        ap_uint<16 * NPC2>* imgOutput1) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_INP_1 = (((HEIGHT / 4)) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_INP_2 = (((HEIGHT / 4)) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput0   offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgInput1   offset=slave  bundle=gmem_in1  depth=__XF_DEPTH_INP_1
    #pragma HLS INTERFACE m_axi      port=imgInput2   offset=slave  bundle=gmem_in2  depth=__XF_DEPTH_INP_2
    #pragma HLS INTERFACE m_axi      port=imgOutput0  offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE m_axi      port=imgOutput1  offset=slave  bundle=gmem_out1 depth=__XF_DEPTH_OUT_1
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgInput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, (HEIGHT / 4), WIDTH, NPC1> _imgInput1((HEIGHT / 4), WIDTH);
    xf::cv::Mat<XF_8UC1, (HEIGHT / 4), WIDTH, NPC1> _imgInput2((HEIGHT / 4), WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgOutput1((HEIGHT / 2), (WIDTH / 2));

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, _imgInput0);
    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, (HEIGHT / 4), WIDTH, NPC1>(imgInput1, _imgInput1);
    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, (HEIGHT / 4), WIDTH, NPC1>(imgInput2, _imgInput2);

    xf::cv::iyuv2nv12<XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput0, _imgInput1, _imgInput2, _imgOutput0,
                                                                   _imgOutput1);

    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput0, imgOutput0);
    xf::cv::xfMat2Array<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(_imgOutput1, imgOutput1);
}
#endif
#if IYUV2RGBA
void cvtcolor_iyuv2rgba(ap_uint<8 * NPC1>* imgInput0,
                        ap_uint<8 * NPC1>* imgInput1,
                        ap_uint<8 * NPC1>* imgInput2,
                        ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_INP_1 = (((HEIGHT / 4)) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_INP_2 = (((HEIGHT / 4)) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC4, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput0   offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgInput1   offset=slave  bundle=gmem_in1  depth=__XF_DEPTH_INP_1
    #pragma HLS INTERFACE m_axi      port=imgInput2   offset=slave  bundle=gmem_in2  depth=__XF_DEPTH_INP_2
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgInput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, (HEIGHT / 4), WIDTH, NPC1> _imgInput1((HEIGHT / 4), WIDTH);
    xf::cv::Mat<XF_8UC1, (HEIGHT / 4), WIDTH, NPC1> _imgInput2((HEIGHT / 4), WIDTH);
    xf::cv::Mat<XF_8UC4, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, _imgInput0);
    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, (HEIGHT / 4), WIDTH, NPC1>(imgInput1, _imgInput1);
    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, (HEIGHT / 4), WIDTH, NPC1>(imgInput2, _imgInput2);

    xf::cv::iyuv2rgba<XF_8UC1, XF_8UC4, HEIGHT, WIDTH, NPC1>(_imgInput0, _imgInput1, _imgInput2, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC4, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if IYUV2RGB
void cvtcolor_iyuv2rgb(ap_uint<8 * NPC1>* imgInput0,
                       ap_uint<8 * NPC1>* imgInput1,
                       ap_uint<8 * NPC1>* imgInput2,
                       ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_INP_1 = (((HEIGHT / 4)) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_INP_2 = (((HEIGHT / 4)) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput0   offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgInput1   offset=slave  bundle=gmem_in1  depth=__XF_DEPTH_INP_1
    #pragma HLS INTERFACE m_axi      port=imgInput2   offset=slave  bundle=gmem_in2  depth=__XF_DEPTH_INP_2
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgInput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, (HEIGHT / 4), WIDTH, NPC1> _imgInput1((HEIGHT / 4), WIDTH);
    xf::cv::Mat<XF_8UC1, (HEIGHT / 4), WIDTH, NPC1> _imgInput2((HEIGHT / 4), WIDTH);
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, _imgInput0);
    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, (HEIGHT / 4), WIDTH, NPC1>(imgInput1, _imgInput1);
    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, (HEIGHT / 4), WIDTH, NPC1>(imgInput2, _imgInput2);

    xf::cv::iyuv2rgb<XF_8UC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgInput0, _imgInput1, _imgInput2, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if IYUV2YUV4
void cvtcolor_iyuv2yuv4(ap_uint<8 * NPC1>* imgInput0,
                        ap_uint<8 * NPC1>* imgInput1,
                        ap_uint<8 * NPC1>* imgInput2,
                        ap_uint<8 * NPC1>* imgOutput0,
                        ap_uint<8 * NPC1>* imgOutput1,
                        ap_uint<8 * NPC1>* imgOutput2) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_INP_1 = (((HEIGHT / 4)) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_INP_2 = (((HEIGHT / 4)) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_1 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_2 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput0   offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgInput1   offset=slave  bundle=gmem_in1  depth=__XF_DEPTH_INP_1
    #pragma HLS INTERFACE m_axi      port=imgInput2   offset=slave  bundle=gmem_in2  depth=__XF_DEPTH_INP_2
    #pragma HLS INTERFACE m_axi      port=imgOutput0  offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE m_axi      port=imgOutput1  offset=slave  bundle=gmem_out1 depth=__XF_DEPTH_OUT_1
    #pragma HLS INTERFACE m_axi      port=imgOutput2  offset=slave  bundle=gmem_out2 depth=__XF_DEPTH_OUT_2
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgInput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, (HEIGHT / 4), WIDTH, NPC1> _imgInput1((HEIGHT / 4), WIDTH);
    xf::cv::Mat<XF_8UC1, (HEIGHT / 4), WIDTH, NPC1> _imgInput2((HEIGHT / 4), WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput1(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput2(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, _imgInput0);
    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, (HEIGHT / 4), WIDTH, NPC1>(imgInput1, _imgInput1);
    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, (HEIGHT / 4), WIDTH, NPC1>(imgInput2, _imgInput2);

    xf::cv::iyuv2yuv4<XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgInput0, _imgInput1, _imgInput2, _imgOutput0, _imgOutput1,
                                                    _imgOutput2);

    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput0, imgOutput0);
    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput1, imgOutput1);
    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput2, imgOutput2);
}
#endif
#if NV122IYUV
void cvtcolor_nv122iyuv(ap_uint<8 * NPC1>* imgInput0,
                        ap_uint<16 * NPC2>* imgInput1,
                        ap_uint<8 * NPC1>* imgOutput0,
                        ap_uint<8 * NPC1>* imgOutput1,
                        ap_uint<8 * NPC1>* imgOutput2) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_INP_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_1 = (((HEIGHT / 4)) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_2 = (((HEIGHT / 4)) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput0   offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgInput1   offset=slave  bundle=gmem_in1  depth=__XF_DEPTH_INP_1
    #pragma HLS INTERFACE m_axi      port=imgOutput0  offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE m_axi      port=imgOutput1  offset=slave  bundle=gmem_out1 depth=__XF_DEPTH_OUT_1
    #pragma HLS INTERFACE m_axi      port=imgOutput2  offset=slave  bundle=gmem_out2 depth=__XF_DEPTH_OUT_2
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgInput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgInput1((HEIGHT / 2), (WIDTH / 2));
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, (HEIGHT / 4), WIDTH, NPC1> _imgOutput1((HEIGHT / 4), WIDTH);
    xf::cv::Mat<XF_8UC1, (HEIGHT / 4), WIDTH, NPC1> _imgOutput2((HEIGHT / 4), WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, _imgInput0);
    xf::cv::Array2xfMat<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(imgInput1, _imgInput1);

    xf::cv::nv122iyuv<XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput0, _imgInput1, _imgOutput0, _imgOutput1,
                                                                   _imgOutput2);

    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput0, imgOutput0);
    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, (HEIGHT / 4), WIDTH, NPC1>(_imgOutput1, imgOutput1);
    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, (HEIGHT / 4), WIDTH, NPC1>(_imgOutput2, imgOutput2);
}
#endif
#if NV122RGBA
void cvtcolor_nv122rgba(ap_uint<8 * NPC1>* imgInput0, ap_uint<16 * NPC2>* imgInput1, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_INP_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC4, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput0   offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgInput1   offset=slave  bundle=gmem_in1  depth=__XF_DEPTH_INP_1
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgInput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgInput1((HEIGHT / 2), (WIDTH / 2));
    xf::cv::Mat<XF_8UC4, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, _imgInput0);
    xf::cv::Array2xfMat<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(imgInput1, _imgInput1);

    xf::cv::nv122rgba<XF_8UC1, XF_8UC2, XF_8UC4, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput0, _imgInput1, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC4, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if NV122YUV4
void cvtcolor_nv122yuv4(ap_uint<8 * NPC1>* imgInput0,
                        ap_uint<16 * NPC2>* imgInput1,
                        ap_uint<8 * NPC1>* imgOutput0,
                        ap_uint<8 * NPC1>* imgOutput1,
                        ap_uint<8 * NPC1>* imgOutput2) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_INP_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_1 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_2 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput0   offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgInput1   offset=slave  bundle=gmem_in1  depth=__XF_DEPTH_INP_1
    #pragma HLS INTERFACE m_axi      port=imgOutput0  offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE m_axi      port=imgOutput1  offset=slave  bundle=gmem_out1 depth=__XF_DEPTH_OUT_1
    #pragma HLS INTERFACE m_axi      port=imgOutput2  offset=slave  bundle=gmem_out2 depth=__XF_DEPTH_OUT_2
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgInput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgInput1((HEIGHT / 2), (WIDTH / 2));
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput1(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput2(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, _imgInput0);
    xf::cv::Array2xfMat<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(imgInput1, _imgInput1);

    xf::cv::nv122yuv4<XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput0, _imgInput1, _imgOutput0, _imgOutput1,
                                                                   _imgOutput2);

    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput0, imgOutput0);
    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput1, imgOutput1);
    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput2, imgOutput2);
}
#endif
#if NV122RGB
void cvtcolor_nv122rgb(ap_uint<8 * NPC1>* imgInput0, ap_uint<16 * NPC2>* imgInput1, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_INP_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput0   offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgInput1   offset=slave  bundle=gmem_in1  depth=__XF_DEPTH_INP_1
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgInput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgInput1((HEIGHT / 2), (WIDTH / 2));
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, _imgInput0);
    xf::cv::Array2xfMat<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(imgInput1, _imgInput1);

    xf::cv::nv122rgb<XF_8UC1, XF_8UC2, XF_8UC3, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput0, _imgInput1, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if NV122BGR
void cvtcolor_nv122bgr(ap_uint<8 * NPC1>* imgInput0, ap_uint<16 * NPC2>* imgInput1, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_INP_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput0   offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgInput1   offset=slave  bundle=gmem_in1  depth=__XF_DEPTH_INP_1
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgInput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgInput1((HEIGHT / 2), (WIDTH / 2));
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, _imgInput0);
    xf::cv::Array2xfMat<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(imgInput1, _imgInput1);

    xf::cv::nv122bgr<XF_8UC1, XF_8UC2, XF_8UC3, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput0, _imgInput1, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if NV122UYVY
void cvtcolor_nv122uyvy(ap_uint<8 * NPC1>* imgInput0, ap_uint<16 * NPC2>* imgInput1, ap_uint<16 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_INP_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_16UC1, NPC1))) / (16 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput0   offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgInput1   offset=slave  bundle=gmem_in1  depth=__XF_DEPTH_INP_1
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgInput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgInput1((HEIGHT / 2), (WIDTH / 2));
    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, _imgInput0);
    xf::cv::Array2xfMat<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(imgInput1, _imgInput1);

    xf::cv::nv122uyvy<XF_8UC1, XF_8UC2, XF_16UC1, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput0, _imgInput1, _imgOutput);

    xf::cv::xfMat2Array<16 * NPC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if NV122YUYV
void cvtcolor_nv122yuyv(ap_uint<8 * NPC1>* imgInput0, ap_uint<16 * NPC2>* imgInput1, ap_uint<16 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_INP_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_16UC1, NPC1))) / (16 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput0   offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgInput1   offset=slave  bundle=gmem_in1  depth=__XF_DEPTH_INP_1
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgInput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgInput1((HEIGHT / 2), (WIDTH / 2));
    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, _imgInput0);
    xf::cv::Array2xfMat<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(imgInput1, _imgInput1);

    xf::cv::nv122yuyv<XF_8UC1, XF_8UC2, XF_16UC1, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput0, _imgInput1, _imgOutput);

    xf::cv::xfMat2Array<16 * NPC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if NV122NV21
void cvtcolor_nv122nv21(ap_uint<8 * NPC1>* imgInput0,
                        ap_uint<16 * NPC2>* imgInput1,
                        ap_uint<8 * NPC1>* imgOutput0,
                        ap_uint<16 * NPC2>* imgOutput1) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_INP_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput0   offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgInput1   offset=slave  bundle=gmem_in1  depth=__XF_DEPTH_INP_1
    #pragma HLS INTERFACE m_axi      port=imgOutput0  offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE m_axi      port=imgOutput1  offset=slave  bundle=gmem_out1 depth=__XF_DEPTH_OUT_1
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgInput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgInput1((HEIGHT / 2), (WIDTH / 2));
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgOutput1((HEIGHT / 2), (WIDTH / 2));

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, _imgInput0);
    xf::cv::Array2xfMat<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(imgInput1, _imgInput1);

    xf::cv::nv122nv21<XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput0, _imgInput1, _imgOutput0, _imgOutput1);

    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput0, imgOutput0);
    xf::cv::xfMat2Array<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(_imgOutput1, imgOutput1);
}
#endif
#if NV212IYUV
void cvtcolor_nv212iyuv(ap_uint<8 * NPC1>* imgInput0,
                        ap_uint<16 * NPC2>* imgInput1,
                        ap_uint<8 * NPC1>* imgOutput0,
                        ap_uint<8 * NPC1>* imgOutput1,
                        ap_uint<8 * NPC1>* imgOutput2) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_INP_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_1 = (((HEIGHT / 4)) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_2 = (((HEIGHT / 4)) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput0   offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgInput1   offset=slave  bundle=gmem_in1  depth=__XF_DEPTH_INP_1
    #pragma HLS INTERFACE m_axi      port=imgOutput0  offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE m_axi      port=imgOutput1  offset=slave  bundle=gmem_out1 depth=__XF_DEPTH_OUT_1
    #pragma HLS INTERFACE m_axi      port=imgOutput2  offset=slave  bundle=gmem_out2 depth=__XF_DEPTH_OUT_2
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgInput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgInput1((HEIGHT / 2), (WIDTH / 2));
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, (HEIGHT / 4), WIDTH, NPC1> _imgOutput1((HEIGHT / 4), WIDTH);
    xf::cv::Mat<XF_8UC1, (HEIGHT / 4), WIDTH, NPC1> _imgOutput2((HEIGHT / 4), WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, _imgInput0);
    xf::cv::Array2xfMat<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(imgInput1, _imgInput1);

    xf::cv::nv212iyuv<XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput0, _imgInput1, _imgOutput0, _imgOutput1,
                                                                   _imgOutput2);

    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput0, imgOutput0);
    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, (HEIGHT / 4), WIDTH, NPC1>(_imgOutput1, imgOutput1);
    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, (HEIGHT / 4), WIDTH, NPC1>(_imgOutput2, imgOutput2);
}
#endif
#if NV212RGBA
void cvtcolor_nv212rgba(ap_uint<8 * NPC1>* imgInput0, ap_uint<16 * NPC2>* imgInput1, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_INP_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC4, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput0   offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgInput1   offset=slave  bundle=gmem_in1  depth=__XF_DEPTH_INP_1
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgInput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgInput1((HEIGHT / 2), (WIDTH / 2));
    xf::cv::Mat<XF_8UC4, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, _imgInput0);
    xf::cv::Array2xfMat<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(imgInput1, _imgInput1);

    xf::cv::nv212rgba<XF_8UC1, XF_8UC2, XF_8UC4, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput0, _imgInput1, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC4, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if NV212RGB
void cvtcolor_nv212rgb(ap_uint<8 * NPC1>* imgInput0, ap_uint<16 * NPC2>* imgInput1, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_INP_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput0   offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgInput1   offset=slave  bundle=gmem_in1  depth=__XF_DEPTH_INP_1
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgInput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgInput1((HEIGHT / 2), (WIDTH / 2));
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, _imgInput0);
    xf::cv::Array2xfMat<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(imgInput1, _imgInput1);

    xf::cv::nv212rgb<XF_8UC1, XF_8UC2, XF_8UC3, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput0, _imgInput1, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if NV212BGR
void cvtcolor_nv212bgr(ap_uint<8 * NPC1>* imgInput0, ap_uint<16 * NPC2>* imgInput1, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_INP_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput0   offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgInput1   offset=slave  bundle=gmem_in1  depth=__XF_DEPTH_INP_1
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgInput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgInput1((HEIGHT / 2), (WIDTH / 2));
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, _imgInput0);
    xf::cv::Array2xfMat<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(imgInput1, _imgInput1);

    xf::cv::nv212bgr<XF_8UC1, XF_8UC2, XF_8UC3, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput0, _imgInput1, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if NV212YUV4
void cvtcolor_nv212yuv4(ap_uint<8 * NPC1>* imgInput0,
                        ap_uint<16 * NPC2>* imgInput1,
                        ap_uint<8 * NPC1>* imgOutput0,
                        ap_uint<8 * NPC1>* imgOutput1,
                        ap_uint<8 * NPC1>* imgOutput2) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_INP_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_1 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_2 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput0   offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgInput1   offset=slave  bundle=gmem_in1  depth=__XF_DEPTH_INP_1
    #pragma HLS INTERFACE m_axi      port=imgOutput0  offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE m_axi      port=imgOutput1  offset=slave  bundle=gmem_out1 depth=__XF_DEPTH_OUT_1
    #pragma HLS INTERFACE m_axi      port=imgOutput2  offset=slave  bundle=gmem_out2 depth=__XF_DEPTH_OUT_2
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgInput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgInput1((HEIGHT / 2), (WIDTH / 2));
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput1(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput2(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, _imgInput0);
    xf::cv::Array2xfMat<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(imgInput1, _imgInput1);

    xf::cv::nv212yuv4<XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput0, _imgInput1, _imgOutput0, _imgOutput1,
                                                                   _imgOutput2);

    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput0, imgOutput0);
    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput1, imgOutput1);
    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput2, imgOutput2);
}
#endif
#if NV212UYVY
void cvtcolor_nv212uyvy(ap_uint<8 * NPC1>* imgInput0, ap_uint<16 * NPC2>* imgInput1, ap_uint<16 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_INP_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_16UC1, NPC1))) / (16 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput0   offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgInput1   offset=slave  bundle=gmem_in1  depth=__XF_DEPTH_INP_1
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgInput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgInput1((HEIGHT / 2), (WIDTH / 2));
    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, _imgInput0);
    xf::cv::Array2xfMat<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(imgInput1, _imgInput1);

    xf::cv::nv212uyvy<XF_8UC1, XF_8UC2, XF_16UC1, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput0, _imgInput1, _imgOutput);

    xf::cv::xfMat2Array<16 * NPC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if NV212YUYV
void cvtcolor_nv212yuyv(ap_uint<8 * NPC1>* imgInput0, ap_uint<16 * NPC2>* imgInput1, ap_uint<16 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_INP_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_16UC1, NPC1))) / (16 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput0   offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgInput1   offset=slave  bundle=gmem_in1  depth=__XF_DEPTH_INP_1
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgInput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgInput1((HEIGHT / 2), (WIDTH / 2));
    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, _imgInput0);
    xf::cv::Array2xfMat<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(imgInput1, _imgInput1);

    xf::cv::nv212yuyv<XF_8UC1, XF_8UC2, XF_16UC1, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput0, _imgInput1, _imgOutput);

    xf::cv::xfMat2Array<16 * NPC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if NV212NV12
void cvtcolor_nv212nv12(ap_uint<8 * NPC1>* imgInput0,
                        ap_uint<16 * NPC2>* imgInput1,
                        ap_uint<8 * NPC1>* imgOutput0,
                        ap_uint<16 * NPC2>* imgOutput1) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_INP_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput0   offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgInput1   offset=slave  bundle=gmem_in1  depth=__XF_DEPTH_INP_1
    #pragma HLS INTERFACE m_axi      port=imgOutput0  offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE m_axi      port=imgOutput1  offset=slave  bundle=gmem_out1 depth=__XF_DEPTH_OUT_1
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgInput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgInput1((HEIGHT / 2), (WIDTH / 2));
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgOutput1((HEIGHT / 2), (WIDTH / 2));

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, _imgInput0);
    xf::cv::Array2xfMat<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(imgInput1, _imgInput1);

    xf::cv::nv212nv12<XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput0, _imgInput1, _imgOutput0, _imgOutput1);

    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput0, imgOutput0);
    xf::cv::xfMat2Array<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(_imgOutput1, imgOutput1);
}
#endif
#if UYVY2IYUV
void cvtcolor_uyvy2iyuv(ap_uint<16 * NPC1>* imgInput,
                        ap_uint<8 * NPC1>* imgOutput0,
                        ap_uint<8 * NPC1>* imgOutput1,
                        ap_uint<8 * NPC1>* imgOutput2) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_16UC1, NPC1))) / (16 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_1 = (((HEIGHT / 4)) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_2 = (((HEIGHT / 4)) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput0  offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE m_axi      port=imgOutput1  offset=slave  bundle=gmem_out1 depth=__XF_DEPTH_OUT_1
    #pragma HLS INTERFACE m_axi      port=imgOutput2  offset=slave  bundle=gmem_out2 depth=__XF_DEPTH_OUT_2
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, (HEIGHT / 4), WIDTH, NPC1> _imgOutput1((HEIGHT / 4), WIDTH);
    xf::cv::Mat<XF_8UC1, (HEIGHT / 4), WIDTH, NPC1> _imgOutput2((HEIGHT / 4), WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<16 * NPC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::uyvy2iyuv<XF_16UC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput0, _imgOutput1, _imgOutput2);

    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput0, imgOutput0);
    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, (HEIGHT / 4), WIDTH, NPC1>(_imgOutput1, imgOutput1);
    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, (HEIGHT / 4), WIDTH, NPC1>(_imgOutput2, imgOutput2);
}
#endif
#if UYVY2NV12
void cvtcolor_uyvy2nv12(ap_uint<16 * NPC1>* imgInput, ap_uint<8 * NPC1>* imgOutput0, ap_uint<16 * NPC2>* imgOutput1) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_16UC1, NPC1))) / (16 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput0  offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE m_axi      port=imgOutput1  offset=slave  bundle=gmem_out1 depth=__XF_DEPTH_OUT_1
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgOutput1((HEIGHT / 2), (WIDTH / 2));

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<16 * NPC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::uyvy2nv12<XF_16UC1, XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput, _imgOutput0, _imgOutput1);

    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput0, imgOutput0);
    xf::cv::xfMat2Array<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(_imgOutput1, imgOutput1);
}
#endif
#if UYVY2NV21
void cvtcolor_uyvy2nv21(ap_uint<16 * NPC1>* imgInput, ap_uint<8 * NPC1>* imgOutput0, ap_uint<16 * NPC2>* imgOutput1) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_16UC1, NPC1))) / (16 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput0  offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE m_axi      port=imgOutput1  offset=slave  bundle=gmem_out1 depth=__XF_DEPTH_OUT_1
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgOutput1((HEIGHT / 2), (WIDTH / 2));

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<16 * NPC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::uyvy2nv21<XF_16UC1, XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput, _imgOutput0, _imgOutput1);

    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput0, imgOutput0);
    xf::cv::xfMat2Array<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(_imgOutput1, imgOutput1);
}
#endif
#if UYVY2RGBA
void cvtcolor_uyvy2rgba(ap_uint<16 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_16UC1, NPC1))) / (16 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC4, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC4, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<16 * NPC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::uyvy2rgba<XF_16UC1, XF_8UC4, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC4, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if UYVY2RGB
void cvtcolor_uyvy2rgb(ap_uint<16 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_16UC1, NPC1))) / (16 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<16 * NPC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::uyvy2rgb<XF_16UC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if UYVY2BGR
void cvtcolor_uyvy2bgr(ap_uint<16 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_16UC1, NPC1))) / (16 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<16 * NPC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::uyvy2bgr<XF_16UC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if UYVY2YUYV
void cvtcolor_uyvy2yuyv(ap_uint<16 * NPC1>* imgInput, ap_uint<16 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_16UC1, NPC1))) / (16 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_16UC1, NPC1))) / (16 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<16 * NPC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::uyvy2yuyv<XF_16UC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<16 * NPC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if YUYV2IYUV
void cvtcolor_yuyv2iyuv(ap_uint<16 * NPC1>* imgInput,
                        ap_uint<8 * NPC1>* imgOutput0,
                        ap_uint<8 * NPC1>* imgOutput1,
                        ap_uint<8 * NPC1>* imgOutput2) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_16UC1, NPC1))) / (16 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_1 = (((HEIGHT / 4)) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_2 = (((HEIGHT / 4)) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput0  offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE m_axi      port=imgOutput1  offset=slave  bundle=gmem_out1 depth=__XF_DEPTH_OUT_1
    #pragma HLS INTERFACE m_axi      port=imgOutput2  offset=slave  bundle=gmem_out2 depth=__XF_DEPTH_OUT_2
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, (HEIGHT / 4), WIDTH, NPC1> _imgOutput1((HEIGHT / 4), WIDTH);
    xf::cv::Mat<XF_8UC1, (HEIGHT / 4), WIDTH, NPC1> _imgOutput2((HEIGHT / 4), WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<16 * NPC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::yuyv2iyuv<XF_16UC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput0, _imgOutput1, _imgOutput2);

    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput0, imgOutput0);
    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, (HEIGHT / 4), WIDTH, NPC1>(_imgOutput1, imgOutput1);
    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, (HEIGHT / 4), WIDTH, NPC1>(_imgOutput2, imgOutput2);
}
#endif
#if YUYV2NV12
void cvtcolor_yuyv2nv12(ap_uint<16 * NPC1>* imgInput, ap_uint<8 * NPC1>* imgOutput0, ap_uint<16 * NPC2>* imgOutput1) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_16UC1, NPC1))) / (16 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput0  offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE m_axi      port=imgOutput1  offset=slave  bundle=gmem_out1 depth=__XF_DEPTH_OUT_1
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgOutput1((HEIGHT / 2), (WIDTH / 2));

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<16 * NPC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::yuyv2nv12<XF_16UC1, XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput, _imgOutput0, _imgOutput1);

    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput0, imgOutput0);
    xf::cv::xfMat2Array<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(_imgOutput1, imgOutput1);
}
#endif
#if YUYV2NV21
void cvtcolor_yuyv2nv21(ap_uint<16 * NPC1>* imgInput, ap_uint<8 * NPC1>* imgOutput0, ap_uint<16 * NPC2>* imgOutput1) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_16UC1, NPC1))) / (16 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_1 =
        (((HEIGHT / 2)) * ((WIDTH / 2)) * (XF_PIXELWIDTH(XF_8UC2, NPC2))) / (16 * NPC2);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput0  offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE m_axi      port=imgOutput1  offset=slave  bundle=gmem_out1 depth=__XF_DEPTH_OUT_1
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput0(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2> _imgOutput1((HEIGHT / 2), (WIDTH / 2));

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<16 * NPC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::yuyv2nv21<XF_16UC1, XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(_imgInput, _imgOutput0, _imgOutput1);

    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput0, imgOutput0);
    xf::cv::xfMat2Array<16 * NPC2, XF_8UC2, (HEIGHT / 2), (WIDTH / 2), NPC2>(_imgOutput1, imgOutput1);
}
#endif
#if YUYV2RGBA
void cvtcolor_yuyv2rgba(ap_uint<16 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_16UC1, NPC1))) / (16 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC4, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC4, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<16 * NPC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::yuyv2rgba<XF_16UC1, XF_8UC4, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC4, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if YUYV2RGB
void cvtcolor_yuyv2rgb(ap_uint<16 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_16UC1, NPC1))) / (16 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<16 * NPC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::yuyv2rgb<XF_16UC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if YUYV2BGR
void cvtcolor_yuyv2bgr(ap_uint<16 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_16UC1, NPC1))) / (16 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<16 * NPC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::yuyv2bgr<XF_16UC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if YUYV2UYVY
void cvtcolor_yuyv2uyvy(ap_uint<16 * NPC1>* imgInput, ap_uint<16 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_16UC1, NPC1))) / (16 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_16UC1, NPC1))) / (16 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<16 * NPC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::yuyv2uyvy<XF_16UC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<16 * NPC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if RGB2GRAY
void cvtcolor_rgb2gray(ap_uint<32 * NPC1>* imgInput, ap_uint<8 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::rgb2gray<XF_8UC3, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if BGR2GRAY
void cvtcolor_bgr2gray(ap_uint<32 * NPC1>* imgInput, ap_uint<8 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::bgr2gray<XF_8UC3, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if GRAY2RGB
void cvtcolor_gray2rgb(ap_uint<8 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::gray2rgb<XF_8UC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if GRAY2BGR
void cvtcolor_gray2bgr(ap_uint<8 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC1, NPC1))) / (8 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<8 * NPC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::gray2bgr<XF_8UC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if RGB2XYZ
void cvtcolor_rgb2xyz(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::rgb2xyz<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if BGR2XYZ
void cvtcolor_bgr2xyz(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::bgr2xyz<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if XYZ2RGB
void cvtcolor_xyz2rgb(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::xyz2rgb<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if XYZ2BGR
void cvtcolor_xyz2bgr(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::xyz2bgr<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if RGB2YCrCb
void cvtcolor_rgb2ycrcb(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::rgb2ycrcb<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if BGR2YCrCb
void cvtcolor_bgr2ycrcb(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::bgr2ycrcb<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if YCrCb2RGB
void cvtcolor_ycrcb2rgb(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::ycrcb2rgb<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if YCrCb2BGR
void cvtcolor_ycrcb2bgr(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::ycrcb2bgr<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if RGB2HLS
void cvtcolor_rgb2hls(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::rgb2hls<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if BGR2HLS
void cvtcolor_bgr2hls(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::bgr2hls<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if HLS2RGB
void cvtcolor_hls2rgb(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::hls2rgb<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if HLS2BGR
void cvtcolor_hls2bgr(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::hls2bgr<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if RGB2HSV
void cvtcolor_rgb2hsv(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::rgb2hsv<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if BGR2HSV
void cvtcolor_bgr2hsv(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::bgr2hsv<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if HSV2RGB
void cvtcolor_hsv2rgb(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::hsv2rgb<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
#if HSV2BGR
void cvtcolor_hsv2bgr(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput) {
    static constexpr int __XF_DEPTH_INP_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);
    static constexpr int __XF_DEPTH_OUT_0 = ((HEIGHT) * (WIDTH) * (XF_PIXELWIDTH(XF_8UC3, NPC1))) / (32 * NPC1);

// clang-format off
    #pragma HLS INTERFACE m_axi      port=imgInput    offset=slave  bundle=gmem_in0  depth=__XF_DEPTH_INP_0
    #pragma HLS INTERFACE m_axi      port=imgOutput   offset=slave  bundle=gmem_out0 depth=__XF_DEPTH_OUT_0
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgInput(HEIGHT, WIDTH);
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> _imgOutput(HEIGHT, WIDTH);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput, _imgInput);

    xf::cv::hsv2bgr<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgInput, _imgOutput);

    xf::cv::xfMat2Array<32 * NPC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(_imgOutput, imgOutput);
}
#endif
