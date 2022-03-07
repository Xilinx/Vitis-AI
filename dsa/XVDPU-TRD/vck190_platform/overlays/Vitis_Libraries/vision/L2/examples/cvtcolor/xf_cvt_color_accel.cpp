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

#include "xf_cvt_color_config.h"
extern "C" {
#if RGBA2IYUV

void cvtcolor_rgba2iyuv(ap_uint<INPUT_PTR_WIDTH>* img_rgba,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_y,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_u,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_v,
                        int rows_rgba,
                        int cols_rgba,
                        int rows_y,
                        int cols_y,
                        int rows_uv,
                        int cols_uv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_rgba  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_y  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_u  	offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi     port=img_v  	offset=slave bundle=gmem5
    #pragma HLS INTERFACE s_axilite port=rows_rgba           
    #pragma HLS INTERFACE s_axilite port=cols_rgba           
    #pragma HLS INTERFACE s_axilite port=rows_y              
    #pragma HLS INTERFACE s_axilite port=cols_y              
    #pragma HLS INTERFACE s_axilite port=rows_uv              
    #pragma HLS INTERFACE s_axilite port=cols_uv            	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC4, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_rgba;
    imgInput0.cols = cols_rgba;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_y;
    imgOutput0.cols = cols_y;

    xf::cv::Mat<XF_8UC1, HEIGHT / 4, WIDTH, NPC1> imgOutput1;
// clang-format off
    #pragma HLS stream variable=imgOutput1.data depth=2
    // clang-format on
    imgOutput1.rows = rows_uv;
    imgOutput1.cols = cols_uv;

    xf::cv::Mat<XF_8UC1, HEIGHT / 4, WIDTH, NPC1> imgOutput2;
// clang-format off
    #pragma HLS stream variable=imgOutput2.data depth=2
    // clang-format on
    imgOutput2.rows = rows_uv;
    imgOutput2.cols = cols_uv;

    xf::cv::accel_utils obj_iny, obj_inu, obj_inv;
// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC4, HEIGHT, WIDTH, NPC1>(img_rgba, imgInput0);
    xf::cv::rgba2iyuv<XF_8UC4, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0, imgOutput1, imgOutput2);
    obj_iny.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, img_y);
    obj_inu.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT / 4, WIDTH, NPC1>(imgOutput1, img_u);
    obj_inv.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT / 4, WIDTH, NPC1>(imgOutput2, img_v);
}
#endif
#if RGBA2NV12

void cvtcolor_rgba2nv12(ap_uint<INPUT_PTR_WIDTH>* img_rgba,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_y,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_uv,
                        int rows_rgba,
                        int cols_rgba,
                        int rows_y,
                        int cols_y,
                        int rows_uv,
                        int cols_uv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_rgba  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_y  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_uv  	offset=slave bundle=gmem3
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=rows_rgba           
    #pragma HLS INTERFACE s_axilite port=cols_rgba           
    #pragma HLS INTERFACE s_axilite port=rows_y              
    #pragma HLS INTERFACE s_axilite port=cols_y              
    #pragma HLS INTERFACE s_axilite port=rows_uv              
    #pragma HLS INTERFACE s_axilite port=cols_uv            	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC4, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_rgba;
    imgInput0.cols = cols_rgba;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_y;
    imgOutput0.cols = cols_y;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgOutput1;
// clang-format off
    #pragma HLS stream variable=imgOutput1.data depth=2
    // clang-format on
    imgOutput1.rows = rows_uv;
    imgOutput1.cols = cols_uv;

    xf::cv::accel_utils obj_y, obj_uv;
// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC4, HEIGHT, WIDTH, NPC1>(img_rgba, imgInput0);
    xf::cv::rgba2nv12<XF_8UC4, XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(imgInput0, imgOutput0, imgOutput1);
    obj_y.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, img_y);
    obj_uv.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(imgOutput1, img_uv);
}
#endif
#if RGBA2NV21

void cvtcolor_rgba2nv21(ap_uint<INPUT_PTR_WIDTH>* img_rgba,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_y,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_uv,
                        int rows_rgba,
                        int cols_rgba,
                        int rows_y,
                        int cols_y,
                        int rows_uv,
                        int cols_uv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_rgba  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_y  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_uv  	offset=slave bundle=gmem3
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=rows_rgba           
    #pragma HLS INTERFACE s_axilite port=cols_rgba           
    #pragma HLS INTERFACE s_axilite port=rows_y              
    #pragma HLS INTERFACE s_axilite port=cols_y              
    #pragma HLS INTERFACE s_axilite port=rows_uv              
    #pragma HLS INTERFACE s_axilite port=cols_uv            	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC4, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_rgba;
    imgInput0.cols = cols_rgba;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_y;
    imgOutput0.cols = cols_y;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgOutput1;
// clang-format off
    #pragma HLS stream variable=imgOutput1.data depth=2
    // clang-format on
    imgOutput1.rows = rows_uv;
    imgOutput1.cols = cols_uv;

    xf::cv::accel_utils obj_y, obj_uv;
// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC4, HEIGHT, WIDTH, NPC1>(img_rgba, imgInput0);
    xf::cv::rgba2nv21<XF_8UC4, XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(imgInput0, imgOutput0, imgOutput1);
    obj_y.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, img_y);
    obj_uv.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(imgOutput1, img_uv);
}
#endif
#if RGBA2YUV4

void cvtcolor_rgba2yuv4(ap_uint<INPUT_PTR_WIDTH>* img_rgba,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_y,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_u,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_v,
                        int rows_rgba,
                        int cols_rgba,
                        int rows_y,
                        int cols_y,
                        int rows_uv,
                        int cols_uv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_rgba  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_y  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_u  	offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi     port=img_v  	offset=slave bundle=gmem3
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=rows_rgba           
    #pragma HLS INTERFACE s_axilite port=cols_rgba           
    #pragma HLS INTERFACE s_axilite port=rows_y              
    #pragma HLS INTERFACE s_axilite port=cols_y              
    #pragma HLS INTERFACE s_axilite port=rows_uv              
    #pragma HLS INTERFACE s_axilite port=cols_uv            	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC4, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_rgba;
    imgInput0.cols = cols_rgba;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_y;
    imgOutput0.cols = cols_y;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput1;
// clang-format off
    #pragma HLS stream variable=imgOutput1.data depth=2
    // clang-format on
    imgOutput1.rows = rows_uv;
    imgOutput1.cols = cols_uv;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput2;
// clang-format off
    #pragma HLS stream variable=imgOutput2.data depth=2
    // clang-format on
    imgOutput2.rows = rows_uv;
    imgOutput2.cols = cols_uv;
    xf::cv::accel_utils obj_y, obj_u, obj_v;
// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC4, HEIGHT, WIDTH, NPC1>(img_rgba, imgInput0);
    xf::cv::rgba2yuv4<XF_8UC4, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0, imgOutput1, imgOutput2);
    obj_y.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, img_y);
    obj_u.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput1, img_u);
    obj_v.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput2, img_v);
}
#endif

#if RGB2IYUV

void cvtcolor_rgb2iyuv(ap_uint<INPUT_PTR_WIDTH>* img_rgb,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_y,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_u,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_v,
                       int rows_rgb,
                       int cols_rgb,
                       int rows_y,
                       int cols_y,
                       int rows_uv,
                       int cols_uv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_y  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_u  	offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi     port=img_v  	offset=slave bundle=gmem5
    #pragma HLS INTERFACE s_axilite port=rows_rgb           
    #pragma HLS INTERFACE s_axilite port=cols_rgb           
    #pragma HLS INTERFACE s_axilite port=rows_y              
    #pragma HLS INTERFACE s_axilite port=cols_y              
    #pragma HLS INTERFACE s_axilite port=rows_uv              
    #pragma HLS INTERFACE s_axilite port=cols_uv            	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_rgb;
    imgInput0.cols = cols_rgb;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_y;
    imgOutput0.cols = cols_y;

    xf::cv::Mat<XF_8UC1, HEIGHT / 4, WIDTH, NPC1> imgOutput1;
// clang-format off
    #pragma HLS stream variable=imgOutput1.data depth=2
    // clang-format on
    imgOutput1.rows = rows_uv;
    imgOutput1.cols = cols_uv;

    xf::cv::Mat<XF_8UC1, HEIGHT / 4, WIDTH, NPC1> imgOutput2;
// clang-format off
    #pragma HLS stream variable=imgOutput2.data depth=2
    // clang-format on
    imgOutput2.rows = rows_uv;
    imgOutput2.cols = cols_uv;

    xf::cv::accel_utils obj_y, obj_u, obj_v;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_rgb, imgInput0);
    xf::cv::rgb2iyuv<XF_8UC3, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0, imgOutput1, imgOutput2);
    obj_y.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, img_y);
    obj_u.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT / 4, WIDTH, NPC1>(imgOutput1, img_u);
    obj_v.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT / 4, WIDTH, NPC1>(imgOutput2, img_v);
}
#endif
#if RGB2NV12

void cvtcolor_rgb2nv12(ap_uint<INPUT_PTR_WIDTH>* img_rgb,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_y,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_uv,
                       int rows_rgb,
                       int cols_rgb,
                       int rows_y,
                       int cols_y,
                       int rows_uv,
                       int cols_uv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_y  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_uv  	offset=slave bundle=gmem3
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=rows_rgb           
    #pragma HLS INTERFACE s_axilite port=cols_rgb           
    #pragma HLS INTERFACE s_axilite port=rows_y              
    #pragma HLS INTERFACE s_axilite port=cols_y              
    #pragma HLS INTERFACE s_axilite port=rows_uv              
    #pragma HLS INTERFACE s_axilite port=cols_uv            	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_rgb;
    imgInput0.cols = cols_rgb;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_y;
    imgOutput0.cols = cols_y;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgOutput1;
// clang-format off
    #pragma HLS stream variable=imgOutput1.data depth=2
    // clang-format on
    imgOutput1.rows = rows_uv;
    imgOutput1.cols = cols_uv;

    xf::cv::accel_utils obj_y, obj_uv;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_rgb, imgInput0);
    xf::cv::rgb2nv12<XF_8UC3, XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(imgInput0, imgOutput0, imgOutput1);
    obj_y.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, img_y);
    obj_uv.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(imgOutput1, img_uv);
}
#endif
#if BGR2NV12

void cvtcolor_bgr2nv12(ap_uint<INPUT_PTR_WIDTH>* img_rgb,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_y,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_uv,
                       int rows_rgb,
                       int cols_rgb,
                       int rows_y,
                       int cols_y,
                       int rows_uv,
                       int cols_uv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_y  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_uv  	offset=slave bundle=gmem3
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=rows_rgb           
    #pragma HLS INTERFACE s_axilite port=cols_rgb           
    #pragma HLS INTERFACE s_axilite port=rows_y              
    #pragma HLS INTERFACE s_axilite port=cols_y              
    #pragma HLS INTERFACE s_axilite port=rows_uv              
    #pragma HLS INTERFACE s_axilite port=cols_uv            	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_rgb;
    imgInput0.cols = cols_rgb;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_y;
    imgOutput0.cols = cols_y;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgOutput1;
// clang-format off
    #pragma HLS stream variable=imgOutput1.data depth=2
    // clang-format on
    imgOutput1.rows = rows_uv;
    imgOutput1.cols = cols_uv;

    xf::cv::accel_utils obj_y, obj_uv;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_rgb, imgInput0);
    xf::cv::bgr2nv12<XF_8UC3, XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(imgInput0, imgOutput0, imgOutput1);
    obj_y.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, img_y);
    obj_uv.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(imgOutput1, img_uv);
}
#endif
#if RGB2NV21

void cvtcolor_rgb2nv21(ap_uint<INPUT_PTR_WIDTH>* img_rgb,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_y,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_uv,
                       int rows_rgb,
                       int cols_rgb,
                       int rows_y,
                       int cols_y,
                       int rows_uv,
                       int cols_uv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_y  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_uv  	offset=slave bundle=gmem3
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=rows_rgb           
    #pragma HLS INTERFACE s_axilite port=cols_rgb           
    #pragma HLS INTERFACE s_axilite port=rows_y              
    #pragma HLS INTERFACE s_axilite port=cols_y              
    #pragma HLS INTERFACE s_axilite port=rows_uv              
    #pragma HLS INTERFACE s_axilite port=cols_uv            	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_rgb;
    imgInput0.cols = cols_rgb;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_y;
    imgOutput0.cols = cols_y;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgOutput1;
// clang-format off
    #pragma HLS stream variable=imgOutput1.data depth=2
    // clang-format on
    imgOutput1.rows = rows_uv;
    imgOutput1.cols = cols_uv;

    xf::cv::accel_utils obj_y, obj_uv;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_rgb, imgInput0);
    xf::cv::rgb2nv21<XF_8UC3, XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(imgInput0, imgOutput0, imgOutput1);
    obj_y.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, img_y);
    obj_uv.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(imgOutput1, img_uv);
}
#endif
#if BGR2NV21

void cvtcolor_bgr2nv21(ap_uint<INPUT_PTR_WIDTH>* img_rgb,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_y,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_uv,
                       int rows_rgb,
                       int cols_rgb,
                       int rows_y,
                       int cols_y,
                       int rows_uv,
                       int cols_uv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_y  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_uv  	offset=slave bundle=gmem3
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=rows_rgb           
    #pragma HLS INTERFACE s_axilite port=cols_rgb           
    #pragma HLS INTERFACE s_axilite port=rows_y              
    #pragma HLS INTERFACE s_axilite port=cols_y              
    #pragma HLS INTERFACE s_axilite port=rows_uv              
    #pragma HLS INTERFACE s_axilite port=cols_uv            	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_rgb;
    imgInput0.cols = cols_rgb;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_y;
    imgOutput0.cols = cols_y;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgOutput1;
// clang-format off
    #pragma HLS stream variable=imgOutput1.data depth=2
    // clang-format on
    imgOutput1.rows = rows_uv;
    imgOutput1.cols = cols_uv;

    xf::cv::accel_utils obj_y, obj_uv;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_rgb, imgInput0);
    xf::cv::bgr2nv21<XF_8UC3, XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(imgInput0, imgOutput0, imgOutput1);
    obj_y.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, img_y);
    obj_uv.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(imgOutput1, img_uv);
}
#endif
#if RGB2YUV4

void cvtcolor_rgb2yuv4(ap_uint<INPUT_PTR_WIDTH>* img_rgb,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_y,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_u,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_v,
                       int rows_rgb,
                       int cols_rgb,
                       int rows_y,
                       int cols_y,
                       int rows_uv,
                       int cols_uv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_y  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_u  	offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi     port=img_v  	offset=slave bundle=gmem4
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=rows_rgb           
    #pragma HLS INTERFACE s_axilite port=cols_rgb           
    #pragma HLS INTERFACE s_axilite port=rows_y             
    #pragma HLS INTERFACE s_axilite port=cols_y             
    #pragma HLS INTERFACE s_axilite port=rows_uv            
    #pragma HLS INTERFACE s_axilite port=cols_uv            
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_rgb;
    imgInput0.cols = cols_rgb;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_y;
    imgOutput0.cols = cols_y;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput1;
// clang-format off
    #pragma HLS stream variable=imgOutput1.data depth=2
    // clang-format on
    imgOutput1.rows = rows_uv;
    imgOutput1.cols = cols_uv;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput2;
// clang-format off
    #pragma HLS stream variable=imgOutput2.data depth=2
    // clang-format on
    imgOutput2.rows = rows_uv;
    imgOutput2.cols = cols_uv;

    xf::cv::accel_utils obj_y, obj_u, obj_v;
// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_rgb, imgInput0);
    xf::cv::rgb2yuv4<XF_8UC3, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0, imgOutput1, imgOutput2);
    obj_y.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, img_y);
    obj_u.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput1, img_u);
    obj_v.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput2, img_v);
}
#endif

#if RGB2YUYV

void cvtcolor_rgb2yuyv(ap_uint<INPUT_PTR_WIDTH>* img_rgb, ap_uint<OUTPUT_PTR_WIDTH>* img_yuyv, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_yuyv  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;
    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_rgb, imgInput0);
    xf::cv::rgb2yuyv<XF_8UC3, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, img_yuyv);
}
#endif
#if BGR2YUYV

void cvtcolor_bgr2yuyv(ap_uint<INPUT_PTR_WIDTH>* img_rgb, ap_uint<OUTPUT_PTR_WIDTH>* img_yuyv, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_yuyv  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;
    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_rgb, imgInput0);
    xf::cv::bgr2yuyv<XF_8UC3, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, img_yuyv);
}
#endif
#if RGB2UYVY

void cvtcolor_rgb2uyvy(ap_uint<INPUT_PTR_WIDTH>* img_rgb, ap_uint<OUTPUT_PTR_WIDTH>* img_uyvy, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_uyvy  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;
    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_rgb, imgInput0);
    xf::cv::rgb2uyvy<XF_8UC3, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, img_uyvy);
}
#endif
#if BGR2UYVY

void cvtcolor_bgr2uyvy(ap_uint<INPUT_PTR_WIDTH>* img_rgb, ap_uint<OUTPUT_PTR_WIDTH>* img_uyvy, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_uyvy  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;
    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_rgb, imgInput0);
    xf::cv::bgr2uyvy<XF_8UC3, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, img_uyvy);
}
#endif
#if IYUV2NV12
void cvtcolor_iyuv2nv12(ap_uint<INPUT_PTR_WIDTH>* img_y,
                        ap_uint<INPUT_PTR_WIDTH>* img_u,
                        ap_uint<INPUT_PTR_WIDTH>* img_v,
                        ap_uint<OUTPUT_PTR_WIDTH>* out_img_y,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_uv,
                        int rows,
                        int cols,
                        int rows_u,
                        int cols_u,
                        int rows_v,
                        int cols_v,
                        int rows_uv,
                        int cols_uv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_y  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_u  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_v  	offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi     port=out_img_y  offset=slave bundle=gmem5
    #pragma HLS INTERFACE m_axi     port=img_uv  	offset=slave bundle=gmem6
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=rows_u              
    #pragma HLS INTERFACE s_axilite port=cols_u            	 
    #pragma HLS INTERFACE s_axilite port=rows_v              
    #pragma HLS INTERFACE s_axilite port=cols_v            	 
    #pragma HLS INTERFACE s_axilite port=rows_uv             
    #pragma HLS INTERFACE s_axilite port=cols_uv
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;

    xf::cv::Mat<XF_8UC1, HEIGHT / 4, WIDTH, NPC1> imgInput1;
// clang-format off
    #pragma HLS stream variable=imgInput1.data depth=2
    // clang-format on
    imgInput1.rows = rows_u;
    imgInput1.cols = cols_u;

    xf::cv::Mat<XF_8UC1, HEIGHT / 4, WIDTH, NPC1> imgInput2;
// clang-format off
    #pragma HLS stream variable=imgInput2.data depth=2
    // clang-format on
    imgInput2.rows = rows_v;
    imgInput2.cols = cols_v;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgOutput1;
// clang-format off
    #pragma HLS stream variable=imgOutput1.data depth=2
    // clang-format on
    imgOutput1.rows = rows_uv;
    imgOutput1.cols = cols_uv;

    xf::cv::accel_utils obj_iny, obj_inu, obj_inv, obj_y, obj_uv;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    obj_iny.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(img_y, imgInput0);
    obj_inu.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT / 4, WIDTH, NPC1>(img_u, imgInput1);
    obj_inv.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT / 4, WIDTH, NPC1>(img_v, imgInput2);

    xf::cv::iyuv2nv12<XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(imgInput0, imgInput1, imgInput2, imgOutput0,
                                                                   imgOutput1);

    obj_y.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, out_img_y);
    obj_uv.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(imgOutput1, img_uv);
}
#endif
#if IYUV2RGBA
void cvtcolor_iyuv2rgba(ap_uint<INPUT_PTR_WIDTH>* img_y,
                        ap_uint<INPUT_PTR_WIDTH>* img_u,
                        ap_uint<INPUT_PTR_WIDTH>* img_v,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_rgba,
                        int rows,
                        int cols,
                        int rows_u,
                        int cols_u,
                        int rows_v,
                        int cols_v) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_y  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_u  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_v  	offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi     port=img_rgba  offset=slave bundle=gmem5
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=rows_u              
    #pragma HLS INTERFACE s_axilite port=cols_u            	 
    #pragma HLS INTERFACE s_axilite port=rows_v              
    #pragma HLS INTERFACE s_axilite port=cols_v
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;

    xf::cv::Mat<XF_8UC1, HEIGHT / 4, WIDTH, NPC1> imgInput1;
// clang-format off
    #pragma HLS stream variable=imgInput1.data depth=2
    // clang-format on
    imgInput1.rows = rows_u;
    imgInput1.cols = cols_u;

    xf::cv::Mat<XF_8UC1, HEIGHT / 4, WIDTH, NPC1> imgInput2;
// clang-format off
    #pragma HLS stream variable=imgInput2.data depth=2
    // clang-format on
    imgInput2.rows = rows_v;
    imgInput2.cols = cols_v;

    xf::cv::Mat<XF_8UC4, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;
    xf::cv::accel_utils obj_iny, obj_inu, obj_inv;
// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    obj_iny.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(img_y, imgInput0);
    obj_inu.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT / 4, WIDTH, NPC1>(img_u, imgInput1);
    obj_inv.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT / 4, WIDTH, NPC1>(img_v, imgInput2);
    xf::cv::iyuv2rgba<XF_8UC1, XF_8UC4, HEIGHT, WIDTH, NPC1>(imgInput0, imgInput1, imgInput2, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC4, HEIGHT, WIDTH, NPC1>(imgOutput0, img_rgba);
}

#endif
#if IYUV2RGB
void cvtcolor_iyuv2rgb(ap_uint<INPUT_PTR_WIDTH>* img_y,
                       ap_uint<INPUT_PTR_WIDTH>* img_u,
                       ap_uint<INPUT_PTR_WIDTH>* img_v,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_rgb,
                       int rows,
                       int cols,
                       int rows_u,
                       int cols_u,
                       int rows_v,
                       int cols_v) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_y  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_u  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_v  	offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi     port=img_rgb  offset=slave bundle=gmem5
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=rows_u              
    #pragma HLS INTERFACE s_axilite port=cols_u            	 
    #pragma HLS INTERFACE s_axilite port=rows_v              
    #pragma HLS INTERFACE s_axilite port=cols_v
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;

    xf::cv::Mat<XF_8UC1, HEIGHT / 4, WIDTH, NPC1> imgInput1;
// clang-format off
    #pragma HLS stream variable=imgInput1.data depth=2
    // clang-format on
    imgInput1.rows = rows_u;
    imgInput1.cols = cols_u;

    xf::cv::Mat<XF_8UC1, HEIGHT / 4, WIDTH, NPC1> imgInput2;
// clang-format off
    #pragma HLS stream variable=imgInput2.data depth=2
    // clang-format on
    imgInput2.rows = rows_v;
    imgInput2.cols = cols_v;

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

    xf::cv::accel_utils obj_iny, obj_inu, obj_inv;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    obj_iny.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(img_y, imgInput0);
    obj_inu.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT / 4, WIDTH, NPC1>(img_u, imgInput1);
    obj_inv.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT / 4, WIDTH, NPC1>(img_v, imgInput2);

    xf::cv::iyuv2rgb<XF_8UC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput0, imgInput1, imgInput2, imgOutput0);

    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_rgb);
}
#endif

#if IYUV2YUV4
void cvtcolor_iyuv2yuv4(ap_uint<INPUT_PTR_WIDTH>* img_y,
                        ap_uint<INPUT_PTR_WIDTH>* img_u,
                        ap_uint<INPUT_PTR_WIDTH>* img_v,
                        ap_uint<OUTPUT_PTR_WIDTH>* out_img_y,
                        ap_uint<OUTPUT_PTR_WIDTH>* out_img_u,
                        ap_uint<OUTPUT_PTR_WIDTH>* out_img_v,
                        int rows,
                        int cols,
                        int rows_u,
                        int cols_u,
                        int rows_v,
                        int cols_v) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_y  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_u  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_v  	offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi     port=out_img_y  offset=slave bundle=gmem4
    #pragma HLS INTERFACE m_axi     port=out_img_u  offset=slave bundle=gmem5
    #pragma HLS INTERFACE m_axi     port=out_img_v  offset=slave bundle=gmem6
// clang-format on

// clang-format off
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=rows_u              
    #pragma HLS INTERFACE s_axilite port=cols_u            	 
    #pragma HLS INTERFACE s_axilite port=rows_v              
    #pragma HLS INTERFACE s_axilite port=cols_v
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;

    xf::cv::Mat<XF_8UC1, HEIGHT / 4, WIDTH, NPC1> imgInput1;
// clang-format off
    #pragma HLS stream variable=imgInput1.data depth=2
    // clang-format on
    imgInput1.rows = rows_u;
    imgInput1.cols = cols_u;

    xf::cv::Mat<XF_8UC1, HEIGHT / 4, WIDTH, NPC1> imgInput2;
// clang-format off
    #pragma HLS stream variable=imgInput2.data depth=2
    // clang-format on
    imgInput2.rows = rows_v;
    imgInput2.cols = cols_v;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput1;
// clang-format off
    #pragma HLS stream variable=imgOutput1.data depth=2
    // clang-format on
    imgOutput1.rows = rows;
    imgOutput1.cols = cols;
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput2;
// clang-format off
    #pragma HLS stream variable=imgOutput2.data depth=2
    // clang-format on
    imgOutput2.rows = rows;
    imgOutput2.cols = cols;

    xf::cv::accel_utils obj_iny, obj_inu, obj_inv, obj_outy, obj_outu, obj_outv;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    obj_iny.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(img_y, imgInput0);
    obj_inu.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT / 4, WIDTH, NPC1>(img_u, imgInput1);
    obj_inv.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT / 4, WIDTH, NPC1>(img_v, imgInput2);
    xf::cv::iyuv2yuv4<XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, imgInput1, imgInput2, imgOutput0, imgOutput1,
                                                    imgOutput2);
    obj_outy.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, out_img_y);
    obj_outu.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput1, out_img_u);
    obj_outv.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput2, out_img_v);
}

#endif

#if NV122IYUV

void cvtcolor_nv122iyuv(ap_uint<INPUT_PTR_WIDTH>* inimg_y,
                        ap_uint<INPUT_PTR_WIDTH>* inimg_uv,
                        ap_uint<OUTPUT_PTR_WIDTH>* outimg_y,
                        ap_uint<OUTPUT_PTR_WIDTH>* outimg_u,
                        ap_uint<OUTPUT_PTR_WIDTH>* outimg_v,
                        int rows_imgy,
                        int cols_imgy,
                        int rows_imguv,
                        int cols_imguv,
                        int rows_outy,
                        int cols_outy,
                        int rows_outuv,
                        int cols_outuv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=inimg_y  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=inimg_uv  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=outimg_y  	offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi     port=outimg_u  	offset=slave bundle=gmem4
    #pragma HLS INTERFACE m_axi     port=outimg_v  	offset=slave bundle=gmem5
// clang-format on

// clang-format off
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=rows_imgy           
    #pragma HLS INTERFACE s_axilite port=cols_imgy           
    #pragma HLS INTERFACE s_axilite port=rows_imguv          
    #pragma HLS INTERFACE s_axilite port=cols_imguv          
    #pragma HLS INTERFACE s_axilite port=rows_outy           
    #pragma HLS INTERFACE s_axilite port=cols_outy           
    #pragma HLS INTERFACE s_axilite port=rows_outuv           
    #pragma HLS INTERFACE s_axilite port=cols_outuv
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_imgy;
    imgInput0.cols = cols_imgy;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgInput1;
// clang-format off
    #pragma HLS stream variable=imgInput1.data depth=2
    // clang-format on
    imgInput1.rows = rows_imguv;
    imgInput1.cols = cols_imguv;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_outy;
    imgOutput0.cols = cols_outy;

    xf::cv::Mat<XF_8UC1, HEIGHT / 4, WIDTH, NPC1> imgOutput1;
// clang-format off
    #pragma HLS stream variable=imgOutput1.data depth=2
    // clang-format on
    imgOutput1.rows = rows_outuv;
    imgOutput1.cols = cols_outuv;

    xf::cv::Mat<XF_8UC1, HEIGHT / 4, WIDTH, NPC1> imgOutput2;
// clang-format off
    #pragma HLS stream variable=imgOutput2.data depth=2
    // clang-format on
    imgOutput2.rows = rows_outuv;
    imgOutput2.cols = cols_outuv;
    xf::cv::accel_utils obj_iny, obj_inuv, obj_outy, obj_outu, obj_outv;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    obj_iny.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(inimg_y, imgInput0);
    obj_inuv.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(inimg_uv, imgInput1);

    xf::cv::nv122iyuv<XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(imgInput0, imgInput1, imgOutput0, imgOutput1,
                                                                   imgOutput2);

    obj_outy.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, outimg_y);
    obj_outu.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT / 4, WIDTH, NPC1>(imgOutput1, outimg_u);
    obj_outv.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT / 4, WIDTH, NPC1>(imgOutput2, outimg_v);
}
#endif

#if NV122RGBA

void cvtcolor_nv122rgba(ap_uint<INPUT_PTR_WIDTH>* inimg_y,
                        ap_uint<INPUT_PTR_WIDTH>* inimg_uv,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_rgb,
                        int rows_imgy,
                        int cols_imgy,
                        int rows_imguv,
                        int cols_imguv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=inimg_y  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=inimg_uv  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem3
    #pragma HLS INTERFACE s_axilite port=rows_imgy           
    #pragma HLS INTERFACE s_axilite port=cols_imgy           
    #pragma HLS INTERFACE s_axilite port=rows_imguv          
    #pragma HLS INTERFACE s_axilite port=cols_imguv          
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_imgy;
    imgInput0.cols = cols_imgy;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgInput1;
// clang-format off
    #pragma HLS stream variable=imgInput1.data depth=2
    // clang-format on
    imgInput1.rows = rows_imguv;
    imgInput1.cols = cols_imguv;

    xf::cv::Mat<XF_8UC4, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_imgy;
    imgOutput0.cols = cols_imgy;

    xf::cv::accel_utils obj_iny, obj_inuv;
// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    obj_iny.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(inimg_y, imgInput0);
    obj_inuv.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(inimg_uv, imgInput1);
    xf::cv::nv122rgba<XF_8UC1, XF_8UC2, XF_8UC4, HEIGHT, WIDTH, NPC1, NPC2>(imgInput0, imgInput1, imgOutput0);

    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC4, HEIGHT, WIDTH, NPC1>(imgOutput0, img_rgb);
}
#endif

#if NV122YUV4
void cvtcolor_nv122yuv4(ap_uint<INPUT_PTR_WIDTH>* inimg_y,
                        ap_uint<INPUT_PTR_WIDTH>* inimg_uv,
                        ap_uint<OUTPUT_PTR_WIDTH>* outimg_y,
                        ap_uint<OUTPUT_PTR_WIDTH>* outimg_u,
                        ap_uint<OUTPUT_PTR_WIDTH>* outimg_v,
                        int rows_imgy,
                        int cols_imgy,
                        int rows_imguv,
                        int cols_imguv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=inimg_y  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=inimg_uv  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=outimg_y  	offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi     port=outimg_u  	offset=slave bundle=gmem4
    #pragma HLS INTERFACE m_axi     port=outimg_v  	offset=slave bundle=gmem5
    #pragma HLS INTERFACE s_axilite port=rows_imgy           
    #pragma HLS INTERFACE s_axilite port=cols_imgy           
    #pragma HLS INTERFACE s_axilite port=rows_imguv          
    #pragma HLS INTERFACE s_axilite port=cols_imguv          
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_imgy;
    imgInput0.cols = cols_imgy;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgInput1;
// clang-format off
    #pragma HLS stream variable=imgInput1.data depth=2
    // clang-format on
    imgInput1.rows = rows_imguv;
    imgInput1.cols = cols_imguv;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_imgy;
    imgOutput0.cols = cols_imgy;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput1;
// clang-format off
    #pragma HLS stream variable=imgOutput1.data depth=2
    // clang-format on
    imgOutput1.rows = rows_imgy;
    imgOutput1.cols = cols_imgy;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput2;
// clang-format off
    #pragma HLS stream variable=imgOutput2.data depth=2
    // clang-format on
    imgOutput2.rows = rows_imgy;
    imgOutput2.cols = cols_imgy;

    xf::cv::accel_utils obj_iny, obj_inuv, obj_outy, obj_outu, obj_outv;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    obj_iny.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(inimg_y, imgInput0);
    obj_inuv.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(inimg_uv, imgInput1);

    xf::cv::nv122yuv4<XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(imgInput0, imgInput1, imgOutput0, imgOutput1,
                                                                   imgOutput2);

    obj_outy.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, outimg_y);
    obj_outu.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput1, outimg_u);
    obj_outv.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput2, outimg_v);
}
#endif
#if NV122RGB

void cvtcolor_nv122rgb(ap_uint<INPUT_PTR_WIDTH>* inimg_y,
                       ap_uint<INPUT_PTR_WIDTH>* inimg_uv,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_rgb,
                       int rows_imgy,
                       int cols_imgy,
                       int rows_imguv,
                       int cols_imguv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=inimg_y  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=inimg_uv  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem3
    #pragma HLS INTERFACE s_axilite port=rows_imgy           
    #pragma HLS INTERFACE s_axilite port=cols_imgy           
    #pragma HLS INTERFACE s_axilite port=rows_imguv          
    #pragma HLS INTERFACE s_axilite port=cols_imguv          
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_imgy;
    imgInput0.cols = cols_imgy;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgInput1;
// clang-format off
    #pragma HLS stream variable=imgInput1.data depth=2
    // clang-format on
    imgInput1.rows = rows_imguv;
    imgInput1.cols = cols_imguv;

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_imgy;
    imgOutput0.cols = cols_imgy;

    xf::cv::accel_utils obj_iny, obj_inuv;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    obj_iny.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(inimg_y, imgInput0);
    obj_inuv.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(inimg_uv, imgInput1);
    xf::cv::nv122rgb<XF_8UC1, XF_8UC2, XF_8UC3, HEIGHT, WIDTH, NPC1, NPC2>(imgInput0, imgInput1, imgOutput0);

    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_rgb);
}
#endif

#if NV122BGR

void cvtcolor_nv122bgr(ap_uint<INPUT_PTR_WIDTH>* inimg_y,
                       ap_uint<INPUT_PTR_WIDTH>* inimg_uv,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_rgb,
                       int rows_imgy,
                       int cols_imgy,
                       int rows_imguv,
                       int cols_imguv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=inimg_y  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=inimg_uv  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem3
    #pragma HLS INTERFACE s_axilite port=rows_imgy           
    #pragma HLS INTERFACE s_axilite port=cols_imgy           
    #pragma HLS INTERFACE s_axilite port=rows_imguv          
    #pragma HLS INTERFACE s_axilite port=cols_imguv          
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_imgy;
    imgInput0.cols = cols_imgy;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgInput1;
// clang-format off
    #pragma HLS stream variable=imgInput1.data depth=2
    // clang-format on
    imgInput1.rows = rows_imguv;
    imgInput1.cols = cols_imguv;

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_imgy;
    imgOutput0.cols = cols_imgy;

    xf::cv::accel_utils obj_iny, obj_inuv;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    obj_iny.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(inimg_y, imgInput0);
    obj_inuv.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(inimg_uv, imgInput1);
    xf::cv::nv122bgr<XF_8UC1, XF_8UC2, XF_8UC3, HEIGHT, WIDTH, NPC1, NPC2>(imgInput0, imgInput1, imgOutput0);

    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_rgb);
}
#endif
#if NV212BGR

void cvtcolor_nv212bgr(ap_uint<INPUT_PTR_WIDTH>* inimg_y,
                       ap_uint<INPUT_PTR_WIDTH>* inimg_uv,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_rgb,
                       int rows_imgy,
                       int cols_imgy,
                       int rows_imguv,
                       int cols_imguv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=inimg_y  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=inimg_uv  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem3
    #pragma HLS INTERFACE s_axilite port=rows_imgy           
    #pragma HLS INTERFACE s_axilite port=cols_imgy           
    #pragma HLS INTERFACE s_axilite port=rows_imguv          
    #pragma HLS INTERFACE s_axilite port=cols_imguv          
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_imgy;
    imgInput0.cols = cols_imgy;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgInput1;
// clang-format off
    #pragma HLS stream variable=imgInput1.data depth=2
    // clang-format on
    imgInput1.rows = rows_imguv;
    imgInput1.cols = cols_imguv;

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_imgy;
    imgOutput0.cols = cols_imgy;

    xf::cv::accel_utils obj_iny, obj_inuv;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    obj_iny.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(inimg_y, imgInput0);
    obj_inuv.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(inimg_uv, imgInput1);
    xf::cv::nv212bgr<XF_8UC1, XF_8UC2, XF_8UC3, HEIGHT, WIDTH, NPC1, NPC2>(imgInput0, imgInput1, imgOutput0);

    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_rgb);
}
#endif
#if NV122YUYV

void cvtcolor_nv122yuyv(ap_uint<INPUT_PTR_WIDTH>* inimg_y,
                        ap_uint<INPUT_PTR_WIDTH>* inimg_uv,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_yuyv,
                        int rows_imgy,
                        int cols_imgy,
                        int rows_imguv,
                        int cols_imguv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=inimg_y  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=inimg_uv  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_yuyv  	offset=slave bundle=gmem3
    #pragma HLS INTERFACE s_axilite port=rows_imgy           
    #pragma HLS INTERFACE s_axilite port=cols_imgy           
    #pragma HLS INTERFACE s_axilite port=rows_imguv          
    #pragma HLS INTERFACE s_axilite port=cols_imguv          
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_imgy;
    imgInput0.cols = cols_imgy;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgInput1;
// clang-format off
    #pragma HLS stream variable=imgInput1.data depth=2
    // clang-format on
    imgInput1.rows = rows_imguv;
    imgInput1.cols = cols_imguv;

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_imgy;
    imgOutput0.cols = cols_imgy;

    xf::cv::accel_utils obj_iny, obj_inuv;
// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    obj_iny.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(inimg_y, imgInput0);
    obj_inuv.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(inimg_uv, imgInput1);
    xf::cv::nv122yuyv<XF_8UC1, XF_8UC2, XF_16UC1, HEIGHT, WIDTH, NPC1, NPC2>(imgInput0, imgInput1, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, img_yuyv);
}
#endif
#if NV122NV21

void cvtcolor_nv122nv21(ap_uint<INPUT_PTR_WIDTH>* inimg_y,
                        ap_uint<INPUT_PTR_WIDTH>* inimg_uv,
                        ap_uint<OUTPUT_PTR_WIDTH>* out_y,
                        ap_uint<OUTPUT_PTR_WIDTH>* out_uv,
                        int rows_y,
                        int cols_y,
                        int rows_uv,
                        int cols_uv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=inimg_y  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=inimg_uv  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=out_y  	offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi     port=out_uv  	offset=slave bundle=gmem4
    #pragma HLS INTERFACE s_axilite port=rows_y           
    #pragma HLS INTERFACE s_axilite port=cols_y           
    #pragma HLS INTERFACE s_axilite port=rows_uv          
    #pragma HLS INTERFACE s_axilite port=cols_uv          
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_y;
    imgInput0.cols = cols_y;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgInput1;
// clang-format off
    #pragma HLS stream variable=imgInput1.data depth=2
    // clang-format on
    imgInput1.rows = rows_uv;
    imgInput1.cols = cols_uv;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_y;
    imgOutput0.cols = cols_y;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgOutput1;
// clang-format off
    #pragma HLS stream variable=imgOutput1.data depth=2
    // clang-format on
    imgOutput1.rows = rows_uv;
    imgOutput1.cols = cols_uv;

    xf::cv::accel_utils obj_iny, obj_inuv, obj_outy, obj_outuv;
// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    obj_iny.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(inimg_y, imgInput0);
    obj_inuv.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(inimg_uv, imgInput1);
    xf::cv::nv122nv21<XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(imgInput0, imgInput1, imgOutput0, imgOutput1);
    obj_outy.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, out_y);
    obj_outuv.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(imgOutput1, out_uv);
}
#endif
#if NV212NV12

void cvtcolor_nv212nv12(ap_uint<INPUT_PTR_WIDTH>* inimg_y,
                        ap_uint<INPUT_PTR_WIDTH>* inimg_uv,
                        ap_uint<OUTPUT_PTR_WIDTH>* out_y,
                        ap_uint<OUTPUT_PTR_WIDTH>* out_uv,
                        int rows_y,
                        int cols_y,
                        int rows_uv,
                        int cols_uv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=inimg_y  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=inimg_uv  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=out_y  	offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi     port=out_uv  	offset=slave bundle=gmem4
    #pragma HLS INTERFACE s_axilite port=rows_y           
    #pragma HLS INTERFACE s_axilite port=cols_y           
    #pragma HLS INTERFACE s_axilite port=rows_uv          
    #pragma HLS INTERFACE s_axilite port=cols_uv          
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_y;
    imgInput0.cols = cols_y;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgInput1;
// clang-format off
    #pragma HLS stream variable=imgInput1.data depth=2
    // clang-format on
    imgInput1.rows = rows_uv;
    imgInput1.cols = cols_uv;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_y;
    imgOutput0.cols = cols_y;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgOutput1;
// clang-format off
    #pragma HLS stream variable=imgOutput1.data depth=2
    // clang-format on
    imgOutput1.rows = rows_uv;
    imgOutput1.cols = cols_uv;

    xf::cv::accel_utils obj_iny, obj_inuv, obj_outy, obj_outuv;
// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    obj_iny.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(inimg_y, imgInput0);
    obj_inuv.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(inimg_uv, imgInput1);
    xf::cv::nv212nv12<XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(imgInput0, imgInput1, imgOutput0, imgOutput1);
    obj_outy.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, out_y);
    obj_outuv.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(imgOutput1, out_uv);
}
#endif

#if NV212YUYV

void cvtcolor_nv212yuyv(ap_uint<INPUT_PTR_WIDTH>* inimg_y,
                        ap_uint<INPUT_PTR_WIDTH>* inimg_uv,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_yuyv,
                        int rows_imgy,
                        int cols_imgy,
                        int rows_imguv,
                        int cols_imguv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=inimg_y  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=inimg_uv  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_yuyv  	offset=slave bundle=gmem3
    #pragma HLS INTERFACE s_axilite port=rows_imgy           
    #pragma HLS INTERFACE s_axilite port=cols_imgy           
    #pragma HLS INTERFACE s_axilite port=rows_imguv          
    #pragma HLS INTERFACE s_axilite port=cols_imguv          
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_imgy;
    imgInput0.cols = cols_imgy;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgInput1;
// clang-format off
    #pragma HLS stream variable=imgInput1.data depth=2
    // clang-format on
    imgInput1.rows = rows_imguv;
    imgInput1.cols = cols_imguv;

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_imgy;
    imgOutput0.cols = cols_imgy;

    xf::cv::accel_utils obj_iny, obj_inuv;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    obj_iny.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(inimg_y, imgInput0);
    obj_inuv.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(inimg_uv, imgInput1);
    xf::cv::nv212yuyv<XF_8UC1, XF_8UC2, XF_16UC1, HEIGHT, WIDTH, NPC1, NPC2>(imgInput0, imgInput1, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, img_yuyv);
}
#endif
#if NV122UYVY

void cvtcolor_nv122uyvy(ap_uint<INPUT_PTR_WIDTH>* inimg_y,
                        ap_uint<INPUT_PTR_WIDTH>* inimg_uv,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_uyvy,
                        int rows_imgy,
                        int cols_imgy,
                        int rows_imguv,
                        int cols_imguv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=inimg_y  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=inimg_uv  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_uyvy  	offset=slave bundle=gmem3
    #pragma HLS INTERFACE s_axilite port=rows_imgy           
    #pragma HLS INTERFACE s_axilite port=cols_imgy           
    #pragma HLS INTERFACE s_axilite port=rows_imguv          
    #pragma HLS INTERFACE s_axilite port=cols_imguv          
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_imgy;
    imgInput0.cols = cols_imgy;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgInput1;
// clang-format off
    #pragma HLS stream variable=imgInput1.data depth=2
    // clang-format on
    imgInput1.rows = rows_imguv;
    imgInput1.cols = cols_imguv;

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_imgy;
    imgOutput0.cols = cols_imgy;

    xf::cv::accel_utils obj_iny, obj_inuv;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    obj_iny.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(inimg_y, imgInput0);
    obj_inuv.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(inimg_uv, imgInput1);
    xf::cv::nv122uyvy<XF_8UC1, XF_8UC2, XF_16UC1, HEIGHT, WIDTH, NPC1, NPC2>(imgInput0, imgInput1, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, img_uyvy);
}
#endif
#if NV212UYVY

void cvtcolor_nv212uyvy(ap_uint<INPUT_PTR_WIDTH>* inimg_y,
                        ap_uint<INPUT_PTR_WIDTH>* inimg_uv,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_uyvy,
                        int rows_imgy,
                        int cols_imgy,
                        int rows_imguv,
                        int cols_imguv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=inimg_y  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=inimg_uv  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_uyvy  	offset=slave bundle=gmem3
    #pragma HLS INTERFACE s_axilite port=rows_imgy           
    #pragma HLS INTERFACE s_axilite port=cols_imgy           
    #pragma HLS INTERFACE s_axilite port=rows_imguv          
    #pragma HLS INTERFACE s_axilite port=cols_imguv          
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_imgy;
    imgInput0.cols = cols_imgy;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgInput1;
// clang-format off
    #pragma HLS stream variable=imgInput1.data depth=2
    // clang-format on
    imgInput1.rows = rows_imguv;
    imgInput1.cols = cols_imguv;

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_imgy;
    imgOutput0.cols = cols_imgy;

    xf::cv::accel_utils obj_iny, obj_inuv;
// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    obj_iny.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(inimg_y, imgInput0);
    obj_inuv.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(inimg_uv, imgInput1);
    xf::cv::nv212uyvy<XF_8UC1, XF_8UC2, XF_16UC1, HEIGHT, WIDTH, NPC1, NPC2>(imgInput0, imgInput1, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, img_uyvy);
}
#endif
#if YUYV2UYVY

void cvtcolor_yuyv2uyvy(ap_uint<INPUT_PTR_WIDTH>* yuyv, ap_uint<INPUT_PTR_WIDTH>* uyvy, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=yuyv  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=uyvy  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows             
    #pragma HLS INTERFACE s_axilite port=cols            
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_16UC1, HEIGHT, WIDTH, NPC1>(yuyv, imgInput0);
    xf::cv::yuyv2uyvy<XF_16UC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, uyvy);
}
#endif
#if UYVY2YUYV

void cvtcolor_uyvy2yuyv(ap_uint<INPUT_PTR_WIDTH>* uyvy, ap_uint<OUTPUT_PTR_WIDTH>* yuyv, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=yuyv  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=uyvy  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows             
    #pragma HLS INTERFACE s_axilite port=cols            
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_16UC1, HEIGHT, WIDTH, NPC1>(uyvy, imgInput0);
    xf::cv::uyvy2yuyv<XF_16UC1, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_16UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, yuyv);
}
#endif
#if NV212IYUV

void cvtcolor_nv212iyuv(ap_uint<INPUT_PTR_WIDTH>* inimg_y,
                        ap_uint<INPUT_PTR_WIDTH>* inimg_uv,
                        ap_uint<OUTPUT_PTR_WIDTH>* outimg_y,
                        ap_uint<OUTPUT_PTR_WIDTH>* outimg_u,
                        ap_uint<OUTPUT_PTR_WIDTH>* outimg_v,
                        int rows_imgy,
                        int cols_imgy,
                        int rows_imguv,
                        int cols_imguv,
                        int rows_outy,
                        int cols_outy,
                        int rows_outuv,
                        int cols_outuv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=inimg_y  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=inimg_uv  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=outimg_y  	offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi     port=outimg_u  	offset=slave bundle=gmem4
    #pragma HLS INTERFACE m_axi     port=outimg_v  	offset=slave bundle=gmem5
// clang-format on

// clang-format off
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=rows_imgy           
    #pragma HLS INTERFACE s_axilite port=cols_imgy           
    #pragma HLS INTERFACE s_axilite port=rows_imguv          
    #pragma HLS INTERFACE s_axilite port=cols_imguv          
    #pragma HLS INTERFACE s_axilite port=rows_outy           
    #pragma HLS INTERFACE s_axilite port=cols_outy           
    #pragma HLS INTERFACE s_axilite port=rows_outuv           
    #pragma HLS INTERFACE s_axilite port=cols_outuv
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_imgy;
    imgInput0.cols = cols_imgy;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgInput1;
// clang-format off
    #pragma HLS stream variable=imgInput1.data depth=2
    // clang-format on
    imgInput1.rows = rows_imguv;
    imgInput1.cols = cols_imguv;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_outy;
    imgOutput0.cols = cols_outy;

    xf::cv::Mat<XF_8UC1, HEIGHT / 4, WIDTH, NPC1> imgOutput1;
// clang-format off
    #pragma HLS stream variable=imgOutput1.data depth=2
    // clang-format on
    imgOutput1.rows = rows_outuv;
    imgOutput1.cols = cols_outuv;

    xf::cv::Mat<XF_8UC1, HEIGHT / 4, WIDTH, NPC1> imgOutput2;
// clang-format off
    #pragma HLS stream variable=imgOutput2.data depth=2
    // clang-format on
    imgOutput2.rows = rows_outuv;
    imgOutput2.cols = cols_outuv;
    xf::cv::accel_utils obj_iny, obj_inuv, obj_outy, obj_outu, obj_outv;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    obj_iny.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(inimg_y, imgInput0);
    obj_inuv.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(inimg_uv, imgInput1);

    xf::cv::nv212iyuv<XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(imgInput0, imgInput1, imgOutput0, imgOutput1,
                                                                   imgOutput2);

    obj_outy.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, outimg_y);
    obj_outu.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT / 4, WIDTH, NPC1>(imgOutput1, outimg_u);
    obj_outv.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT / 4, WIDTH, NPC1>(imgOutput2, outimg_v);
}
#endif
#if NV212RGBA

void cvtcolor_nv212rgba(ap_uint<INPUT_PTR_WIDTH>* inimg_y,
                        ap_uint<INPUT_PTR_WIDTH>* inimg_uv,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_rgba,
                        int rows_imgy,
                        int cols_imgy,
                        int rows_imguv,
                        int cols_imguv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=inimg_y  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=inimg_uv  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_rgba  	offset=slave bundle=gmem3
    #pragma HLS INTERFACE s_axilite port=rows_imgy           
    #pragma HLS INTERFACE s_axilite port=cols_imgy           
    #pragma HLS INTERFACE s_axilite port=rows_imguv          
    #pragma HLS INTERFACE s_axilite port=cols_imguv          
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_imgy;
    imgInput0.cols = cols_imgy;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgInput1;
// clang-format off
    #pragma HLS stream variable=imgInput1.data depth=2
    // clang-format on
    imgInput1.rows = rows_imguv;
    imgInput1.cols = cols_imguv;

    xf::cv::Mat<XF_8UC4, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_imgy;
    imgOutput0.cols = cols_imgy;

    xf::cv::accel_utils obj_iny, obj_inuv;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    obj_iny.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(inimg_y, imgInput0);
    obj_inuv.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(inimg_uv, imgInput1);
    xf::cv::nv212rgba<XF_8UC1, XF_8UC2, XF_8UC4, HEIGHT, WIDTH, NPC1>(imgInput0, imgInput1, imgOutput0);

    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC4, HEIGHT, WIDTH, NPC1>(imgOutput0, img_rgba);
}
#endif
#if NV212RGB

void cvtcolor_nv212rgb(ap_uint<INPUT_PTR_WIDTH>* inimg_y,
                       ap_uint<INPUT_PTR_WIDTH>* inimg_uv,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_rgb,
                       int rows_imgy,
                       int cols_imgy,
                       int rows_imguv,
                       int cols_imguv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=inimg_y  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=inimg_uv  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem3
    #pragma HLS INTERFACE s_axilite port=rows_imgy           
    #pragma HLS INTERFACE s_axilite port=cols_imgy           
    #pragma HLS INTERFACE s_axilite port=rows_imguv          
    #pragma HLS INTERFACE s_axilite port=cols_imguv          
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_imgy;
    imgInput0.cols = cols_imgy;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgInput1;
// clang-format off
    #pragma HLS stream variable=imgInput1.data depth=2
    // clang-format on
    imgInput1.rows = rows_imguv;
    imgInput1.cols = cols_imguv;

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_imgy;
    imgOutput0.cols = cols_imgy;

    xf::cv::accel_utils obj_iny, obj_inuv;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    obj_iny.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(inimg_y, imgInput0);
    obj_inuv.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(inimg_uv, imgInput1);
    xf::cv::nv212rgb<XF_8UC1, XF_8UC2, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput0, imgInput1, imgOutput0);

    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_rgb);
}
#endif
/*#if NV212IYUV

        void cvtcolor_nv212iyuv(ap_uint<INPUT_PTR_WIDTH> *inimg_y, ap_uint<INPUT_PTR_WIDTH> *inimg_uv,
ap_uint<OUTPUT_PTR_WIDTH> *outimg_y,ap_uint<OUTPUT_PTR_WIDTH> *outimg_u,ap_uint<OUTPUT_PTR_WIDTH> *outimg_v,int
rows_imgy, int cols_imgy,int rows_imguv,int cols_imguv,int rows_outy,int cols_outy,int rows_outuv,int cols_outuv)
        {
// clang-format off
                #pragma HLS INTERFACE m_axi     port=inimg_y  	offset=slave bundle=gmem1
                #pragma HLS INTERFACE m_axi     port=inimg_uv  	offset=slave bundle=gmem2
                #pragma HLS INTERFACE m_axi     port=outimg_y  	offset=slave bundle=gmem3
                #pragma HLS INTERFACE m_axi     port=outimg_u  	offset=slave bundle=gmem4
                #pragma HLS INTERFACE m_axi     port=outimg_v  	offset=slave bundle=gmem5
// clang-format on

// clang-format off
// clang-format on

// clang-format off
                #pragma HLS INTERFACE s_axilite port=rows_imgy
                #pragma HLS INTERFACE s_axilite port=cols_imgy
                #pragma HLS INTERFACE s_axilite port=rows_imguv
                #pragma HLS INTERFACE s_axilite port=cols_imguv
                #pragma HLS INTERFACE s_axilite port=rows_outy
                #pragma HLS INTERFACE s_axilite port=cols_outy
                #pragma HLS INTERFACE s_axilite port=rows_outuv
                #pragma HLS INTERFACE s_axilite port=cols_outuv
// clang-format on

// clang-format off
                #pragma HLS INTERFACE s_axilite port=return
// clang-format on


                 xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1>   imgInput0;
// clang-format off
        #pragma HLS stream variable=imgInput0.data depth=2
// clang-format on
                imgInput0.rows=rows_imgy; imgInput0.cols=cols_imgy;

                         xf::cv::Mat<XF_8UC2, HEIGHT/2, WIDTH/2, NPC2>   imgInput1;
// clang-format off
        #pragma HLS stream variable=imgInput1.data depth=2
// clang-format on
                imgInput1.rows=rows_imguv; imgInput1.cols=cols_imguv;

                 xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1>   imgOutput0;
// clang-format off
        #pragma HLS stream variable=imgOutput0.data depth=2
// clang-format on
                imgOutput0.rows=rows_outy; imgOutput0.cols=cols_outy;

                 xf::cv::Mat<XF_8UC1, HEIGHT/4, WIDTH, NPC1> imgOutput1;
// clang-format off
        #pragma HLS stream variable=imgOutput1.data depth=2
// clang-format on
                imgOutput1.rows=rows_outuv; imgOutput1.cols=cols_outuv;

                 xf::cv::Mat<XF_8UC1, HEIGHT/4, WIDTH, NPC1> imgOutput2;
// clang-format off
        #pragma HLS stream variable=imgOutput2.data depth=2
// clang-format on
                imgOutput2.rows=rows_outuv; imgOutput2.cols=cols_outuv;

// clang-format off
                #pragma HLS DATAFLOW
// clang-format on
                xf::cv::Array2xfMat<INPUT_PTR_WIDTH,XF_8UC1,HEIGHT, WIDTH, NPC1>  (inimg_y, imgInput0);
                xf::cv::Array2xfMat<INPUT_PTR_WIDTH,XF_8UC2,HEIGHT/2, WIDTH/2, NPC2>  (inimg_uv, imgInput1);

                xf::cv::nv212iyuv<XF_8UC1,XF_8UC2,HEIGHT,WIDTH,NPC1,NPC2>(imgInput0,imgInput1,imgOutput0,imgOutput1,imgOutput2);

                xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH,XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0,outimg_y);
                xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH,XF_8UC1, HEIGHT/4, WIDTH, NPC1>(imgOutput1,outimg_u);
                xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH,XF_8UC1, HEIGHT/4, WIDTH, NPC1>(imgOutput2,outimg_v);

        }
#endif*/
#if NV212YUV4
void cvtcolor_nv212yuv4(ap_uint<INPUT_PTR_WIDTH>* inimg_y,
                        ap_uint<INPUT_PTR_WIDTH>* inimg_uv,
                        ap_uint<OUTPUT_PTR_WIDTH>* outimg_y,
                        ap_uint<OUTPUT_PTR_WIDTH>* outimg_u,
                        ap_uint<OUTPUT_PTR_WIDTH>* outimg_v,
                        int rows_imgy,
                        int cols_imgy,
                        int rows_imguv,
                        int cols_imguv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=inimg_y  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=inimg_uv  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=outimg_y  	offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi     port=outimg_u  	offset=slave bundle=gmem4
    #pragma HLS INTERFACE m_axi     port=outimg_v  	offset=slave bundle=gmem5
    #pragma HLS INTERFACE s_axilite port=rows_imgy           
    #pragma HLS INTERFACE s_axilite port=cols_imgy           
    #pragma HLS INTERFACE s_axilite port=rows_imguv          
    #pragma HLS INTERFACE s_axilite port=cols_imguv          
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_imgy;
    imgInput0.cols = cols_imgy;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgInput1;
// clang-format off
    #pragma HLS stream variable=imgInput1.data depth=2
    // clang-format on
    imgInput1.rows = rows_imguv;
    imgInput1.cols = cols_imguv;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_imgy;
    imgOutput0.cols = cols_imgy;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput1;
// clang-format off
    #pragma HLS stream variable=imgOutput1.data depth=2
    // clang-format on
    imgOutput1.rows = rows_imgy;
    imgOutput1.cols = cols_imgy;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput2;
// clang-format off
    #pragma HLS stream variable=imgOutput2.data depth=2
    // clang-format on
    imgOutput2.rows = rows_imgy;
    imgOutput2.cols = cols_imgy;

    xf::cv::accel_utils obj_iny, obj_inuv, obj_outy, obj_outu, obj_outv;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    obj_iny.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(inimg_y, imgInput0);
    obj_inuv.Array2xfMat<INPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(inimg_uv, imgInput1);

    xf::cv::nv212yuv4<XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(imgInput0, imgInput1, imgOutput0, imgOutput1,
                                                                   imgOutput2);

    obj_outy.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, outimg_y);
    obj_outu.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput1, outimg_u);
    obj_outv.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput2, outimg_v);
}
#endif
#if UYVY2IYUV

void cvtcolor_uyvy2iyuv(ap_uint<INPUT_PTR_WIDTH>* img_uyvy,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_y,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_u,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_v,
                        int rows_uyvy,
                        int cols_uyvy,
                        int rows_y,
                        int cols_y,
                        int rows_u,
                        int cols_u,
                        int rows_v,
                        int cols_v) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_uyvy  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_y  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_u  	offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi     port=img_v  	offset=slave bundle=gmem5
    #pragma HLS INTERFACE s_axilite port=rows_uyvy           
    #pragma HLS INTERFACE s_axilite port=cols_uyvy           
    #pragma HLS INTERFACE s_axilite port=rows_y              
    #pragma HLS INTERFACE s_axilite port=cols_y              
    #pragma HLS INTERFACE s_axilite port=rows_u              
    #pragma HLS INTERFACE s_axilite port=cols_u            	 
    #pragma HLS INTERFACE s_axilite port=rows_v              
    #pragma HLS INTERFACE s_axilite port=cols_v            	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_uyvy;
    imgInput0.cols = cols_uyvy;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_y;
    imgOutput0.cols = cols_y;

    xf::cv::Mat<XF_8UC1, HEIGHT / 4, WIDTH, NPC1> imgOutput1;
// clang-format off
    #pragma HLS stream variable=imgOutput1.data depth=2
    // clang-format on
    imgOutput1.rows = rows_u;
    imgOutput1.cols = cols_u;

    xf::cv::Mat<XF_8UC1, HEIGHT / 4, WIDTH, NPC1> imgOutput2;
// clang-format off
    #pragma HLS stream variable=imgOutput2.data depth=2
    // clang-format on
    imgOutput1.rows = rows_v;
    imgOutput1.cols = cols_v;

    xf::cv::accel_utils obj_outy, obj_outu, obj_outv;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_16UC1, HEIGHT, WIDTH, NPC1>(img_uyvy, imgInput0);
    xf::cv::uyvy2iyuv<XF_16UC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0, imgOutput1, imgOutput2);
    obj_outy.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, img_y);
    obj_outu.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT / 4, WIDTH, NPC1>(imgOutput1, img_u);
    obj_outv.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT / 4, WIDTH, NPC1>(imgOutput2, img_v);
}
#endif
#if UYVY2NV12
void cvtcolor_uyvy2nv12(ap_uint<INPUT_PTR_WIDTH>* img_uyvy,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_y,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_uv,
                        int rows_uyvy,
                        int cols_uyvy,
                        int rows_y,
                        int cols_y,
                        int rows_uv,
                        int cols_uv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_uyvy  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_y  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_uv  	offset=slave bundle=gmem3
// clang-format on

// clang-format off
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=rows_uyvy           
    #pragma HLS INTERFACE s_axilite port=cols_uyvy           
    #pragma HLS INTERFACE s_axilite port=rows_y              
    #pragma HLS INTERFACE s_axilite port=cols_y              
    #pragma HLS INTERFACE s_axilite port=rows_uv              
    #pragma HLS INTERFACE s_axilite port=cols_uv
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_uyvy;
    imgInput0.cols = cols_uyvy;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_y;
    imgOutput0.cols = cols_y;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgOutput1;
// clang-format off
    #pragma HLS stream variable=imgOutput1.data depth=2
    // clang-format on
    imgOutput1.rows = rows_uv;
    imgOutput1.cols = cols_uv;

    xf::cv::accel_utils obj_outy, obj_outuv;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_16UC1, HEIGHT, WIDTH, NPC1>(img_uyvy, imgInput0);
    xf::cv::uyvy2nv12<XF_16UC1, XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(imgInput0, imgOutput0, imgOutput1);
    obj_outy.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, img_y);
    obj_outuv.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(imgOutput1, img_uv);
}
#endif
#if UYVY2NV21
void cvtcolor_uyvy2nv21(ap_uint<INPUT_PTR_WIDTH>* img_uyvy,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_y,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_uv,
                        int rows_uyvy,
                        int cols_uyvy,
                        int rows_y,
                        int cols_y,
                        int rows_uv,
                        int cols_uv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_uyvy  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_y  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_uv  	offset=slave bundle=gmem3
// clang-format on

// clang-format off
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=rows_uyvy           
    #pragma HLS INTERFACE s_axilite port=cols_uyvy           
    #pragma HLS INTERFACE s_axilite port=rows_y              
    #pragma HLS INTERFACE s_axilite port=cols_y              
    #pragma HLS INTERFACE s_axilite port=rows_uv              
    #pragma HLS INTERFACE s_axilite port=cols_uv
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_uyvy;
    imgInput0.cols = cols_uyvy;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_y;
    imgOutput0.cols = cols_y;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgOutput1;
// clang-format off
    #pragma HLS stream variable=imgOutput1.data depth=2
    // clang-format on
    imgOutput1.rows = rows_uv;
    imgOutput1.cols = cols_uv;

    xf::cv::accel_utils obj_outy, obj_outuv;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_16UC1, HEIGHT, WIDTH, NPC1>(img_uyvy, imgInput0);
    xf::cv::uyvy2nv21<XF_16UC1, XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(imgInput0, imgOutput0, imgOutput1);
    obj_outy.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, img_y);
    obj_outuv.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(imgOutput1, img_uv);
}
#endif
#if UYVY2RGBA

void cvtcolor_uyvy2rgba(ap_uint<INPUT_PTR_WIDTH>* img_uyvy,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_rgba,
                        int rows_uyvy,
                        int cols_uyvy) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_uyvy  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_rgba  	offset=slave bundle=gmem2
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=rows_uyvy           
    #pragma HLS INTERFACE s_axilite port=cols_uyvy
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_uyvy;
    imgInput0.cols = cols_uyvy;

    xf::cv::Mat<XF_8UC4, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_uyvy;
    imgOutput0.cols = cols_uyvy;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_16UC1, HEIGHT, WIDTH, NPC1>(img_uyvy, imgInput0);
    xf::cv::uyvy2rgba<XF_16UC1, XF_8UC4, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC4, HEIGHT, WIDTH, NPC1>(imgOutput0, img_rgba);
}

#endif

#if UYVY2RGB
void cvtcolor_uyvy2rgb(ap_uint<INPUT_PTR_WIDTH>* img_uyvy,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_rgb,
                       int rows_uyvy,
                       int cols_uyvy) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_uyvy  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem2
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=rows_uyvy           
    #pragma HLS INTERFACE s_axilite port=cols_uyvy
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_uyvy;
    imgInput0.cols = cols_uyvy;

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_uyvy;
    imgOutput0.cols = cols_uyvy;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_16UC1, HEIGHT, WIDTH, NPC1>(img_uyvy, imgInput0);
    xf::cv::uyvy2rgb<XF_16UC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_rgb);
}

#endif

#if YUYV2IYUV

void cvtcolor_yuyv2iyuv(ap_uint<INPUT_PTR_WIDTH>* img_yuyv,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_y,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_u,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_v,
                        int rows_yuyv,
                        int cols_yuyv,
                        int rows_y,
                        int cols_y,
                        int rows_u,
                        int cols_u,
                        int rows_v,
                        int cols_v) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_yuyv  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_y  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_u  	offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi     port=img_v  	offset=slave bundle=gmem5
    #pragma HLS INTERFACE s_axilite port=rows_yuyv           
    #pragma HLS INTERFACE s_axilite port=cols_yuyv           
    #pragma HLS INTERFACE s_axilite port=rows_y              
    #pragma HLS INTERFACE s_axilite port=cols_y              
    #pragma HLS INTERFACE s_axilite port=rows_u              
    #pragma HLS INTERFACE s_axilite port=cols_u            	 
    #pragma HLS INTERFACE s_axilite port=rows_v              
    #pragma HLS INTERFACE s_axilite port=cols_v            	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_yuyv;
    imgInput0.cols = cols_yuyv;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_y;
    imgOutput0.cols = cols_y;

    xf::cv::Mat<XF_8UC1, HEIGHT / 4, WIDTH, NPC1> imgOutput1;
// clang-format off
    #pragma HLS stream variable=imgOutput1.data depth=2
    // clang-format on
    imgOutput1.rows = rows_u;
    imgOutput1.cols = cols_u;

    xf::cv::Mat<XF_8UC1, HEIGHT / 4, WIDTH, NPC1> imgOutput2;
// clang-format off
    #pragma HLS stream variable=imgOutput2.data depth=2
    // clang-format on
    imgOutput2.rows = rows_v;
    imgOutput2.cols = cols_v;

    xf::cv::accel_utils obj_outy, obj_outu, obj_outv;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_16UC1, HEIGHT, WIDTH, NPC1>(img_yuyv, imgInput0);

    xf::cv::yuyv2iyuv<XF_16UC1, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0, imgOutput1, imgOutput2);

    obj_outy.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, img_y);
    obj_outu.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT / 4, WIDTH, NPC1>(imgOutput1, img_u);
    obj_outv.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT / 4, WIDTH, NPC1>(imgOutput2, img_v);
}
#endif
#if YUYV2NV12

void cvtcolor_yuyv2nv12(ap_uint<INPUT_PTR_WIDTH>* img_yuyv,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_y,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_uv,
                        int rows_yuyv,
                        int cols_yuyv,
                        int rows_y,
                        int cols_y,
                        int rows_uv,
                        int cols_uv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_yuyv  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_y  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_uv  	offset=slave bundle=gmem3
// clang-format on

// clang-format off
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=rows_yuyv           
    #pragma HLS INTERFACE s_axilite port=cols_yuyv           
    #pragma HLS INTERFACE s_axilite port=rows_y              
    #pragma HLS INTERFACE s_axilite port=cols_y              
    #pragma HLS INTERFACE s_axilite port=rows_uv             
    #pragma HLS INTERFACE s_axilite port=cols_uv
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_yuyv;
    imgInput0.cols = cols_yuyv;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_y;
    imgOutput0.cols = cols_y;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgOutput1;
// clang-format off
    #pragma HLS stream variable=imgOutput1.data depth=2
    // clang-format on
    imgOutput1.rows = rows_uv;
    imgOutput1.cols = cols_uv;

    xf::cv::accel_utils obj_outy, obj_outuv;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_16UC1, HEIGHT, WIDTH, NPC1>(img_yuyv, imgInput0);
    xf::cv::yuyv2nv12<XF_16UC1, XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(imgInput0, imgOutput0, imgOutput1);
    obj_outy.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, img_y);
    obj_outuv.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(imgOutput1, img_uv);
}
#endif
#if YUYV2NV21

void cvtcolor_yuyv2nv21(ap_uint<INPUT_PTR_WIDTH>* img_yuyv,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_y,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_uv,
                        int rows_yuyv,
                        int cols_yuyv,
                        int rows_y,
                        int cols_y,
                        int rows_uv,
                        int cols_uv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_yuyv  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_y  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_uv  	offset=slave bundle=gmem3
// clang-format on

// clang-format off
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=rows_yuyv           
    #pragma HLS INTERFACE s_axilite port=cols_yuyv           
    #pragma HLS INTERFACE s_axilite port=rows_y              
    #pragma HLS INTERFACE s_axilite port=cols_y              
    #pragma HLS INTERFACE s_axilite port=rows_uv             
    #pragma HLS INTERFACE s_axilite port=cols_uv
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_yuyv;
    imgInput0.cols = cols_yuyv;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_y;
    imgOutput0.cols = cols_y;

    xf::cv::Mat<XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2> imgOutput1;
// clang-format off
    #pragma HLS stream variable=imgOutput1.data depth=2
    // clang-format on
    imgOutput1.rows = rows_uv;
    imgOutput1.cols = cols_uv;

    xf::cv::accel_utils obj_outy, obj_outuv;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_16UC1, HEIGHT, WIDTH, NPC1>(img_yuyv, imgInput0);
    xf::cv::yuyv2nv21<XF_16UC1, XF_8UC1, XF_8UC2, HEIGHT, WIDTH, NPC1, NPC2>(imgInput0, imgOutput0, imgOutput1);
    obj_outy.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, img_y);
    obj_outuv.xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC2, HEIGHT / 2, WIDTH / 2, NPC2>(imgOutput1, img_uv);
}
#endif
#if YUYV2RGBA

void cvtcolor_yuyv2rgba(ap_uint<INPUT_PTR_WIDTH>* img_yuyv,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_rgba,
                        int rows_yuyv,
                        int cols_yuyv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_yuyv  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_rgba  	offset=slave bundle=gmem2
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=rows_yuyv           
    #pragma HLS INTERFACE s_axilite port=cols_yuyv
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_yuyv;
    imgInput0.cols = cols_yuyv;

    xf::cv::Mat<XF_8UC4, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_yuyv;
    imgOutput0.cols = cols_yuyv;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_16UC1, HEIGHT, WIDTH, NPC1>(img_yuyv, imgInput0);
    xf::cv::yuyv2rgba<XF_16UC1, XF_8UC4, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC4, HEIGHT, WIDTH, NPC1>(imgOutput0, img_rgba);
}
#endif
#if YUYV2RGB

void cvtcolor_yuyv2rgb(ap_uint<INPUT_PTR_WIDTH>* img_yuyv,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_rgb,
                       int rows_yuyv,
                       int cols_yuyv) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_yuyv  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem2
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=rows_yuyv           
    #pragma HLS INTERFACE s_axilite port=cols_yuyv
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_16UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows_yuyv;
    imgInput0.cols = cols_yuyv;

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows_yuyv;
    imgOutput0.cols = cols_yuyv;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_16UC1, HEIGHT, WIDTH, NPC1>(img_yuyv, imgInput0);
    xf::cv::yuyv2rgb<XF_16UC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_rgb);
}
#endif
#if RGB2GRAY

void cvtcolor_rgb2gray(ap_uint<INPUT_PTR_WIDTH>* img_rgb, ap_uint<OUTPUT_PTR_WIDTH>* img_gray, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_gray  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_rgb, imgInput0);
    xf::cv::rgb2gray<XF_8UC3, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, img_gray);
}
#endif
#if BGR2GRAY
void cvtcolor_bgr2gray(ap_uint<INPUT_PTR_WIDTH>* img_bgr, ap_uint<OUTPUT_PTR_WIDTH>* img_gray, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_bgr  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_gray  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_bgr, imgInput0);
    xf::cv::bgr2gray<XF_8UC3, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(imgOutput0, img_gray);
}
#endif
#if GRAY2RGB
void cvtcolor_gray2rgb(ap_uint<INPUT_PTR_WIDTH>* img_gray, ap_uint<OUTPUT_PTR_WIDTH>* img_rgb, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_gray  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(img_gray, imgInput0);
    xf::cv::gray2rgb<XF_8UC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_rgb);
}
#endif
#if GRAY2BGR

void cvtcolor_gray2bgr(ap_uint<INPUT_PTR_WIDTH>* img_gray, ap_uint<OUTPUT_PTR_WIDTH>* img_bgr, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_gray  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_bgr  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPC1>(img_gray, imgInput0);
    xf::cv::gray2bgr<XF_8UC1, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_bgr);
}
#endif
#if RGB2BGR

void cvtcolor_rgb2bgr(ap_uint<INPUT_PTR_WIDTH>* img_rgb, ap_uint<OUTPUT_PTR_WIDTH>* img_bgr, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_bgr  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_rgb, imgInput0);
    xf::cv::rgb2bgr<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_bgr);
}
#endif
#if BGR2RGB

void cvtcolor_bgr2rgb(ap_uint<INPUT_PTR_WIDTH>* img_rgb, ap_uint<OUTPUT_PTR_WIDTH>* img_bgr, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_bgr  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_rgb, imgInput0);
    xf::cv::bgr2rgb<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_bgr);
}
#endif
#if RGB2XYZ

void cvtcolor_rgb2xyz(ap_uint<INPUT_PTR_WIDTH>* img_rgb, ap_uint<OUTPUT_PTR_WIDTH>* img_xyz, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_xyz  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_rgb, imgInput0);
    xf::cv::rgb2xyz<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_xyz);
}
#endif
#if BGR2XYZ
void cvtcolor_bgr2xyz(ap_uint<INPUT_PTR_WIDTH>* img_bgr, ap_uint<OUTPUT_PTR_WIDTH>* img_xyz, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_bgr  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_xyz  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_bgr, imgInput0);
    xf::cv::bgr2xyz<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_xyz);
}
#endif
#if XYZ2RGB

void cvtcolor_xyz2rgb(ap_uint<INPUT_PTR_WIDTH>* img_xyz, ap_uint<OUTPUT_PTR_WIDTH>* img_rgb, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_xyz  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_xyz, imgInput0);
    xf::cv::xyz2rgb<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_rgb);
}
#endif
#if XYZ2BGR
void cvtcolor_xyz2bgr(ap_uint<INPUT_PTR_WIDTH>* img_xyz, ap_uint<OUTPUT_PTR_WIDTH>* img_bgr, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_xyz  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_bgr  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_xyz, imgInput0);
    xf::cv::xyz2bgr<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_bgr);
}
#endif
#if RGB2YCrCb

void cvtcolor_rgb2ycrcb(ap_uint<INPUT_PTR_WIDTH>* img_rgb, ap_uint<OUTPUT_PTR_WIDTH>* img_ycrcb, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_ycrcb  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_rgb, imgInput0);
    xf::cv::rgb2ycrcb<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_ycrcb);
}
#endif

#if BGR2YCrCb

void cvtcolor_bgr2ycrcb(ap_uint<INPUT_PTR_WIDTH>* img_bgr, ap_uint<OUTPUT_PTR_WIDTH>* img_ycrcb, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_bgr  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_ycrcb  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_bgr, imgInput0);
    xf::cv::bgr2ycrcb<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_ycrcb);
}
#endif

#if YCrCb2RGB

void cvtcolor_ycrcb2rgb(ap_uint<INPUT_PTR_WIDTH>* img_ycrcb, ap_uint<OUTPUT_PTR_WIDTH>* img_rgb, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_ycrcb  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_ycrcb, imgInput0);
    xf::cv::ycrcb2rgb<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_rgb);
}
#endif
#if YCrCb2BGR

void cvtcolor_ycrcb2bgr(ap_uint<INPUT_PTR_WIDTH>* img_ycrcb, ap_uint<OUTPUT_PTR_WIDTH>* img_bgr, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_ycrcb  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_bgr  		offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_ycrcb, imgInput0);
    xf::cv::ycrcb2bgr<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_bgr);
}
#endif
#if RGB2HLS
void cvtcolor_rgb2hls(ap_uint<INPUT_PTR_WIDTH>* img_rgb, ap_uint<OUTPUT_PTR_WIDTH>* img_hls, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_hls  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_rgb, imgInput0);
    xf::cv::rgb2hls<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_hls);
}
#endif
#if BGR2HLS
void cvtcolor_bgr2hls(ap_uint<INPUT_PTR_WIDTH>* img_bgr, ap_uint<OUTPUT_PTR_WIDTH>* img_hls, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_bgr  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_hls  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_bgr, imgInput0);
    xf::cv::bgr2hls<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_hls);
}
#endif
#if HLS2RGB

void cvtcolor_hls2rgb(ap_uint<INPUT_PTR_WIDTH>* img_hls, ap_uint<OUTPUT_PTR_WIDTH>* img_rgb, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_hls  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_hls, imgInput0);
    xf::cv::hls2rgb<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_rgb);
}
#endif
#if HLS2BGR

void cvtcolor_hls2bgr(ap_uint<INPUT_PTR_WIDTH>* img_hls, ap_uint<OUTPUT_PTR_WIDTH>* img_bgr, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_hls  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_bgr  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_hls, imgInput0);
    xf::cv::hls2bgr<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_bgr);
}
#endif
#if RGB2HSV

void cvtcolor_rgb2hsv(ap_uint<INPUT_PTR_WIDTH>* img_rgb, ap_uint<OUTPUT_PTR_WIDTH>* img_hsv, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_hsv  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_rgb, imgInput0);
    xf::cv::rgb2hsv<XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_bgr);
}

#endif
#if BGR2HSV

void cvtcolor_bgr2hsv(ap_uint<INPUT_PTR_WIDTH>* img_bgr, ap_uint<OUTPUT_PTR_WIDTH>* img_hsv, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_bgr  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_hsv  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_bgr, imgInput0);
    xf::cv::bgr2hsv<XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_hsv);
}
#endif
#if HSV2RGB

void cvtcolor_hsv2rgb(ap_uint<INPUT_PTR_WIDTH>* img_hsv, ap_uint<OUTPUT_PTR_WIDTH>* img_rgb, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_hsv  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_rgb  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_hsv, imgInput0);
    xf::cv::hsv2rgb<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_rgb);
}
#endif
#if HSV2BGR

void cvtcolor_hsv2bgr(ap_uint<INPUT_PTR_WIDTH>* img_hsv, ap_uint<OUTPUT_PTR_WIDTH>* img_bgr, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_hsv  	offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_bgr  	offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows              	 
    #pragma HLS INTERFACE s_axilite port=cols              	 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0;
// clang-format off
    #pragma HLS stream variable=imgInput0.data depth=2
    // clang-format on
    imgInput0.rows = rows;
    imgInput0.cols = cols;
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgOutput0;
// clang-format off
    #pragma HLS stream variable=imgOutput0.data depth=2
    // clang-format on
    imgOutput0.rows = rows;
    imgOutput0.cols = cols;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(img_hsv, imgInput0);
    xf::cv::hsv2bgr<XF_8UC3, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgInput0, imgOutput0);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC3, HEIGHT, WIDTH, NPC1>(imgOutput0, img_bgr);
}
#endif
}
