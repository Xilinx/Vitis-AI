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

#include "xf_colordetect_config.h"

static constexpr int __XF_DEPTH = (HEIGHT * WIDTH * (XF_PIXELWIDTH(IN_TYPE, NPC1)) / 8) / (INPUT_PTR_WIDTH / 8);
static constexpr int __XF_DEPTH_OUT = (HEIGHT * WIDTH * (XF_PIXELWIDTH(OUT_TYPE, NPC1)) / 8) / (OUTPUT_PTR_WIDTH / 8);
static constexpr int __XF_DEPTH_FILTER = (FILTER_SIZE * FILTER_SIZE);

void colordetect_accel(ap_uint<INPUT_PTR_WIDTH>* img_in,
                       unsigned char* low_thresh,
                       unsigned char* high_thresh,
                       unsigned char* process_shape,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                       int rows,
                       int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi      port=img_in        offset=slave  bundle=gmem0 depth=__XF_DEPTH
   
    #pragma HLS INTERFACE m_axi      port=low_thresh    offset=slave  bundle=gmem1 depth=__XF_DEPTH_FILTER
    #pragma HLS INTERFACE s_axilite  port=low_thresh 			      
    #pragma HLS INTERFACE m_axi      port=high_thresh   offset=slave  bundle=gmem2 depth=__XF_DEPTH_FILTER
    #pragma HLS INTERFACE s_axilite  port=high_thresh 			      
    #pragma HLS INTERFACE s_axilite  port=rows 			      
    #pragma HLS INTERFACE s_axilite  port=cols 			      
    #pragma HLS INTERFACE m_axi      port=process_shape offset=slave  bundle=gmem3 depth=__XF_DEPTH_FILTER
    #pragma HLS INTERFACE s_axilite  port=process_shape			      
    #pragma HLS INTERFACE m_axi      port=img_out       offset=slave  bundle=gmem4 depth=__XF_DEPTH_OUT
  
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> imgInput(rows, cols);
    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> rgb2hsv(rows, cols);
    xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPC1> imgHelper1(rows, cols);
    xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPC1> imgHelper2(rows, cols);
    xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPC1> imgHelper3(rows, cols);
    xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPC1> imgHelper4(rows, cols);
    xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPC1> imgOutput(rows, cols);

    // Copy the shape data:
    unsigned char _kernel[FILTER_SIZE * FILTER_SIZE];
    for (unsigned int i = 0; i < FILTER_SIZE * FILTER_SIZE; ++i) {
// clang-format off
        #pragma HLS PIPELINE
        // clang-format on
        _kernel[i] = process_shape[i];
    }

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC1>(img_in, imgInput);

    // Convert RGBA to HSV:
    xf::cv::bgr2hsv<IN_TYPE, HEIGHT, WIDTH, NPC1>(imgInput, rgb2hsv);

    // Do the color thresholding:
    xf::cv::colorthresholding<IN_TYPE, OUT_TYPE, MAXCOLORS, HEIGHT, WIDTH, NPC1>(rgb2hsv, imgHelper1, low_thresh,
                                                                                 high_thresh);

    // Use erode and dilate to fully mark color areas:
    xf::cv::erode<XF_BORDER_CONSTANT, OUT_TYPE, HEIGHT, WIDTH, XF_KERNEL_SHAPE, FILTER_SIZE, FILTER_SIZE, ITERATIONS,
                  NPC1>(imgHelper1, imgHelper2, _kernel);
    xf::cv::dilate<XF_BORDER_CONSTANT, OUT_TYPE, HEIGHT, WIDTH, XF_KERNEL_SHAPE, FILTER_SIZE, FILTER_SIZE, ITERATIONS,
                   NPC1>(imgHelper2, imgHelper3, _kernel);
    xf::cv::dilate<XF_BORDER_CONSTANT, OUT_TYPE, HEIGHT, WIDTH, XF_KERNEL_SHAPE, FILTER_SIZE, FILTER_SIZE, ITERATIONS,
                   NPC1>(imgHelper3, imgHelper4, _kernel);
    xf::cv::erode<XF_BORDER_CONSTANT, OUT_TYPE, HEIGHT, WIDTH, XF_KERNEL_SHAPE, FILTER_SIZE, FILTER_SIZE, ITERATIONS,
                  NPC1>(imgHelper4, imgOutput, _kernel);

    // Convert _dst xf::cv::Mat object to output array:
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, OUT_TYPE, HEIGHT, WIDTH, NPC1>(imgOutput, img_out);

    return;

} // End of kernel
