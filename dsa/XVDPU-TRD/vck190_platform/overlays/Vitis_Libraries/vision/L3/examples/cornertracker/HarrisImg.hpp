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

void HarrisImg(ap_uint<INPUT_PTR_WIDTH>* inHarris,
               unsigned int* list,
               unsigned int* params,
               int harris_rows,
               int harris_cols,
               uint16_t Thresh,
               uint16_t k,
               uint32_t* nCorners,
               bool harris_flag) {
    const int pROWS = HEIGHT;
    const int pCOLS = WIDTH;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1> in_harris_mat(harris_rows, harris_cols);
// clang-format off
    #pragma HLS stream variable=in_harris_mat.data depth=2
    // clang-format on
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1> out_harris_mat(harris_rows, harris_cols);
// clang-format off
    #pragma HLS stream variable=out_harris_mat.data depth=2
// clang-format on

// clang-format off
        #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, XF_NPPC1>(inHarris, in_harris_mat);
    xf::cv::cornerHarris<FILTER_WIDTH, BLOCK_WIDTH, NMS_RADIUS, XF_8UC1, HEIGHT, WIDTH, XF_NPPC1>(
        in_harris_mat, out_harris_mat, Thresh, k);

    xf::cv::cornersImgToList<MAXCORNERS, XF_8UC1, HEIGHT, WIDTH, XF_NPPC1>(out_harris_mat, list, nCorners);
}
