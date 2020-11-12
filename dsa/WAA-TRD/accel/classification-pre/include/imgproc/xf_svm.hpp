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

#ifndef _XF_SVM_H_
#define _XF_SVM_H_

#include "common/xf_common.hpp"

/****************************************************************
 *  xFSVM: SVM core computation (dot product function)
 *  ------
 *
 *  Inputs:
 *  -------
 *  in_1: Input array 1
 *  in_2: Input array 2
 *  idx1: Starting index of Input array 1
 *  idx2: Starting index of Input array 2
 *  frac1: Fractional bits in the Input array 1
 *  frac2: Fractional bits in the Input array 2
 *  n: number of kernel operations
 *
 *  Output:
 *  -------
 *  out_frac: Fractional bits in the output result
 *
 ***************************************************************/
namespace xf {
namespace cv {

template <int SRC1_T, int SRC2_T, int DST_T, int ROWS1, int COLS1, int ROWS2, int COLS2, int NPC, int N>
ap_int<XF_PIXELDEPTH(DST_T)> xfSVM(xf::cv::Mat<SRC1_T, ROWS1, COLS1, NPC>& in_1,
                                   xf::cv::Mat<SRC2_T, ROWS2, COLS2, NPC>& in_2,
                                   uint16_t idx1,
                                   uint16_t idx2,
                                   uchar_t frac1,
                                   uchar_t frac2,
                                   uint16_t n,
                                   uchar_t* out_frac) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    // temporary result
    ap_int<XF_PIXELDEPTH(DST_T)> result = 0;

svmCoreLoop:
    for (int i = 0; i < n; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=N max=N avg=N
        #pragma HLS PIPELINE
        // clang-format on

        // Dot product operation
        ap_int<XF_PIXELDEPTH(DST_T)> tmp_svm =
            (ap_int<XF_PIXELDEPTH(DST_T)>)(in_1.read(idx1 + i) * in_2.read(idx2 + i));
        result += tmp_svm;
    }

    *out_frac = frac1 + frac2;
    return result;
}

template <int SRC1_T, int SRC2_T, int DST_T, int ROWS1, int COLS1, int ROWS2, int COLS2, int NPC = 1, int N>
void SVM(xf::cv::Mat<SRC1_T, ROWS1, COLS1, NPC>& in_1,
         xf::cv::Mat<SRC2_T, ROWS2, COLS2, NPC>& in_2,
         uint16_t idx1,
         uint16_t idx2,
         uchar_t frac1,
         uchar_t frac2,
         uint16_t n,
         uchar_t* out_frac,
         ap_int<XF_PIXELDEPTH(DST_T)>* result) {
#ifndef __SYNTHESIS__
    assert(((SRC1_T == XF_16SC1)) && "Only 16 bit, single channel images are supported");
    assert(((SRC2_T == XF_16SC1)) && "Only 16 bit, single channel images are supported");
#endif
    ap_int<XF_PIXELDEPTH(DST_T)> svm_res = xfSVM<SRC1_T, SRC2_T, DST_T, ROWS1, COLS1, ROWS2, COLS2, NPC, N>(
        in_1, in_2, idx1, idx2, frac1, frac2, n, out_frac);

    *result = svm_res;
}
} // namespace cv
} // namespace xf
#endif
