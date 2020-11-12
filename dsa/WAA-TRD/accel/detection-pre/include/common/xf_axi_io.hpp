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

#ifndef ___XF__AXI_IO__
#define ___XF__AXI_IO__
#include "utils/x_hls_utils.h"
#include <assert.h>

namespace xf {
namespace cv {

template <int W, typename T>
void AXIGetBitFields(ap_uint<W> pix, int start, int w, T& val) {
// clang-format off
    #pragma HLS inline
    // clang-format on
    assert(start >= 0 && start + w <= W);
    val = (T)pix(start + w - 1, start);
}

template <int W>
void AXIGetBitFields(ap_uint<W> pix, int start, int w, float& val) {
// clang-format off
    #pragma HLS inline
    // clang-format on
    assert(w == 32 && start >= 0 && start + w <= W);
    fp_struct<float> temp((ap_uint<32>)pix(start + w - 1, start));
    val = temp.to_float();
}

template <int W>
void AXIGetBitFields(ap_uint<W> pix, int start, int w, double& val) {
// clang-format off
    #pragma HLS inline
    // clang-format on
    assert(w == 64 && start >= 0 && start + w <= W);
    fp_struct<double> temp((ap_uint<64>)pix(start + w - 1, start));
    val = temp.to_double();
}

template <int W, typename T>
void AXIGetBitFields(ap_axiu<W, 1, 1, 1> axi, int start, int w, T& val) {
// clang-format off
    #pragma HLS inline
    // clang-format on
    AXIGetBitFields(axi.data, start, w, val);
}

template <int W, typename T>
void AXISetBitFields(ap_uint<W>& pix, int start, int w, T val) {
// clang-format off
    #pragma HLS inline
    // clang-format on
    assert(start >= 0 && start + w <= W);
    pix(start + w - 1, start) = val;
}

template <int W>
void AXISetBitFields(ap_uint<W>& pix, int start, int w, float val) {
// clang-format off
    #pragma HLS inline
    // clang-format on
    assert(w == 32 && start >= 0 && start + w <= W);
    fp_struct<float> temp(val);
    pix(start + w - 1, start) = temp.data();
}

template <int W>
void AXISetBitFields(ap_uint<W>& pix, int start, int w, double val) {
// clang-format off
    #pragma HLS inline
    // clang-format on
    assert(w == 64 && start >= 0 && start + w <= W);
    fp_struct<double> temp(val);
    pix(start + w - 1, start) = temp.data();
}

template <int W, typename T>
void AXISetBitFields(ap_axiu<W, 1, 1, 1>& axi, int start, int w, T val) {
// clang-format off
    #pragma HLS inline
    // clang-format on
    AXISetBitFields(axi.data, start, w, val);
}

} // namespace cv
}; // namespace xf

#endif
