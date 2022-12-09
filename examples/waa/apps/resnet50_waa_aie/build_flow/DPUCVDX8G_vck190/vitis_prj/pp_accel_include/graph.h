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

#ifndef ADF_GRAPH_RESIZE_NORM_H
#define ADF_GRAPH_RESIZE_NORM_H

#include <adf.h>
#include <array>
#include <type_traits>

#include "kernels.h"

using namespace adf;

/*
 * ADF graph to compute weighted moving average of
 * the last 8 samples in a stream of numbers
 */

template <int M, int N>
class PosArray {
   public:
    std::array<uint16, N> arr;
    PosArray() : arr() {
        for (int c = 0; c != N; ++c) {
            float x_scale = (float)M / (float)N;
            float idx_x_pre = ((c + 0.5) * x_scale) - 0.5;
            float idx_x_pre_pos_1 = (idx_x_pre > (M - 1)) * (M - 1);
            float idx_x_pre_pos_2 = ((idx_x_pre >= 0) && (idx_x_pre < M)) * idx_x_pre;
            float idx_x_pre_pos = idx_x_pre_pos_1 + idx_x_pre_pos_2;
            int idx_x = (int)idx_x_pre_pos;
            arr[c] = idx_x;
        }
    }
};

template <int M, int N>
class WtsArrayX {
   public:
    std::array<uint8, N * 4> arr;
    WtsArrayX() : arr() {
        for (int c = 0; c != N; ++c) {
            float x_scale = (float)M / (float)N;
            float idx_x_pre = ((c + 0.5) * x_scale) - 0.5;
            float idx_x_pre_pos_1 = (idx_x_pre > (M - 1)) * (M - 1);
            float idx_x_pre_pos_2 = ((idx_x_pre >= 0) && (idx_x_pre < M)) * idx_x_pre;
            float idx_x_pre_pos = idx_x_pre_pos_1 + idx_x_pre_pos_2;
            int idx_x = (int)idx_x_pre_pos;
            uint8 Wx = (idx_x_pre_pos - idx_x) * 256;
            arr[((c / 4) * 16) + 2 * (c % 4) + 0] = 255 - Wx;
            arr[((c / 4) * 16) + 2 * (c % 4) + 1] = Wx;
            arr[((c / 4) * 16) + 2 * (c % 4) + 8] = 255 - Wx;
            arr[((c / 4) * 16) + 2 * (c % 4) + 9] = Wx;
        }
    }
};

template <int M, int N>
class WtsArrayY {
   public:
    std::array<uint8, N> arr;
    WtsArrayY() : arr() {
        for (int c = 0; c != N; ++c) {
            float x_scale = (float)M / (float)N;
            float idx_x_pre = ((c + 0.5) * x_scale) - 0.5;
            float idx_x_pre_pos_1 = (idx_x_pre > (M - 1)) * (M - 1);
            float idx_x_pre_pos_2 = ((idx_x_pre >= 0) && (idx_x_pre < M)) * idx_x_pre;
            float idx_x_pre_pos = idx_x_pre_pos_1 + idx_x_pre_pos_2;
            int idx_x = (int)idx_x_pre_pos;
            uint8 Wx = (idx_x_pre_pos - idx_x) * 256;
            arr[c] = 255 - Wx;
        }
    }
};

auto pos = PosArray<IMAGE_WIDTH_IN, IMAGE_WIDTH_OUT>();
auto wtsX = WtsArrayX<IMAGE_WIDTH_IN, IMAGE_WIDTH_OUT>();
auto wtsY = WtsArrayY<IMAGE_HEIGHT_IN, IMAGE_HEIGHT_OUT>();

class resizeNormGraph : public adf::graph {
   private:
    kernel k;

   public:
    input_plio in1;
    output_plio out1;
    port<input> a0;
    port<input> a1;
    port<input> a2;
    port<input> a3;
    port<input> b0;
    port<input> b1;
    port<input> b2;
    port<input> b3;

    resizeNormGraph() {
        k = kernel::create_object<
            ResizeNormRunner<TILE_WIDTH_IN, TILE_HEIGHT_IN, TILE_WIDTH_OUT, TILE_HEIGHT_OUT, IMAGE_HEIGHT_OUT> >(
            pos.arr, wtsX.arr, wtsY.arr);
        in1 = input_plio::create("DataIn0", adf::plio_128_bits, "data/input.txt");
        out1 = output_plio::create("DataOut0", adf::plio_128_bits, "data/output.txt");

        // create nets to connect kernels and IO ports
        connect<window<TILE_WINDOW_SIZE_IN> >(in1.out[0], k.in[0]);
        connect<window<TILE_WINDOW_SIZE_OUT> >(k.out[0], out1.in[0]);
        //connect<parameter>(row, async(k.in[1]));
        connect<parameter>(a0, async(k.in[1]));
        connect<parameter>(a1, async(k.in[2]));
        connect<parameter>(a2, async(k.in[3]));
        connect<parameter>(a3, async(k.in[4]));
        connect<parameter>(b0, async(k.in[5]));
        connect<parameter>(b1, async(k.in[6]));
        connect<parameter>(b2, async(k.in[7]));
        connect<parameter>(b3, async(k.in[8]));

        location<parameter>(k.param[1]) = offset(0);

        // specify kernel sources
        source(k) = "xf_resize_normalize.cc";

        runtime<ratio>(k) = 1.0;
    }
};

#endif
