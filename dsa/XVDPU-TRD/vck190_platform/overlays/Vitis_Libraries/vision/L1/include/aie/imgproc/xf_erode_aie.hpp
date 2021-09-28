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

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <adf.h>
#include <adf/stream/types.h>
#include <adf/window/types.h>
#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>
#include <common/xf_aie_utils.hpp>

#ifndef _AIE_ERODE_H_
#define _AIE_ERODE_H_

namespace xf {
namespace cv {
namespace aie {

template <typename T, int VECTORIZATION_FACTOR>
void erode_rect_3x3_api(input_window<T>* img_in, output_window<T>* img_out) {
    T* img_in_ptr = (T*)img_in->ptr;
    T* img_out_ptr = (T*)img_out->ptr;

    const int16_t img_width = xfcvGetTileWidth(img_in_ptr);
    const int16_t img_height = xfcvGetTileHeight(img_in_ptr);

    xfcvCopyMetaData(img_in_ptr, img_out_ptr);

    T* _input = (T*)xfcvGetImgDataPtr(img_in_ptr);
    T* _res = (T*)xfcvGetImgDataPtr(img_out_ptr);

    auto out = ::aie::begin_vector<VECTORIZATION_FACTOR>(_res);

    ::aie::vector<T, VECTORIZATION_FACTOR> A;
    ::aie::vector<T, VECTORIZATION_FACTOR> B;
    ::aie::vector<T, VECTORIZATION_FACTOR> min;

    // produce res
    for (int _res_s0_y = 0; _res_s0_y < img_height; _res_s0_y++) {
        T* _input_r[3];
        _input_r[0] = _input + std::max((_res_s0_y - 1), 0) * img_width;
        _input_r[1] = _input + (_res_s0_y)*img_width;
        _input_r[2] = _input + std::min((_res_s0_y + 1), (img_height - 1)) * img_width;

        //@Left border {
        {
            B = ::aie::load_v<VECTORIZATION_FACTOR>(_input_r[0]);
            _input_r[0] += VECTORIZATION_FACTOR;
            A = ::aie::shuffle_up(B, 1);
            A[0] = B[0];
            min = ::aie::min(A, B);

            A = ::aie::shuffle_down(B, 1);
            A[(VECTORIZATION_FACTOR - 1)] = *(_input_r[0]);
            min = ::aie::min(min, A);

            B = ::aie::load_v<VECTORIZATION_FACTOR>(_input_r[1]);
            _input_r[1] += VECTORIZATION_FACTOR;
            min = ::aie::min(min, B);
            A = ::aie::shuffle_up(B, 1);
            A[0] = B[0];
            min = ::aie::min(min, A);
            A = ::aie::shuffle_down(B, 1);
            A[(VECTORIZATION_FACTOR - 1)] = *(_input_r[1]);
            min = ::aie::min(min, A);

            B = ::aie::load_v<VECTORIZATION_FACTOR>(_input_r[2]);
            _input_r[2] += VECTORIZATION_FACTOR;
            min = ::aie::min(min, B);
            A = ::aie::shuffle_up(B, 1);
            A[0] = B[0];
            min = ::aie::min(min, A);
            A = ::aie::shuffle_down(B, 1);
            A[(VECTORIZATION_FACTOR - 1)] = *(_input_r[2]);
            min = ::aie::min(min, A);

            *out++ = min;
        }
        //@}

        //@Middle reigon {
        {
            for (int _res_s0_x = VECTORIZATION_FACTOR; _res_s0_x < (img_width - VECTORIZATION_FACTOR);
                 _res_s0_x += VECTORIZATION_FACTOR)
                chess_prepare_for_pipelining {
                    // y-1
                    B = ::aie::load_v<VECTORIZATION_FACTOR>(_input_r[0]);
                    A = ::aie::shuffle_up(B, 1);
                    A[0] = *(_input_r[0] - 1);
                    min = ::aie::min(A, B);

                    _input_r[0] += VECTORIZATION_FACTOR;
                    A = ::aie::shuffle_down(B, 1);
                    A[VECTORIZATION_FACTOR - 1] = *(_input_r[0]);
                    min = ::aie::min(min, A);

                    // y
                    B = ::aie::load_v<VECTORIZATION_FACTOR>(_input_r[1]);
                    min = ::aie::min(min, B);
                    A = ::aie::shuffle_up(B, 1);
                    A[0] = *(_input_r[1] - 1);
                    min = ::aie::min(min, A);

                    _input_r[1] += VECTORIZATION_FACTOR;
                    A = ::aie::shuffle_down(B, 1);
                    A[(VECTORIZATION_FACTOR - 1)] = *(_input_r[1]);
                    min = ::aie::min(min, A);

                    // y+1
                    B = ::aie::load_v<VECTORIZATION_FACTOR>(_input_r[2]);
                    min = ::aie::min(min, B);
                    A = ::aie::shuffle_up(B, 1);
                    A[0] = *(_input_r[2] - 1);
                    min = ::aie::min(min, A);

                    _input_r[2] += VECTORIZATION_FACTOR;
                    A = ::aie::shuffle_down(B, 1);
                    A[(VECTORIZATION_FACTOR - 1)] = *(_input_r[2]);
                    min = ::aie::min(min, A);
                    *out++ = min;
                }
        }
        //@}

        //@Right border {
        {
            B = ::aie::load_v<VECTORIZATION_FACTOR>(_input_r[0]);
            A = ::aie::shuffle_up(B, 1);
            A[0] = *(_input_r[0] - 1);
            min = ::aie::min(A, B);
            A = ::aie::shuffle_down(B, 1);
            A[(VECTORIZATION_FACTOR - 1)] = B[(VECTORIZATION_FACTOR - 1)];
            min = ::aie::min(min, A);

            B = ::aie::load_v<VECTORIZATION_FACTOR>(_input_r[1]);
            min = ::aie::min(min, B);
            A = ::aie::shuffle_up(B, 1);
            A[0] = *(_input_r[1] - 1);
            min = ::aie::min(min, A);
            A = ::aie::shuffle_down(B, 1);
            A[(VECTORIZATION_FACTOR - 1)] = B[(VECTORIZATION_FACTOR - 1)];
            min = ::aie::min(min, A);

            B = ::aie::load_v<VECTORIZATION_FACTOR>(_input_r[2]);
            min = ::aie::min(min, B);
            A = ::aie::shuffle_up(B, 1);
            A[0] = *(_input_r[2] - 1);
            min = ::aie::min(min, A);
            A = ::aie::shuffle_down(B, 1);
            A[(VECTORIZATION_FACTOR - 1)] = B[(VECTORIZATION_FACTOR - 1)];
            min = ::aie::min(min, A);

            *out++ = min;
        }
        //@}
    } // for _res_s0_y
}

} // aie
} // cv
} // xf

#endif
