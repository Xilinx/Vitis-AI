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

#ifndef _XF_MEAN_SHIFT_WRAPPER_HPP_
#define _XF_MEAN_SHIFT_WRAPPER_HPP_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "xf_mean_shift_kernel.hpp"

namespace xf {
namespace cv {
template <int MAXOBJ, int MAXITERS, int OBJ_ROWS, int OBJ_COLS, int SRC_T, int ROWS, int COLS, int NPC>
void MeanShift(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _in_mat,
               uint16_t* x1,
               uint16_t* y1,
               uint16_t* obj_height,
               uint16_t* obj_width,
               uint16_t* dx,
               uint16_t* dy,
               uint16_t* status,
               uint8_t frame_status,
               uint8_t no_objects,
               uint8_t no_iters) {
    // local arrays for memcopy
    uint16_t img_height[1], img_width[1], objects[1], frame[1];
    uint16_t tlx[MAXOBJ], tly[MAXOBJ], _obj_height[MAXOBJ], _obj_width[MAXOBJ], dispx[MAXOBJ], dispy[MAXOBJ];
    uint16_t track_status[MAXOBJ];

    for (int i = 0; i < no_objects; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAXOBJ
        #pragma HLS PIPELINE II=1
        // clang-format on
        tlx[i] = x1[i];
        tly[i] = y1[i];
        _obj_width[i] = obj_width[i];
        _obj_height[i] = obj_height[i];
        dispx[i] = dx[i];
        dispy[i] = dy[i];
        track_status[i] = status[i];
    }

    xFMeanShiftKernel<OBJ_ROWS, OBJ_COLS, SRC_T, ROWS, COLS, MAXOBJ, MAXITERS, NPC>(
        _in_mat, tlx, tly, _obj_height, _obj_width, dispx, dispy, track_status, frame_status, no_objects, no_iters);

    for (int i = 0; i < no_objects; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAXOBJ
        #pragma HLS PIPELINE II=1
        // clang-format on
        dx[i] = dispx[i];
        dy[i] = dispy[i];
        status[i] = track_status[i];
    }
}
} // namespace cv
} // namespace xf

#endif // _XF_MEAN_SHIFT_WRAPPER_HPP_
