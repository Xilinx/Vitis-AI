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

#ifndef _XF_MEAN_SHIFT_CONFIG_H
#define _XF_MEAN_SHIFT_CONFIG_H

#include "hls_stream.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_mean_shift.hpp"
#include "xf_config_params.h"

// set 1 for video
#define VIDEO_INPUT 0

// total number of frames to be tracked
#define TOTAL_FRAMES 100

#define XF_HEIGHT 1080
#define XF_WIDTH 1920

// Modify the information on objects to be tracked
// coordinate system uses (row, col) where row = 0 and col = 0 correspond to the top-left corner of the input image
const int X1[XF_MAX_OBJECTS] = {50, 170,
                                300}; // col coordinates of the top-left corner of all the objects to be tracked
const int Y1[XF_MAX_OBJECTS] = {50, 200,
                                400}; // row coordinates of the top-left corner of all the objects to be tracked
const int WIDTH_MST[XF_MAX_OBJECTS] = {100, 100,
                                       100}; // width of all the objects to be tracked (measured from top-left corner)
const int HEIGHT_MST[XF_MAX_OBJECTS] = {100, 100,
                                        100}; // height of all the objects to be tracked (measured from top-left corner)

#define IN_TYPE unsigned int

#define INPUT_PTR_WIDTH 32

void mean_shift_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                      uint16_t* tlx,
                      uint16_t* tly,
                      uint16_t* obj_height,
                      uint16_t* obj_width,
                      uint16_t* dx,
                      uint16_t* dy,
                      uint16_t* track,
                      uint8_t frame_status,
                      uint8_t no_objects,
                      uint8_t no_of_iterations,
                      int rows,
                      int cols);

#endif
