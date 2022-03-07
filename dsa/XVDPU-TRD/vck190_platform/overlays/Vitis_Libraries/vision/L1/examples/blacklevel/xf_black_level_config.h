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

#ifndef __XF_BLACK_LEVEL_CONFIG_H__
#define __XF_BLACK_LEVEL_CONFIG_H__

#include "imgproc/xf_black_level.hpp"

#define T_8U 1
#define T_16U 0
#define IMAGE_PTR_WIDTH 8
#define IMAGE_THRES_WIDTH 8
#define IMAGE_MUL_WIDTH 16
#define IMAGE_MUL_FL_BITS 8
#define IMAGE_SIZE_WIDTH 16

#define XF_MAX_ROWS 128
#define XF_MAX_COLS 128
#define XF_SRC_T XF_8UC1
#define XF_NPPC XF_NPPC1
#define XF_USE_DSP 1

#endif // __XF_BLACK_LEVEL_CONFIG_H__
