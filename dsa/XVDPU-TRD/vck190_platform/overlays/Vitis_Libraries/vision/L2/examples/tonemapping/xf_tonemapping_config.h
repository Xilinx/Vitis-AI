/*
 * Copyright 2020 Xilinx, Inc.
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

#ifndef _XF_LTM_CONFIG_HPP_
#define _XF_LTM_CONFIG_HPP_
#include "imgproc/xf_ltm.hpp"

/*  User configurable parameters */
static constexpr int HEIGHT = 676;
static constexpr int WIDTH = 1024;

static constexpr int IN_TYPE = XF_16UC3;
static constexpr int OUT_TYPE = XF_8UC3;

static constexpr int IN_PTR_WIDTH = 512;
static constexpr int OUT_PTR_WIDTH = 512;
static constexpr int BLOCK_HEIGHT = 64;
static constexpr int BLOCK_WIDTH = 64;

static constexpr int NPC = XF_NPPC4;

extern "C" {
void tonemapping_accel(ap_uint<IN_PTR_WIDTH>* in_ptr,
                       ap_uint<OUT_PTR_WIDTH>* out_ptr,
                       int height,
                       int width,
                       int blk_height,
                       int blk_width);
}
#endif
