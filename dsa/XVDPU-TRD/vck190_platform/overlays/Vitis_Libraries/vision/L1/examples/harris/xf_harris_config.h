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

#ifndef _XF_HARRIS_CONFIG_H_
#define _XF_HARRIS_CONFIG_H_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "features/xf_harris.hpp"
#include "xf_config_params.h"

#define CH_TYPE XF_GRAY
#define INPUT_PTR_WIDTH 64
#define OUTPUT_PTR_WIDTH 64
/*
 * Pack pixels in and write into streams
 */

#if RO
#define NPIX XF_NPPC8
#endif
#if NO
#define NPIX XF_NPPC1
#endif

#define WIDTH 128
#define HEIGHT 128

#define IMGSIZE WIDTH* HEIGHT

#if RO
#define IN_TYPE ap_uint<64>
#endif
#if NO
#define IN_TYPE ap_uint<8>
#endif

void cornerHarris_accel(
    ap_uint<INPUT_PTR_WIDTH>* img_inp, ap_uint<OUTPUT_PTR_WIDTH>* img_out, int rows, int cols, int threshold, int k);
#endif
