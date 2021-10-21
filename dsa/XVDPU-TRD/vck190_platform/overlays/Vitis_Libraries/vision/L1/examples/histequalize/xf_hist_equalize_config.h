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

#ifndef _XF_HIST_EQUALIZE_CONFIG_H_
#define _XF_HIST_EQUALIZE_CONFIG_H_

#include "hls_stream.h"
#include "ap_int.h"
#include "xf_config_params.h"
#include "common/xf_common.hpp"
#include "imgproc/xf_hist_equalize.hpp"

/*  define the input and output types  */
#if NO
#define NPC_T XF_NPPC1
// port widths
#define INPUT_PTR_WIDTH 8
#define OUTPUT_PTR_WIDTH 8
#endif

#if RO
// port widths
#define INPUT_PTR_WIDTH 64
#define OUTPUT_PTR_WIDTH 64

#define NPC_T XF_NPPC8
#endif

// Maximum rows and cols
#define HEIGHT 128
#define WIDTH 128

void equalizeHist_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                        ap_uint<INPUT_PTR_WIDTH>* img_inp1,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                        int rows,
                        int cols);

#endif // _XF_HIST_EQUALIZE_CONFIG_H_
