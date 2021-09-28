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

#ifndef _XF_HOG_DESCRIPTOR_CONFIG_H_
#define _XF_HOG_DESCRIPTOR_CONFIG_H_

#include "hls_stream.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_hog_descriptor.hpp"
#include "xf_config_params.h"

/* set the various hog parameters */
#define XF_WIN_STRIDE 8
#define XF_BLOCK_HEIGHT 16
#define XF_BLOCK_WIDTH 16
#define XF_CELL_HEIGHT 8
#define XF_CELL_WIDTH 8
#define XF_NO_OF_BINS 9

#define OUT_T1 ap_uint16_t

// Various parameters used for testing purpose and template usages, not to be edited
#define XF_NOVCPB (XF_BLOCK_HEIGHT / XF_CELL_HEIGHT)                // number of vertical cells per block
#define XF_NOHCPB (XF_BLOCK_WIDTH / XF_CELL_WIDTH)                  // number of horizontal cells per block
#define XF_NOBPB (XF_NO_OF_BINS * XF_NOHCPB * XF_NOVCPB)            // number of bins per block
#define XF_NOVBPW ((XF_WIN_HEIGHT / XF_CELL_HEIGHT) - 1)            // number of vertical blocks per window
#define XF_NOHBPW ((XF_WIN_WIDTH / XF_CELL_WIDTH) - 1)              // number of horizontal blocks per window
#define XF_NODPW (XF_NOBPB * XF_NOVBPW * XF_NOHBPW)                 // number of descriptors per window
#define XF_NOVW (((XF_HEIGHT - XF_WIN_HEIGHT) / XF_WIN_STRIDE) + 1) // number of vertical windows in the image
#define XF_NOHW (((XF_WIDTH - XF_WIN_WIDTH) / XF_WIN_STRIDE) + 1)   // number of horizontal windows in the image
#define XF_NOVB ((XF_HEIGHT / XF_CELL_HEIGHT) - 1)                  // number of vertical blocks in the image
#define XF_NOHB ((XF_WIDTH / XF_CELL_WIDTH) - 1)                    // number of horizontal blocks in the image

#if REPETITIVE_BLOCKS
#define XF_DESC_SIZE ((XF_NOVW * XF_NOHW * XF_NODPW) >> 1)
#define XF_OUTPUT_MODE XF_HOG_RB
#elif NON_REPETITIVE_BLOCKS
#define XF_DESC_SIZE ((XF_NOVB * XF_NOHB * XF_NOBPB) >> 1)
#define XF_OUTPUT_MODE XF_HOG_NRB
#endif

#if GRAY_T
#define IN_TYPE XF_8UC1
#define XF_INPUT_COLOR XF_GRAY
#elif RGB_T
#define IN_TYPE XF_8UC3
#define XF_INPUT_COLOR XF_RGB
#endif

// Set the output pixel depth:
#define OUT_TYPE XF_32UC1

// Set the optimization type: // Only XF_NPPC1 is supported currently for HoG
#define NPC XF_NPPC1

// setting pointer widths for interface
#define INPUT_PTR_WIDTH 32
#define OUTPUT_PTR_WIDTH 32
//

void hog_descriptor_accel(
    ap_uint<INPUT_PTR_WIDTH>* img_in, ap_uint<OUTPUT_PTR_WIDTH>* desc_out, int rows, int cols, int _desc_size);

#endif // end of _XF_HOG_CONFIG_H_
