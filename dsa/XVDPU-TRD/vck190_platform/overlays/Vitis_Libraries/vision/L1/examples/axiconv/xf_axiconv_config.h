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

#ifndef _XF_AXICONV_CONFIG_H_
#define _XF_AXICONV_CONFIG_H_

#include "hls_stream.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "common/xf_infra.hpp"
#include "xf_config_params.h"

/* config width and height */
#define XF_HEIGHT 720
#define XF_WIDTH 1280

#define IN_TYPE ap_uint<8>
#define OUT_TYPE ap_uint<8>

#define INPUT_PTR_WIDTH 32
#define OUTPUT_PTR_WIDTH 32

void axiconv_accel(hls::stream<ap_axiu<8, 1, 1, 1> >& _src,
                   hls::stream<ap_axiu<8, 1, 1, 1> >& _dst,
                   int rows,
                   int cols);

#endif // _XF_AXICONV_CONFIG_H_
