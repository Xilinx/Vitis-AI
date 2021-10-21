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

#ifndef __XF_VITIS_BFMATCHER_CONFIG_H__
#define __XF_VITIS_BFMATCHER_CONFIG_H__

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_bfmatcher.hpp"
#include "xf_config_params.h"

// Max keypoints for memory allocations
#define MAX_KEYPOINTS 10000

// hardware kernel declaration
void bfmatcher_accel(ap_uint<INPUT_PTR_WIDTH>* desc_list1,
                     ap_uint<INPUT_PTR_WIDTH>* desc_list2,
                     ap_int<OUTPUT_PTR_WIDTH>* desc_match_idx,
                     uint32_t num_keypoints1,
                     uint32_t num_keypoints2,
                     float ratio_thresh);

#endif // __XF_VITIS_BFMATCHER_CONFIG_H__
