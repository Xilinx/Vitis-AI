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

#ifndef _XF_KALMANFILTER_CONFIG_H_
#define _XF_KALMANFILTER_CONFIG_H_

#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "xf_config_params.h"

#include "video/xf_kalmanfilter.hpp"

// Set the pixel depth:
#define TYPE XF_32FC1
#define PTR_WIDTH 32

// Set the optimization type:
#define NPC1 XF_NPPC1

void kalmanfilter_accel(ap_uint<32>* in_A,
                        ap_uint<32>* in_B,
                        ap_uint<32>* in_Uq,
                        ap_uint<32>* in_Dq,
                        ap_uint<32>* in_H,
                        ap_uint<32>* in_X0,
                        ap_uint<32>* in_U0,
                        ap_uint<32>* in_D0,
                        ap_uint<32>* in_R,
                        ap_uint<32>* in_u,
                        ap_uint<32>* in_y,
                        unsigned char control_flag,
                        ap_uint<32>* out_X,
                        ap_uint<32>* out_U,
                        ap_uint<32>* out_D);

#endif //_XF_KALMANFILTER_CONFIG_H_
