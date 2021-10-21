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

/**
 *  @brief FPGA FD accelerator kernel
 *
 *  $DateTime: 2018/02/05 02:36:41 $
 */

#ifndef _XF_FINTECH_KERNEL_FDLV_HPP_
#define _XF_FINTECH_KERNEL_FDLV_HPP_

extern "C" {

void fd_bs_lv_kernel(ap_uint<512>* xGrid,
                     ap_uint<512>* tGrid,
                     ap_uint<512>* sigma,
                     ap_uint<512>* rate,
                     ap_uint<512>* initialCondition,
                     float theta,
                     DT boundaryLower,
                     DT boundaryUpper,
                     unsigned int tSteps,
                     ap_uint<512>* solution);
}

#endif