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
#pragma once

#define GL_TEST 0

#include <ap_int.h>
#include <hls_stream.h>
//#include "xf_arrdata_config.h"
//#include "xf_nms_config.h"
#include "xf_sort_config.h"
#include "xf_sort.hpp"
#include "xf_arrdata.hpp"
#include "xf_nms.hpp"



// Set Port width
#define GMEM_GL_IN_PTR_WIDTH    32
#define GMEM_GL_OUT_PTR_WIDTH   128
#define GMEM_GL_PRIOR_PTR_WIDTH 64
#define GMEM_LUT_PTR_WIDTH 		16
#define GMEM_BOX_IN_PTR_WIDTH	8

#define GMEM_NMS_PTR_WIDTH  128

#define CONFIG_MAX_BOX_PER_CLASS 32
/*
void sort_nms_accel(
		ap_uint<IN_PTR_WIDTH>* inConf,
		ap_uint<GMEM_BOX_IN_PTR_WIDTH>* inBoxes1,
		ap_uint<GMEM_BOX_IN_PTR_WIDTH>* inBoxes2,
		ap_uint<GMEM_BOX_IN_PTR_WIDTH>* inBoxes3,
		ap_uint<GMEM_BOX_IN_PTR_WIDTH>* inBoxes4,
		ap_uint<GMEM_GL_PRIOR_PTR_WIDTH>* priors,
		ap_uint<GMEM_GL_OUT_PTR_WIDTH>* outBoxes,
		int inputSize_perclass,
		short int nms_th);
*/		
