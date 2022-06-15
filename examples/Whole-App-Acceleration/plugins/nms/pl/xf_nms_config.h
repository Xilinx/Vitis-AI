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

#ifndef _XF_NMS_CONFIG_H_
#define _XF_NMS_CONFIG_H_

#include <ap_int.h>

//#define ConvertToFP(fVal, iPart, fbits)	((int)((iPart<<fbits) + ((fVal-(float)iPart))*(1<<fbits)))
#define ConvertToFP(fVal, fbits) ( fVal * (1 << fbits) )

// Set Port width
//#define IN_PTR_WIDTH 64
//#define OUT_PTR_WIDTH 64

//# Max entries in a class
//#define MAX_ENTRY_PER_CLASS 16

//# Data bitwidth
//#define DATABITS 16
#define STEP 16

#define MAX_BOX_PER_CLASS 32
#define TEST_CLASS 14

#define TEST_ENABLE 1
#define FIXED_ENABLE 1
#define ELE_PACK 4

#if FIXED_ENABLE==0
#define IN_DATATYPE float
#define OUT_DATATYPE float
#else
#if CONFIG1
#define IN_DATATYPE ap_int<8>
#define OUT_DATATYPE ap_int<8>
#else
#define IN_DATATYPE short int //ap_int<16>
#define OUT_DATATYPE short int//ap_int<16>
#endif
#endif

#define CONFIG_8BIT 0

#if CONFIG_8BIT
//#define FL_BOX 14
//#define BW_VAL 16384.0
//#define FL_BOX 3
//#define BW_VAL 8.0
#define NMS_INPTR int8_t
#define FL_SCORE 3
#define FL_ID 0
#define BW 8
#define FIX_TYPE ap_int<8>
#define MUL_BW 16
#else
#define NMS_INPTR short int
#define FL_BOX 14
#define FL_SCORE 14
#define FL_ID 0
#define BW 16
#define BW_VAL 16384.0
#define FIX_TYPE ap_int<16>
#define MUL_BW 32
#endif


struct box_e {
	ap_int<BW> x, y, w, h, cls_id, score;
};
typedef box_e s_box;

struct boxf_e {
	float x, y, w, h;
	int cls_id;
	float score;
};
typedef boxf_e sf_box;


//# Acceleration function
void nms_accel(
		ap_uint<IN_PTR_WIDTH>* inBoxes,
		ap_uint<OUT_PTR_WIDTH>* outBoxes,
		int num_entry_perclass,
		short int nms_th);

#endif //_XF_NMS_CONFIG_H_
