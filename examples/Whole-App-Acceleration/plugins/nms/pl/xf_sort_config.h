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

#ifndef _XF_SORT_CONFIG_H_
#define _XF_SORT_CONFIG_H_

//#include "common/xf_common.hpp"
//#include "common/xf_utility.hpp"

#include "./xf_sort.hpp"

//total class
#define NUM_CLASS 91//4

//number of Class Process per class
#define NCPC 8

//TopK buffer depth
#define TOPK 80//4//150//4

//data bit width
#define DATA_BITS 8

//index bit width
#define INDEX_BITS 12

//index bit width for ddr write data
#define INDEX_BITS_OUT 16

//Calculate number of comparator, should be power of 2
#define NUM_CLASS_CEIL  ((NUM_CLASS + NCPC - 1)/NCPC)*NCPC
#define NUM_AVIL_CLKS  int(NUM_CLASS_CEIL/NCPC) // Floor vaue
#define NUM_COMP_TMP int((TOPK + NUM_AVIL_CLKS-1)/NUM_AVIL_CLKS) // Ceil value
#define NUM_COMP ( (NUM_COMP_TMP>16 && NUM_COMP_TMP<=32)*32 + (NUM_COMP_TMP>8 && NUM_COMP_TMP<=16)*16 + (NUM_COMP_TMP>4 && NUM_COMP_TMP<=8)*8 + (NUM_COMP_TMP>2 && NUM_COMP_TMP<=4)*4 + (NUM_COMP_TMP==2 || NUM_COMP_TMP==1)*NUM_COMP_TMP )  

// Set Port width
#define IN_PTR_WIDTH 8*NCPC
#define OUT_PTR_WIDTH 16*NUM_COMP
#define OUT_PTR_WIDTH_INDEX 16*NUM_COMP
#define OUT_PTR_WIDTH_SIZE 16


//SORT order 0-descending 1-ascending
#define SORT_ORDER 0

void sort_accel(ap_uint<IN_PTR_WIDTH>* in,
                ap_uint<OUT_PTR_WIDTH>* out_scoreIndex,
                int inputSize_perclass);

#endif //_XF_SORT_CONFIG_H_
