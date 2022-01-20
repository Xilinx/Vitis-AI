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

#ifndef __XF_PYR_DENSE_OPTICAL_FLOW_CONFIG_TYPES__
#define __XF_PYR_DENSE_OPTICAL_FLOW_CONFIG_TYPES__
#include <stdint.h>
#include <stdlib.h>
#include "ap_int.h"

#define WINDOW_SIZE_FL 5

// FindIT operating and output width
#define TYPE_IT_WIDTH 17
#define TYPE_IT_INT 9
#define TYPE_IT_TYPE ap_fixed<TYPE_IT_WIDTH, TYPE_IT_INT>

// #define TYPE_ITCMP_WIDTH TYPE_FLOW_WIDTH + 12 + 4 //extra precision for interpolation
// #define TYPE_ITCMP_INT   TYPE_FLOW_INT + 12 //12 is the number of rows width. this is to compute the product i*flow
// #define TYPE_ITCMP_TYPE   ap_fixed<TYPE_ITCMP_WIDTH,TYPE_ITCMP_INT>

// sum IX2, Iy2 and IXIY width (2*9 + 7 + 2) 9 for IX IY, 7 for *121 window size, 2 fracs for adding 65.025
#define TYPE_SIXIY_WIDTH 27
#define TYPE_SIXIY_INT 25
#define TYPE_SIXIY_TYPE ap_fixed<TYPE_SIXIY_WIDTH, TYPE_SIXIY_INT>

// IT WIDTH + IX/IY width + *121 width = 17 + 9 + 7 = 33 + 1 bit padding, 8 bits for fractional
#define TYPE_SIXYIT_WIDTH 34
#define TYPE_SIXYIT_INT 26
#define TYPE_SIXYIT_TYPE ap_fixed<TYPE_SIXYIT_WIDTH, TYPE_SIXYIT_INT>

// this can be optimized. For computing interpolation
#define TYPE_RMAPPX_WIDTH 48
#define TYPE_RMAPPX_INT 16
#define TYPE_RMAPPX_TYPE ap_fixed<TYPE_RMAPPX_WIDTH, TYPE_RMAPPX_INT>

// Scale precision in scaleup module.
#define TYPE_SCALE_WIDTH 17
#define TYPE_SCALE_INT 1
#define TYPE_SCALE_TYPE ap_ufixed<TYPE_SCALE_WIDTH, TYPE_SCALE_INT>

// determinant precision based on the operation
#define TYPE_DET_WIDTH (TYPE_SIXIY_WIDTH * 2) + 1
#define TYPE_DET_INT (TYPE_SIXIY_INT * 2) + 1
#define TYPE_DET_TYPE ap_fixed<TYPE_DET_WIDTH, TYPE_DET_INT>

// 1/det precision. Fixed after trial and error. HUGE impact on the precision.
#define TYPE_DIVBY_WIDTH 40
#define TYPE_DIVBY_INT 2
#define TYPE_DIVBY_TYPE ap_fixed<TYPE_DIVBY_WIDTH, TYPE_DIVBY_INT>

// Flow computation precision
#define TYPE_FLCMP_WIDTH                        \
    (TYPE_SIXIY_WIDTH + TYPE_SIXYIT_WIDTH + 1 + \
     TYPE_DIVBY_WIDTH) // TODO: Optimize the width by truncating the bits to account for from Divide by
#define TYPE_FLCMP_INT (TYPE_SIXIY_INT + TYPE_SIXYIT_INT + 1 + TYPE_DIVBY_INT)
#define TYPE_FLCMP_TYPE ap_fixed<TYPE_FLCMP_WIDTH, TYPE_FLCMP_INT>

// Scale computation precision in scaleup module.
// TYPE_FLOW_WIDTH + TYPE_SCALE_WIDTH
// #define TYPE_SCCMP_WIDTH TYPE_FLOW_WIDTH + TYPE_SCALE_WIDTH + 12
// #define TYPE_SCCMP_INT   TYPE_FLOW_INT + 12 //12 is the number of rows width. this is to compute the product i*flow
// #define TYPE_SCCMP_TYPE   ap_fixed<TYPE_SCCMP_WIDTH,TYPE_SCCMP_INT>
#endif
