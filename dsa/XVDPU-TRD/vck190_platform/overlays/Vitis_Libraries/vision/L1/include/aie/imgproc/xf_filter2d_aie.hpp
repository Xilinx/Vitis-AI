/*
 * Copyright 2021 Xilinx, Inc.
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

#include <adf.h>
//#include "../include.h"

#ifndef _AIE_FILTER2D_H_
#define _AIE_FILTER2D_H_

namespace xf {
namespace cv {
namespace aie {

int32_t kernel_coeff[16] = {64, 128, 64, 128, 256, 128, 64, 128, 64};
int image_width = 128;
int image_height = 32;
int stride = 128;

/**
 * ----------------------------------------------------------------------------
 * 32-bit 2D filter with in kernel border support
 * ----------------------------------------------------------------------------
 * Kernel coefficients (3x3):  k0 k1 k2 k3 k4 k5 k6 k7  | k8   -  -  -  -  -  -
 *
 * Image data (by rows):       d0  1  2  3  4  5  6  7  8  9   -  -  -  -  -  -
 *                             d16 17 18 19 20 21 22 23 24 25  -  -  -  -  -  -
 *                             d32 33 34 35 36 37 38 39 40 41  -  -  -  -  -  -
 *
 * Intrinsic multiplier:       o0  = k0*(d0)
 *                             o1  = k0*(d1)
 *                             o2  = k0*(d2)
 *                             o3  = k0*(d3)
 *                             o4  = k0*(d4)
 *                             o5  = k0*(d5)
 *                             o6  = k0*(d6)
 *                             o7  = k0*(d7)
 *
 * ----------------------------------------------------------------------------
 * Kernel vector multiplication:   Data:                                Kernel:
 * ----------------------------------------------------------------------------
 * o0..7  = k0*(d0..7)            d0  1  2  3  4  5  6  7              k0 k1 k2
 *       += k1*(d1..8)                1  2  3  4  5  6  7  8           k3 k4 k5
 *       += k2*(d2..9)                   2  3  4  5  6  7  8  9        k6 k7 k8
 *       += k3*(d16..23)          d16 17 18 19 20 21 22 23
 *       += k4*(d17..24)              17 18 19 20 21 22 23 24
 *       += k5*(d18..25)                 18 19 20 21 22 23 24 25
 *       += k6*(d32..39)          d32 33 34 35 36 37 38 39
 *       += k7*(d33..40)              33 34 35 36 37 38 39 40
 *       += k8*(d34..41)                 34 35 36 37 38 39 40 41
 *
 * ----------------------------------------------------------------------------
 * lmac8 intrinsic
 * ----------------------------------------------------------------------------
 * acc = lmac8( acc,        data_buf ,              n ,       0x76543210 ,  kernel_vec0,             0, zoffset);
 *               ^              ^                   ^              ^             ^                   ^             ^
 *        (accumulator) (single large buffer) (start x index) (x offsets) (coefficient buffer) (start z index) (z
 * offsets)
 *                                                            (per lane )                                      (per lane
 * )
 *                                            (     32b granularity     )                      (    32b granularity )
 *
 * ----------------------------------------------------------------------------
 * Image regions for border effect handling
 * ----------------------------------------------------------------------------
 *
 *   _____________________________
 *  |  |                       |  |
 *  |6_|__________2____________|7_|  first row
 *  |  |                       |  |
 *  |  |                       |  |
 *  |  |                       |  |
 *  |4 |          1            |5 |
 *  |  |                       |  |
 *  |  |                       |  |
 *  |__|_______________________|__|
 *  |  |                       |  |
 *  |8_|__________3____________|9_|  last row
 *
 */

void filter2D_api(input_window_int32* img_in, output_window_int32* img_out) {
    v8int32* restrict ptr_img_buffer = (v8int32*)img_in->ptr;
    v8int32* restrict ptr_img_out = (v8int32*)img_out->ptr;

    v32int32 data_buf;
    v16int32 data_buf1;
    v16int32 data_buf2;
    v8int32 data_out;
    v8acc80 acc;

    v8int32* restrict ptr_coeff_buffer = (v8int32*)kernel_coeff;
    v8int32 kernel_vec0 = *(ptr_coeff_buffer)++; // 1st 8 kernel values (0 .. 7)
    v8int32 kernel_vec1 = *(ptr_coeff_buffer)++; // 2nd 8 kernel values (8 .. 15)

    v8int32* restrict ptr0 = ptr_img_buffer;
    v8int32* restrict ptr1 = ptr_img_buffer + 1 * stride / PARALLEL_FACTOR_32b;
    v8int32* restrict ptr2 = ptr_img_buffer + 2 * stride / PARALLEL_FACTOR_32b;
    v8int32* restrict ptr_out = ptr_img_out;

    // 3x3 kernel positions
    //
    // 0 1 2
    // 3 4 5
    // 6 7 8

    // **************************************************************************
    // Unrolling loops over rows and columns to support the different image regions
    // **************************************************************************

    // **************************************************************************
    // First row filtering, regions 6, 2, 7
    // **************************************************************************
    {
        // **************************************************************************
        // Region 6
        // **************************************************************************

        // row 1 data used twice (or we use 0)
        data_buf = upd_w(data_buf, 0, *(ptr0++));                     // r1:00++07|_________|_________|_________
        acc = lmul8(data_buf, 0, 0x65432100, kernel_vec0, 0, 0);      // kernel 0 (r1:00,00..06)
        acc = lmac8(acc, data_buf, 0, 0x76543210, kernel_vec0, 1, 0); // kernel 1 (r1:00..07)
        data_buf = upd_w(data_buf, 1, *(ptr0--));                     // r1:00..07|r1:08++15|_________|_________
        acc = lmac8(acc, data_buf, 1, 0x76543210, kernel_vec0, 2, 0); // kernel 2 (r1:01..08)

        // 2nd row (uses same data as row 1)
        acc = lmac8(acc, data_buf, 0, 0x65432100, kernel_vec0, 3, 0); // kernel 3 (r1:00,00..06)
        acc = lmac8(acc, data_buf, 0, 0x76543210, kernel_vec0, 4, 0); // kernel 4 (r1:00..07)
        acc = lmac8(acc, data_buf, 1, 0x76543210, kernel_vec0, 5, 0); // kernel 5 (r1:01..08)

        // 3rd row (uses row 2 data)
        data_buf = upd_w(data_buf, 0, *(ptr1++));                     // r2:00++07|_________|_________|_________
        acc = lmac8(acc, data_buf, 0, 0x65432100, kernel_vec0, 6, 0); // kernel 6 (r2:00,00..06)
        acc = lmac8(acc, data_buf, 0, 0x76543210, kernel_vec0, 7, 0); // kernel 6 (r2:00..07)
        data_buf = upd_w(data_buf, 1, *(ptr1--));                     // r2:00..07|r1:08++15|_________|_________
        acc = lmac8(acc, data_buf, 1, 0x76543210, kernel_vec1, 0, 0); // kernel 8 (r2:01..08)

        // Store result
        data_out = srs(acc, SRS_SHIFT);
        *(ptr_out++) = data_out;

        ptr0++;
        ptr1++;
        // **************************************************************************
        // Region 2
        // **************************************************************************
        // row 1 data used twice (or we use 0)
        for (int j = 0; j < image_width - 2 * PARALLEL_FACTOR_32b; j += PARALLEL_FACTOR_32b) // 8x samples per loop
            chess_prepare_for_pipelining {
                // row 1 data used twice (or we use 0)
                data_buf1 = upd_w(data_buf1, 1, *(ptr0--));               // _________|r1:08++15|_________|_________
                acc = lmul8(data_buf1, 8, 0x76543210, kernel_vec0, 1, 0); // kernel 1 (r1:08..15)
                data_buf1 = upd_w(data_buf1, 0, *(ptr0));                 // r1:00++07|r1:08..15|_________|_________
                ptr0 = ptr0 + 2;
                acc = lmac8(acc, data_buf1, 7, 0x76543210, kernel_vec0, 0, 0); // kernel 0 (r1:07..14)
                data_buf1 = upd_w(data_buf1, 0, *(ptr0)); // r1:16++23|r1:08..15|_________|_________
                ptr0 = ptr0 - 2;
                acc = lmac8(acc, data_buf1, 9, 0x76543210, kernel_vec0, 2, 0); // kernel 2 (r1:09..16)

                // 2nd row
                acc = lmac8(acc, data_buf1, 9, 0x76543210, kernel_vec0, 5, 0); // kernel 5 (r1:09..16)
                data_buf1 = upd_w(data_buf1, 0, *(ptr0)); // r1:00++07|r1:08..15|_________|_________
                ptr0 = ptr0 + 2;
                acc = lmac8(acc, data_buf1, 8, 0x76543210, kernel_vec0, 4, 0); // kernel 4 (r1:08..15)
                acc = lmac8(acc, data_buf1, 7, 0x76543210, kernel_vec0, 3, 0); // kernel 3 (r1:07..14)

                // 3rd row
                data_buf2 = upd_w(data_buf2, 1, *(ptr1--)); // _________|r2:08++15|_________|_________
                acc = lmac8(acc, data_buf2, 8, 0x76543210, kernel_vec0, 7, 0); // kernel 7 (r2:08..15)
                data_buf2 = upd_w(data_buf2, 0, *(ptr1)); // r2:00++07|r2:08..15|_________|_________
                ptr1 = ptr1 + 2;
                acc = lmac8(acc, data_buf2, 7, 0x76543210, kernel_vec0, 6, 0); // kernel 6 (r2:07..14)
                data_buf2 = upd_w(data_buf2, 0, *(ptr1)); // r2:16++23|r2:08..15|_________|_________
                acc = lmac8(acc, data_buf2, 9, 0x76543210, kernel_vec1, 0, 0); // kernel 8 (r2:09..16)

                // Store result
                data_out = srs(acc, SRS_SHIFT);
                *(ptr_out++) = data_out;
            }
        ptr0--;
        ptr1--;

        // **************************************************************************
        // Region 7
        // **************************************************************************
        // row 1 data used twice (or we use 0)
        data_buf = upd_w(data_buf, 0, *(ptr0++));                     // r1:00++07|_________|_________|_________
        data_buf = upd_w(data_buf, 1, *(ptr0));                       // r1:00..07|r1:08++15|_________|_________
        acc = lmul8(data_buf, 7, 0x76543210, kernel_vec0, 0, 0);      // kernel 0 (r1:07..14)
        acc = lmac8(acc, data_buf, 8, 0x76543210, kernel_vec0, 1, 0); // kernel 1 (r1:08..15)
        acc = lmac8(acc, data_buf, 9, 0x66543210, kernel_vec0, 2, 0); // kernel 2 (r1:09..15)

        // 2nd row
        acc = lmac8(acc, data_buf, 7, 0x76543210, kernel_vec0, 3, 0); // kernel 3 (r1:07..14)
        acc = lmac8(acc, data_buf, 8, 0x76543210, kernel_vec0, 4, 0); // kernel 4 (r1:08..15)
        acc = lmac8(acc, data_buf, 9, 0x66543210, kernel_vec0, 5, 0); // kernel 5 (r1:09..15)

        // 3rd row
        data_buf = upd_w(data_buf, 0, *(ptr1++));                     // r2:00++07|_________|_________|_________
        data_buf = upd_w(data_buf, 1, *(ptr1));                       // r2:00..07|r2:08++15|_________|_________
        acc = lmac8(acc, data_buf, 7, 0x76543210, kernel_vec0, 6, 0); // kernel 6 (r2:07..14)
        acc = lmac8(acc, data_buf, 8, 0x76543210, kernel_vec0, 7, 0); // kernel 7 (r2:08..15)
        acc = lmac8(acc, data_buf, 9, 0x66543210, kernel_vec1, 0, 0); // kernel 8 (r2:09..15)

        // Store result
        data_out = srs(acc, SRS_SHIFT);
        *(ptr_out++) = data_out;

        // Increment row pointers to next row
        ptr0 = ptr_img_buffer;
        ptr1 = ptr_img_buffer + 1 * stride / PARALLEL_FACTOR_32b;
        ptr2 = ptr_img_buffer + 2 * stride / PARALLEL_FACTOR_32b;
    }
    // end of first row processing

    // **************************************************************************
    // Middle rows filtering, regions 4, 1, 5
    // **************************************************************************
    for (int i = 0; i < image_height - 2; i++) {
        // **********************************************************************
        // Region 4
        // **********************************************************************
        data_buf = upd_w(data_buf, 0, *(ptr0++));                     // r1:00++07|_________|_________|_________
        acc = lmul8(data_buf, 0, 0x65432100, kernel_vec0, 0, 0);      // kernel 0 (r1:00,00..06)
        acc = lmac8(acc, data_buf, 0, 0x76543210, kernel_vec0, 1, 0); // kernel 1 (r1:00..07)
        data_buf = upd_w(data_buf, 1, *(ptr0--));                     // r1:00..07|r1:08++15|_________|_________
        acc = lmac8(acc, data_buf, 1, 0x76543210, kernel_vec0, 2, 0); // kernel 2 (r1:01..08)

        // 2nd row
        data_buf = upd_w(data_buf, 0, *(ptr1++));                     // r2:00++07|_________|_________|_________
        acc = lmac8(acc, data_buf, 0, 0x65432100, kernel_vec0, 3, 0); // kernel 3 (r2:00,00..06)
        acc = lmac8(acc, data_buf, 0, 0x76543210, kernel_vec0, 4, 0); // kernel 4 (r2:00..07)
        data_buf = upd_w(data_buf, 1, *(ptr1--));                     // r2:00..07|r2:08++15|_________|_________
        acc = lmac8(acc, data_buf, 1, 0x76543210, kernel_vec0, 5, 0); // kernel 5 (r2:01..08)

        // 3rd row
        data_buf = upd_w(data_buf, 0, *(ptr2++));                     // r3:00++07|_________|_________|_________
        acc = lmac8(acc, data_buf, 0, 0x65432100, kernel_vec0, 6, 0); // kernel 6 (r3:00,00..06)
        acc = lmac8(acc, data_buf, 0, 0x76543210, kernel_vec0, 7, 0); // kernel 6 (r3:00..07)
        data_buf = upd_w(data_buf, 1, *(ptr2--));                     // r3:00..07|r3:08++15|_________|_________
        acc = lmac8(acc, data_buf, 1, 0x76543210, kernel_vec1, 0, 0); // kernel 8 (r3:01..08)

        // Store result
        data_out = srs(acc, SRS_SHIFT);
        *(ptr_out++) = data_out;

        ptr0++;
        ptr1++;
        ptr2++;
        // **********************************************************************
        // Region 1: generic case, border effect free
        // **********************************************************************
        for (int j = 0; j < image_width - 2 * PARALLEL_FACTOR_32b; j += PARALLEL_FACTOR_32b) // 16x samples per loop
            chess_prepare_for_pipelining {
                // row 1 data used twice (or we use 0)
                data_buf1 = upd_w(data_buf1, 1, *(ptr0--));               // _________|r1:08++15|_________|_________
                acc = lmul8(data_buf1, 8, 0x76543210, kernel_vec0, 1, 0); // kernel 1 (r1:08..15)
                data_buf1 = upd_w(data_buf1, 0, *(ptr0));                 // r1:00++07|r1:08..15|_________|_________
                ptr0 = ptr0 + 2;
                acc = lmac8(acc, data_buf1, 7, 0x76543210, kernel_vec0, 0, 0); // kernel 0 (r1:07..14)
                data_buf1 = upd_w(data_buf1, 0, *(ptr0)); // r1:16++23|r1:08..15|_________|_________
                acc = lmac8(acc, data_buf1, 9, 0x76543210, kernel_vec0, 2, 0); // kernel 2 (r1:09..16)

                // 2nd row
                data_buf2 = upd_w(data_buf2, 1, *(ptr1--)); // _________|r2:08++15|_________|_________
                acc = lmac8(acc, data_buf2, 8, 0x76543210, kernel_vec0, 4, 0); // kernel 4 (r1:08..15)
                data_buf2 = upd_w(data_buf2, 0, *(ptr1)); // r2:00++07|r2:08..15|_________|_________
                ptr1 = ptr1 + 2;
                acc = lmac8(acc, data_buf2, 7, 0x76543210, kernel_vec0, 3, 0); // kernel 3 (r1:07..14)
                data_buf2 = upd_w(data_buf2, 0, *(ptr1)); // r2:16++23|r2:08..15|_________|_________
                acc = lmac8(acc, data_buf2, 9, 0x76543210, kernel_vec0, 5, 0); // kernel 5 (r1:09..16)

                // 3rd row
                data_buf1 = upd_w(data_buf1, 1, *(ptr2--)); // _________|r3:08++15|_________|_________
                acc = lmac8(acc, data_buf1, 8, 0x76543210, kernel_vec0, 7, 0); // kernel 7 (r1:08..15)
                data_buf1 = upd_w(data_buf1, 0, *(ptr2)); // r3:00++07|r3:08..15|_________|_________
                ptr2 = ptr2 + 2;
                acc = lmac8(acc, data_buf1, 7, 0x76543210, kernel_vec0, 6, 0); // kernel 6 (r1:07..14)
                data_buf1 = upd_w(data_buf1, 0, *(ptr2)); // r3:16++23|r3:08..15|_________|_________
                acc = lmac8(acc, data_buf1, 9, 0x76543210, kernel_vec1, 0, 0); // kernel 8 (r1:09..16)

                // Store result
                data_out = srs(acc, SRS_SHIFT);
                *(ptr_out++) = data_out;
            }
        ptr0--;
        ptr1--;
        ptr2--;

        // **********************************************************************
        // Region 5
        // **********************************************************************
        // row 1 data used twice (or we use 0)
        data_buf = upd_w(data_buf, 0, *(ptr0++));                     // r1:00++07|_________|_________|_________
        data_buf = upd_w(data_buf, 1, *(ptr0));                       // r1:00..07|r1:08++15|_________|_________
        acc = lmul8(data_buf, 7, 0x76543210, kernel_vec0, 0, 0);      // kernel 0 (r1:07..14)
        acc = lmac8(acc, data_buf, 8, 0x76543210, kernel_vec0, 1, 0); // kernel 1 (r1:08..15)
        acc = lmac8(acc, data_buf, 9, 0x66543210, kernel_vec0, 2, 0); // kernel 2 (r1:09..15)

        // 2nd row
        data_buf = upd_w(data_buf, 0, *(ptr1++));                     // r2:00++07|_________|_________|_________
        data_buf = upd_w(data_buf, 1, *(ptr1));                       // r2:00..07|r2:08++15|_________|_________
        acc = lmac8(acc, data_buf, 7, 0x76543210, kernel_vec0, 3, 0); // kernel 3 (r2:07..14)
        acc = lmac8(acc, data_buf, 8, 0x76543210, kernel_vec0, 4, 0); // kernel 4 (r2:08..15)
        acc = lmac8(acc, data_buf, 9, 0x66543210, kernel_vec0, 5, 0); // kernel 5 (r2:09..15)

        // 3rd row
        data_buf = upd_w(data_buf, 0, *(ptr2++));                     // r3:00++07|_________|_________|_________
        data_buf = upd_w(data_buf, 1, *(ptr2));                       // r3:00..07|r3:08++15|_________|_________
        acc = lmac8(acc, data_buf, 7, 0x76543210, kernel_vec0, 6, 0); // kernel 6 (r3:07..14)
        acc = lmac8(acc, data_buf, 8, 0x76543210, kernel_vec0, 7, 0); // kernel 7 (r3:08..15)
        acc = lmac8(acc, data_buf, 9, 0x66543210, kernel_vec1, 0, 0); // kernel 8 (r3:09..15)

        // Store result
        data_out = srs(acc, SRS_SHIFT);
        *(ptr_out++) = data_out;

        // Increment row pointers to next row
        ptr0 = ptr_img_buffer + (i + 1) * stride / PARALLEL_FACTOR_32b;
        ptr1 = ptr_img_buffer + (i + 2) * stride / PARALLEL_FACTOR_32b;
        ptr2 = ptr_img_buffer + (i + 3) * stride / PARALLEL_FACTOR_32b;
    }

    // **************************************************************************
    // Last row filtering, regions 8, 3, 9
    // **************************************************************************
    {
        // **************************************************************************
        // Region 8
        // **************************************************************************
        // 1st row
        data_buf = upd_w(data_buf, 0, *(ptr0++));                     // r1:00++07|_________|_________|_________
        acc = lmul8(data_buf, 0, 0x65432100, kernel_vec0, 0, 0);      // kernel 0 (r1:00,00..06)
        acc = lmac8(acc, data_buf, 0, 0x76543210, kernel_vec0, 1, 0); // kernel 1 (r1:00..07)
        data_buf = upd_w(data_buf, 1, *(ptr0--));                     // r1:00..07|r1:08++15|_________|_________
        acc = lmac8(acc, data_buf, 1, 0x76543210, kernel_vec0, 2, 0); // kernel 2 (r1:01..08)

        // 2nd row
        data_buf = upd_w(data_buf, 0, *(ptr1++));                     // r2:00++07|_________|_________|_________
        acc = lmac8(acc, data_buf, 0, 0x65432100, kernel_vec0, 3, 0); // kernel 3 (r2:00,00..06)
        acc = lmac8(acc, data_buf, 0, 0x76543210, kernel_vec0, 4, 0); // kernel 4 (r2:00..07)
        data_buf = upd_w(data_buf, 1, *(ptr1--));                     // r2:00..07|r2:08++15|_________|_________
        acc = lmac8(acc, data_buf, 1, 0x76543210, kernel_vec0, 5, 0); // kernel 5 (r2:01..08)

        // 3rd row (uses row 2 data)
        acc = lmac8(acc, data_buf, 0, 0x65432100, kernel_vec0, 6, 0); // kernel 6 (r3:00,00..06)
        acc = lmac8(acc, data_buf, 0, 0x76543210, kernel_vec0, 7, 0); // kernel 7 (r3:00..07)
        acc = lmac8(acc, data_buf, 1, 0x76543210, kernel_vec1, 0, 0); // kernel 8 (r3:01..08)

        // Store result
        data_out = srs(acc, SRS_SHIFT);
        *(ptr_out++) = data_out;

        ptr0++;
        ptr1++;
        // **************************************************************************
        // Region 3
        // **************************************************************************
        for (int j = 0; j < image_width - 2 * PARALLEL_FACTOR_32b; j += PARALLEL_FACTOR_32b) // 8x samples per loop
            chess_prepare_for_pipelining {
                // 1st row
                data_buf1 = upd_w(data_buf1, 1, *(ptr0--));               // _________|r1:08++15|_________|_________
                acc = lmul8(data_buf1, 8, 0x76543210, kernel_vec0, 1, 0); // kernel 1 (r1:08..15)
                data_buf1 = upd_w(data_buf1, 0, *(ptr0));                 // r1:00++07|r1:08..15|_________|_________
                ptr0 = ptr0 + 2;
                acc = lmac8(acc, data_buf1, 7, 0x76543210, kernel_vec0, 0, 0); // kernel 0 (r1:07..14)
                data_buf1 = upd_w(data_buf1, 0, *(ptr0)); // r1:16++23|r1:08..15|_________|_________
                acc = lmac8(acc, data_buf1, 9, 0x76543210, kernel_vec0, 2, 0); // kernel 2 (r1:09..16)

                // 2nd row
                data_buf2 = upd_w(data_buf2, 1, *(ptr1--)); // _________|r2:08++15|_________|_________
                acc = lmac8(acc, data_buf2, 8, 0x76543210, kernel_vec0, 4, 0); // kernel 4 (r1:08..15)
                data_buf2 = upd_w(data_buf2, 0, *(ptr1)); // r2:00++07|r2:08..15|_________|_________
                ptr1 = ptr1 + 2;
                acc = lmac8(acc, data_buf2, 7, 0x76543210, kernel_vec0, 3, 0); // kernel 3 (r1:07..14)
                data_buf2 = upd_w(data_buf2, 0, *(ptr1)); // r2:16++23|r2:08..15|_________|_________
                ptr1 = ptr1 - 2;
                acc = lmac8(acc, data_buf2, 9, 0x76543210, kernel_vec0, 5, 0); // kernel 5 (r1:09..16)

                // 3rd row (uses row 2 data)
                acc = lmac8(acc, data_buf2, 9, 0x76543210, kernel_vec1, 0, 0); // kernel 8 (r1:09..16)
                data_buf2 = upd_w(data_buf2, 0, *(ptr1)); // r2:00++07|r2:08..15|_________|_________
                ptr1 = ptr1 + 2;
                acc = lmac8(acc, data_buf2, 8, 0x76543210, kernel_vec0, 7, 0); // kernel 7 (r1:08..15)
                acc = lmac8(acc, data_buf2, 7, 0x76543210, kernel_vec0, 6, 0); // kernel 6 (r1:07..14)

                // Store result
                data_out = srs(acc, SRS_SHIFT);
                *(ptr_out++) = data_out;
            }
        ptr0--;
        ptr1--;

        // **************************************************************************
        // Region 9
        // **************************************************************************
        data_buf = upd_w(data_buf, 0, *(ptr0++));                     // r1:00++07|_________|_________|_________
        data_buf = upd_w(data_buf, 1, *(ptr0));                       // r1:00..07|r1:08++15|_________|_________
        acc = lmul8(data_buf, 7, 0x76543210, kernel_vec0, 0, 0);      // kernel 0 (r1:07..14)
        acc = lmac8(acc, data_buf, 8, 0x76543210, kernel_vec0, 1, 0); // kernel 1 (r1:08..15)
        acc = lmac8(acc, data_buf, 9, 0x66543210, kernel_vec0, 2, 0); // kernel 2 (r1:09..15)

        // 2nd row
        data_buf = upd_w(data_buf, 0, *(ptr1++));                     // r2:00++07|_________|_________|_________
        data_buf = upd_w(data_buf, 1, *(ptr1));                       // r2:00..07|r2:08++15|_________|_________
        acc = lmac8(acc, data_buf, 7, 0x76543210, kernel_vec0, 3, 0); // kernel 3 (r2:07..14)
        acc = lmac8(acc, data_buf, 8, 0x76543210, kernel_vec0, 4, 0); // kernel 4 (r2:08..15)
        acc = lmac8(acc, data_buf, 9, 0x66543210, kernel_vec0, 5, 0); // kernel 5 (r2:09..15)

        // 3rd row (uses row 2 data)
        acc = lmac8(acc, data_buf, 7, 0x76543210, kernel_vec0, 6, 0); // kernel 6 (r2:07..14)
        acc = lmac8(acc, data_buf, 8, 0x76543210, kernel_vec0, 7, 0); // kernel 7 (r2:08..15)
        acc = lmac8(acc, data_buf, 9, 0x66543210, kernel_vec1, 0, 0); // kernel 8 (r2:09..15)

        // Store result
        data_out = srs(acc, SRS_SHIFT);
        *(ptr_out++) = data_out;
    }
}

} // aie
} // cv
} // xf
#endif
