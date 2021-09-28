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
#include <common/xf_aie_utils.hpp>
#include <algorithm>

#define PARALLEL_FACTOR_16b 16 // Parallelization factor for 16b operations (16x mults)
#define SRS_SHIFT 10           // SRS shift used can be increased if input data likewise adjusted)

namespace xf {
namespace cv {
namespace aie {

/**
 * 16-bit gaussian (3x3) with border effect handling
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
__attribute__((noinline)) void gaussian_k3_border(input_window_int16* img_in,
                                                  const int16_t (&coeff)[16],
                                                  output_window_int16* img_out) {
    int16* restrict img_in_ptr = (int16*)img_in->ptr;
    int16* restrict img_out_ptr = (int16*)img_out->ptr;

    const int16_t image_width = xfcvGetTileWidth(img_in_ptr);
    const int16_t image_height = xfcvGetTileHeight(img_in_ptr);
    const int16_t stride = image_width;

    xfcvCopyMetaData(img_in_ptr, img_out_ptr);

    v16int16* restrict ptr_img_buffer = (v16int16*)xfcvGetImgDataPtr(img_in_ptr);
    v16int16* restrict ptr_img_out = (v16int16*)xfcvGetImgDataPtr(img_out_ptr);

    v32int16 data_buf1;
    v32int16 data_buf2;
    v32int16 data_buf3;
    v16int16 data_buf_c;

    v16int16 data_out;
    v16acc48 acc;

    v16int16 kernel_vec = *(v16int16*)(&coeff[0]);
    v32int16 kernel_vec_32;

    v8int16* restrict ptr0 = (v8int16*)(ptr_img_buffer);
    v8int16* restrict ptr1 = (v8int16*)(ptr_img_buffer + 1 * stride / PARALLEL_FACTOR_16b);
    v8int16* restrict ptr2 = (v8int16*)(ptr_img_buffer + 2 * stride / PARALLEL_FACTOR_16b);
    v16int16* restrict ptr_out = ptr_img_out;

    // 3x3 kernel positions
    //
    // k0 k1 0 k2 0
    // k3 k4 0 k5 0
    // k6 k7 0 k8 0

    // **************************************************************************
    // Unrolling loops over rows and columns to support the different image regions
    // **************************************************************************

    // **************************************************************************
    // First row filtering, regions 6, 2, 7
    // **************************************************************************
    {
        v16int16* restrict lptr0 = (v16int16*)ptr0;
        v16int16* restrict lptr1 = (v16int16*)ptr1;
        v16int16* restrict lptr2 = (v16int16*)ptr2;

        // **************************************************************************
        // Region 6
        // **************************************************************************
        {
            // 1st row @{
            data_buf_c = *(lptr0);
            kernel_vec_32 = upd_w(kernel_vec_32, 0, kernel_vec);
            kernel_vec_32 = upd_w(kernel_vec_32, 1, null_v16int16());
            acc = mul16(kernel_vec_32, 0, 0x70707070, 0x70707070, 0x3020, data_buf_c, 0, 0x65432100, 0xedcba987,
                        0); // k0*d[0,0-14] + 0*d[0,0-14]
            v32int16 data_buf1t;
            data_buf1t = upd_w(data_buf1t, 0, *(lptr0++)); // r1:00++15|_________
            data_buf1t = upd_w(data_buf1t, 1, *(lptr0));   // r1:00..15|r1:16++31
            acc = mac16(acc, data_buf1t, 0, 0x03020100, 0x07060504, 0x2110, kernel_vec, 1, 0, 0,
                        2); // k1*d[0-15] + k2*d[1-16]
            //@}

            // 2nd row (replica of 1st) @{
            acc = mac16(acc, kernel_vec_32, 0, 0x72727272, 0x72727272, 0x3121, data_buf_c, 0, 0x65432100, 0xedcba987,
                        0); // k3*d[0,0-14] + 0*d[0,0-14]
            acc = mac16(acc, data_buf1t, 0, 0x03020100, 0x07060504, 0x2110, kernel_vec, 6, 0, 0,
                        2); // k4*d[0-15] + k5*d[1-16]
            //@}

            // 3rd row
            data_buf_c = *(lptr1);
            acc = mac16(acc, kernel_vec_32, 0, 0x75757575, 0x75757575, 0x3020, data_buf_c, 0, 0x65432100, 0xedcba987,
                        0); // k6*d[0,0-14] + 0*d[0,0-14]
            v32int16 data_buf2t;
            data_buf2t = upd_w(data_buf2t, 0, *(lptr1++)); // r2:00++15|_________
            data_buf2t = upd_w(data_buf2t, 1, *(lptr1));   // r2:00..15|r2:16++31
            acc = mac16(acc, data_buf2t, 0, 0x03020100, 0x07060504, 0x2110, kernel_vec, 11, 0, 0,
                        2); // k7*d[0-15] +  k8*d[1-16]

            // Store result
            data_out = srs(acc, SRS_SHIFT);
            *(ptr_out++) = data_out;
        }

        // Update data pointers
        ptr0++;
        ptr1++;
        ptr2++;
        lptr0 = (v16int16*)ptr0;
        lptr1 = (v16int16*)ptr1;
        lptr2 = (v16int16*)ptr2;

        // **************************************************************************
        // Region 2
        // **************************************************************************
        // row 1 data used twice (or we use 0)
        for (int j = 0; j < image_width - 2 * PARALLEL_FACTOR_16b; j += PARALLEL_FACTOR_16b) // 8x samples per loop
            chess_prepare_for_pipelining {
                // 1st row
                v32int16 data_buf1t;
                data_buf1t = upd_w(data_buf1t, 0, *(lptr0++)); // r1:00++15|_________
                data_buf1t = upd_w(data_buf1t, 1, *(lptr0));   // r1:00..15|r1:16++31
                acc = mul16(data_buf1t, 6, 0x03020100, 0x07060504, 0x3221, kernel_vec, 0, 0, 0,
                            1); // k0*d[7-22] + k1*d[8-23]
                acc = mac16(acc, data_buf1t, 8, 0x03020100, 0x07060504, 0x3221, kernel_vec, 3, 0, 0,
                            1); // k2*d[9-24] + 0*d[10-25]

                // 2nd row (replicate row 1)
                acc = mac16(acc, data_buf1t, 6, 0x03020100, 0x07060504, 0x3221, kernel_vec, 5, 0, 0,
                            1); // k3*d[7-22] + k4*d[8-23]
                acc = mac16(acc, data_buf1t, 8, 0x03020100, 0x07060504, 0x3221, kernel_vec, 8, 0, 0,
                            1); // k5*d[9-24] + 0*d[10-25]

                // 3rd row
                v32int16 data_buf2t;
                data_buf2t = upd_w(data_buf2t, 0, *(lptr1++)); // r3:00++15|_________
                data_buf2t = upd_w(data_buf2t, 1, *(lptr1));   // r3:00..15|r3:16++31
                acc = mac16(acc, data_buf2t, 6, 0x03020100, 0x07060504, 0x3221, kernel_vec, 10, 0, 0,
                            1); // k6*d[7-22] + k7*d[8-23]
                acc = mac16(acc, data_buf2t, 8, 0x03020100, 0x07060504, 0x3221, kernel_vec, 13, 0, 0,
                            1); // k8*d[9-24] + 0*d[10-25]

                // Store result
                data_out = srs(acc, SRS_SHIFT);
                *(ptr_out++) = data_out;
            }

        // Update data pointers
        ptr0 = (v8int16*)lptr0;
        ptr1 = (v8int16*)lptr1;
        ptr2 = (v8int16*)lptr2;
        ptr0--;
        ptr1--;
        ptr2--;
        lptr0 = (v16int16*)ptr0;
        lptr1 = (v16int16*)ptr1;
        lptr2 = (v16int16*)ptr2;

        // **************************************************************************
        // Region 7
        // **************************************************************************
        {
            // 1st row
            v32int16 data_buf1t;
            data_buf1t = upd_w(data_buf1t, 0, *(lptr0++)); // r1:00++15|_________
            data_buf1t = upd_w(data_buf1t, 1, *(lptr0));   // r1:00..15|r1:16++31
            acc = mul16(data_buf1t, 14, 0x03020100, 0x07060504, 0x3221, kernel_vec, 0, 0, 0,
                        1); // k0*d[15-30] + k1*d[16-31]
            acc = mac16(acc, data_buf1t, 16, 0x03020100, 0x07060504, 0x2110, kernel_vec, 2, 0x00000000, 0x10000000,
                        1); // [15{0},k2]*d[16-31] + [15{k2},0]*d[17-32]

            // 2nd row (Repeat 1st row)
            acc = mac16(acc, data_buf1t, 14, 0x03020100, 0x07060504, 0x3221, kernel_vec, 5, 0, 0,
                        1); // k3*d[15-30] + k4*d[16-31]
            acc = mac16(acc, data_buf1t, 16, 0x03020100, 0x07060504, 0x2110, kernel_vec, 7, 0x00000000, 0x10000000,
                        1); // [15{0},k5]*d[16-31] + [15{k5},0]*d[17-32]

            // 3rd row
            v32int16 data_buf2t;
            data_buf2t = upd_w(data_buf2t, 0, *(lptr1++)); // r3:00++15|_________
            data_buf2t = upd_w(data_buf2t, 1, *(lptr1));   // r3:00..15|r3:16++31
            acc = mac16(acc, data_buf2t, 14, 0x03020100, 0x07060504, 0x3221, kernel_vec, 10, 0, 0,
                        1); // k6*d[15-30] + k7*d[16-31]
            acc = mac16(acc, data_buf2t, 16, 0x03020100, 0x07060504, 0x2110, kernel_vec, 12, 0x00000000, 0x10000000,
                        1); // [15{0},k8]*d[16-31] + [15{k8},0]*d[17-32]

            // Store result
            data_out = srs(acc, SRS_SHIFT);
            *(ptr_out++) = data_out;
        }

        // Increment row pointers to next row
        ptr0 = (v8int16*)(ptr_img_buffer);
        ptr1 = (v8int16*)(ptr_img_buffer + 1 * stride / PARALLEL_FACTOR_16b);
        ptr2 = (v8int16*)(ptr_img_buffer + 2 * stride / PARALLEL_FACTOR_16b);
    }
    // end of first row processing

    // **************************************************************************
    // Middle rows filtering, regions 4, 1, 5
    // *************************************************************************n
    for (int i = 0; i < image_height - 2; i++) {
        v16int16* restrict lptr0 = (v16int16*)ptr0;
        v16int16* restrict lptr1 = (v16int16*)ptr1;
        v16int16* restrict lptr2 = (v16int16*)ptr2;

        // **********************************************************************
        // Region 4
        // **********************************************************************
        {
            // 1st row @{
            data_buf_c = *(lptr0);
            kernel_vec_32 = upd_w(kernel_vec_32, 0, kernel_vec);
            kernel_vec_32 = upd_w(kernel_vec_32, 1, null_v16int16());
            acc = mul16(kernel_vec_32, 0, 0x70707070, 0x70707070, 0x3020, data_buf_c, 0, 0x65432100, 0xedcba987,
                        0); // k0*d[0,0-14] + 0*d[0,0-14]

            v32int16 data_buf1t;
            data_buf1t = upd_w(data_buf1t, 0, *(lptr0++)); // r1:00++15|_________
            data_buf1t = upd_w(data_buf1t, 1, *(lptr0));   // r1:00..15|r1:16++31
            acc = mac16(acc, data_buf1t, 0, 0x03020100, 0x07060504, 0x2110, kernel_vec, 1, 0, 0,
                        2); // k1*d[0-15] + k2*d[1-16]
            //@}

            // 2nd row (replica of 1st) @{
            data_buf_c = *(lptr1);
            acc = mac16(acc, kernel_vec_32, 0, 0x72727272, 0x72727272, 0x3121, data_buf_c, 0, 0x65432100, 0xedcba987,
                        0); // k3*d[0,0-14] + 0*d[0,0-14]

            v32int16 data_buf2t;
            data_buf2t = upd_w(data_buf2t, 0, *(lptr1++)); // r2:00++15|_________
            data_buf2t = upd_w(data_buf2t, 1, *(lptr1));   // r2:00..15|r2:16++31
            acc = mac16(acc, data_buf2t, 0, 0x03020100, 0x07060504, 0x2110, kernel_vec, 6, 0, 0,
                        2); // k4*d[0-15] + k5*d[1-16]

            //@}

            // 3rd row
            data_buf_c = *(lptr2);
            acc = mac16(acc, kernel_vec_32, 0, 0x75757575, 0x75757575, 0x3020, data_buf_c, 0, 0x65432100, 0xedcba987,
                        0); // k6*d[0,0-14] + 0*d[0,0-14]

            v32int16 data_buf3t;
            data_buf3t = upd_w(data_buf3t, 0, *(lptr2++)); // r2:00++15|_________
            data_buf3t = upd_w(data_buf3t, 1, *(lptr2));   // r2:00..15|r2:16++31
            acc = mac16(acc, data_buf3t, 0, 0x03020100, 0x07060504, 0x2110, kernel_vec, 11, 0, 0,
                        2); // k7*d[0-15] +  k8*d[1-16]

            // Store result
            data_out = srs(acc, SRS_SHIFT);
            *(ptr_out++) = data_out;
        }

        // Update data pointers
        ptr0++;
        ptr1++;
        ptr2++;
        lptr0 = (v16int16*)ptr0;
        lptr1 = (v16int16*)ptr1;
        lptr2 = (v16int16*)ptr2;

        // **********************************************************************
        // Region 1: generic case, border effect free
        // **********************************************************************
        for (int j = 0; j < image_width - 2 * PARALLEL_FACTOR_16b; j += PARALLEL_FACTOR_16b) // 16x samples per loop
            chess_prepare_for_pipelining {
                // 1st row
                v32int16 data_buf1t;
                data_buf1t = upd_w(data_buf1t, 0, *(lptr0++)); // r1:00++15|_________
                data_buf1t = upd_w(data_buf1t, 1, *(lptr0));   // r1:00..15|r1:16++31
                acc = mul16(data_buf1t, 6, 0x03020100, 0x07060504, 0x3221, kernel_vec, 0, 0, 0,
                            1); // k0*d[7-22] + k1*d[8-23]
                acc = mac16(acc, data_buf1t, 8, 0x03020100, 0x07060504, 0x3221, kernel_vec, 3, 0, 0,
                            1); // k2*d[9-24] + 0*d[10-25]

                // 2nd row
                v32int16 data_buf2t;
                data_buf2t = upd_w(data_buf2t, 0, *(lptr1++)); // r2:00++15|_________
                data_buf2t = upd_w(data_buf2t, 1, *(lptr1));   // r2:00..15|r2:16++31
                acc = mac16(acc, data_buf2t, 6, 0x03020100, 0x07060504, 0x3221, kernel_vec, 5, 0, 0,
                            1); // k3*d[7-22] + k4*d[8-23]
                acc = mac16(acc, data_buf2t, 8, 0x03020100, 0x07060504, 0x3221, kernel_vec, 8, 0, 0,
                            1); // k5*d[9-24] + 0*d[10-25]

                // 3rd row
                v32int16 data_buf3t;
                data_buf3t = upd_w(data_buf3t, 0, *(lptr2++)); // r3:00++15|_________
                data_buf3t = upd_w(data_buf3t, 1, *(lptr2));   // r3:00..15|r3:16++31
                acc = mac16(acc, data_buf3t, 6, 0x03020100, 0x07060504, 0x3221, kernel_vec, 10, 0, 0,
                            1); // k6*d[7-22] + k7*d[8-23]
                acc = mac16(acc, data_buf3t, 8, 0x03020100, 0x07060504, 0x3221, kernel_vec, 13, 0, 0,
                            1); // k8*d[9-24] + 0*d[10-25]

                // Store result
                data_out = srs(acc, SRS_SHIFT);
                *(ptr_out++) = data_out;
            }

        // Update data pointers
        ptr0 = (v8int16*)lptr0;
        ptr1 = (v8int16*)lptr1;
        ptr2 = (v8int16*)lptr2;
        ptr0--;
        ptr1--;
        ptr2--;
        lptr0 = (v16int16*)ptr0;
        lptr1 = (v16int16*)ptr1;
        lptr2 = (v16int16*)ptr2;

        // **********************************************************************
        // Region 5
        // **********************************************************************
        {
            // 1st row
            v32int16 data_buf1t;
            data_buf1t = upd_w(data_buf1t, 0, *(lptr0++)); // r1:00++15|_________
            data_buf1t = upd_w(data_buf1t, 1, *(lptr0));   // r1:00..15|r1:16++31
            acc = mul16(data_buf1t, 14, 0x03020100, 0x07060504, 0x3221, kernel_vec, 0, 0, 0,
                        1); // k0*d[15-30] + k1*d[16-31]
            acc = mac16(acc, data_buf1t, 16, 0x03020100, 0x07060504, 0x2110, kernel_vec, 2, 0x00000000, 0x10000000,
                        1); // [15{0},k2]*d[16-31] + [15{k2},0]*d[17-32]

            // 2nd row
            v32int16 data_buf2t;
            data_buf2t = upd_w(data_buf2t, 0, *(lptr1++)); // r3:00++15|_________
            data_buf2t = upd_w(data_buf2t, 1, *(lptr1));   // r3:00..15|r3:16++31
            acc = mac16(acc, data_buf2t, 14, 0x03020100, 0x07060504, 0x3221, kernel_vec, 5, 0, 0,
                        1); // k3*d[15-30] + k4*d[16-31]
            acc = mac16(acc, data_buf2t, 16, 0x03020100, 0x07060504, 0x2110, kernel_vec, 7, 0x00000000, 0x10000000,
                        1); // [15{0},k5]*d[16-31] + [15{k5},0]*d[17-32]

            // 3rd row
            v32int16 data_buf3t;
            data_buf3t = upd_w(data_buf3t, 0, *(lptr2++)); // r3:00++15|_________
            data_buf3t = upd_w(data_buf3t, 1, *(lptr2));   // r3:00..15|r3:16++31
            acc = mac16(acc, data_buf3t, 14, 0x03020100, 0x07060504, 0x3221, kernel_vec, 10, 0, 0,
                        1); // k6*d[15-30] + k7*d[16-31]
            acc = mac16(acc, data_buf3t, 16, 0x03020100, 0x07060504, 0x2110, kernel_vec, 12, 0x00000000, 0x10000000,
                        1); // [15{0},k8]*d[16-31] + [15{k8},0]*d[17-32]

            // Store result
            data_out = srs(acc, SRS_SHIFT);
            *(ptr_out++) = data_out;
        }

        // Increment row pointers to next row
        ptr0 = (v8int16*)(ptr_img_buffer + (i + 1) * stride / PARALLEL_FACTOR_16b);
        ptr1 = (v8int16*)(ptr_img_buffer + (i + 2) * stride / PARALLEL_FACTOR_16b);
        ptr2 = (v8int16*)(ptr_img_buffer + (i + 3) * stride / PARALLEL_FACTOR_16b);
    }

    // **************************************************************************
    // Last row filtering, regions 8, 3, 9
    // **************************************************************************
    {
        v16int16* restrict lptr0 = (v16int16*)ptr0;
        v16int16* restrict lptr1 = (v16int16*)ptr1;
        v16int16* restrict lptr2 = (v16int16*)ptr2;

        // **************************************************************************
        // Region 8
        // **************************************************************************
        {
            // 1st row @{
            data_buf_c = *(lptr0);
            kernel_vec_32 = upd_w(kernel_vec_32, 0, kernel_vec);
            kernel_vec_32 = upd_w(kernel_vec_32, 1, null_v16int16());
            acc = mul16(kernel_vec_32, 0, 0x70707070, 0x70707070, 0x3020, data_buf_c, 0, 0x65432100, 0xedcba987,
                        0); // k0*d[0,0-14] + 0*d[0,0-14]

            v32int16 data_buf1t;
            data_buf1t = upd_w(data_buf1t, 0, *(lptr0++)); // r1:00++15|_________
            data_buf1t = upd_w(data_buf1t, 1, *(lptr0));   // r1:00..15|r1:16++31
            acc = mac16(acc, data_buf1t, 0, 0x03020100, 0x07060504, 0x2110, kernel_vec, 1, 0, 0,
                        2); // k1*d[0-15] + k2*d[1-16]
            //@}

            // 2nd row
            data_buf_c = *(lptr1);
            acc = mac16(acc, kernel_vec_32, 0, 0x72727272, 0x72727272, 0x3121, data_buf_c, 0, 0x65432100, 0xedcba987,
                        0); // k3*d[0,0-14] + 0*d[0,0-14]

            v32int16 data_buf2t;
            data_buf2t = upd_w(data_buf2t, 0, *(lptr1++)); // r2:00++15|_________
            data_buf2t = upd_w(data_buf2t, 1, *(lptr1));   // r2:00..15|r2:16++31
            acc = mac16(acc, data_buf2t, 0, 0x03020100, 0x07060504, 0x2110, kernel_vec, 6, 0, 0,
                        2); // k4*d[0-15] + k5*d[1-16]

            // 3rd row (replica of 2nd) @{
            acc = mac16(acc, kernel_vec_32, 0, 0x75757575, 0x75757575, 0x3020, data_buf_c, 0, 0x65432100, 0xedcba987,
                        0); // k6*d[0,0-14] + 0*d[0,0-14]
            acc = mac16(acc, data_buf2t, 0, 0x03020100, 0x07060504, 0x2110, kernel_vec, 11, 0, 0,
                        2); // k7*d[0-15] +  k8*d[1-16]
            //@}

            // Store result
            data_out = srs(acc, SRS_SHIFT);
            *(ptr_out++) = data_out;

            // Update data pointers
            ptr0++;
            ptr1++;
            ptr2++;
            lptr0 = (v16int16*)ptr0;
            lptr1 = (v16int16*)ptr1;
            lptr2 = (v16int16*)ptr2;
        }

        // **************************************************************************
        // Region 3
        // **************************************************************************
        for (int j = 0; j < image_width - 2 * PARALLEL_FACTOR_16b; j += PARALLEL_FACTOR_16b) // 8x samples per loop
            chess_prepare_for_pipelining {
                // 1st row
                v32int16 data_buf1t;
                data_buf1t = upd_w(data_buf1t, 0, *(lptr0++)); // r1:00++15|_________
                data_buf1t = upd_w(data_buf1t, 1, *(lptr0));   // r1:00..15|r1:16++31
                acc = mul16(data_buf1t, 6, 0x03020100, 0x07060504, 0x3221, kernel_vec, 0, 0, 0,
                            1); // k0*d[7-22] + k1*d[8-23]
                acc = mac16(acc, data_buf1t, 8, 0x03020100, 0x07060504, 0x3221, kernel_vec, 3, 0, 0,
                            1); // k2*d[9-24] + 0*d[10-25]

                // 2nd row
                v32int16 data_buf2t;
                data_buf2t = upd_w(data_buf2t, 0, *(lptr1++)); // r3:00++15|_________
                data_buf2t = upd_w(data_buf2t, 1, *(lptr1));   // r3:00..15|r3:16++31
                acc = mac16(acc, data_buf2t, 6, 0x03020100, 0x07060504, 0x3221, kernel_vec, 5, 0, 0,
                            1); // k3*d[7-22] + k4*d[8-23]
                acc = mac16(acc, data_buf2t, 8, 0x03020100, 0x07060504, 0x3221, kernel_vec, 8, 0, 0,
                            1); // k5*d[9-24] + 0*d[10-25]

                // 3rd row (Repeat row 2nd)
                acc = mac16(acc, data_buf2t, 6, 0x03020100, 0x07060504, 0x3221, kernel_vec, 10, 0, 0,
                            1); // k6*d[7-22] + k7*d[8-23]
                acc = mac16(acc, data_buf2t, 8, 0x03020100, 0x07060504, 0x3221, kernel_vec, 13, 0, 0,
                            1); // k8*d[9-24] + 0*d[10-25]

                // Store result
                data_out = srs(acc, SRS_SHIFT);
                *(ptr_out++) = data_out;
            }

        // Update data pointers
        ptr0 = (v8int16*)lptr0;
        ptr1 = (v8int16*)lptr1;
        ptr2 = (v8int16*)lptr2;
        ptr0--;
        ptr1--;
        ptr2--;
        lptr0 = (v16int16*)ptr0;
        lptr1 = (v16int16*)ptr1;
        lptr2 = (v16int16*)ptr2;

        // **************************************************************************
        // Region 9
        // **************************************************************************
        {
            // 1st row
            v32int16 data_buf1t;
            data_buf1t = upd_w(data_buf1t, 0, *(lptr0++)); // r1:00++15|_________
            data_buf1t = upd_w(data_buf1t, 1, *(lptr0));   // r1:00..15|r1:16++31
            acc = mul16(data_buf1t, 14, 0x03020100, 0x07060504, 0x3221, kernel_vec, 0, 0, 0,
                        1); // k0*d[15-30] + k1*d[16-31]
            acc = mac16(acc, data_buf1t, 16, 0x03020100, 0x07060504, 0x2110, kernel_vec, 2, 0x00000000, 0x10000000,
                        1); // [15{0},k2]*d[16-31] + [15{k2},0]*d[17-32]

            // 2nd row
            v32int16 data_buf2t;
            data_buf2t = upd_w(data_buf2t, 0, *(lptr1++)); // r3:00++15|_________
            data_buf2t = upd_w(data_buf2t, 1, *(lptr1));   // r3:00..15|r3:16++31
            acc = mac16(acc, data_buf2t, 14, 0x03020100, 0x07060504, 0x3221, kernel_vec, 5, 0, 0,
                        1); // k3*d[15-30] + k4*d[16-31]
            acc = mac16(acc, data_buf2t, 16, 0x03020100, 0x07060504, 0x2110, kernel_vec, 7, 0x00000000, 0x10000000,
                        1); // [15{0},k5]*d[16-31] + [15{k5},0]*d[17-32]

            // 3rd row (Repeat 2nd row)
            acc = mac16(acc, data_buf2t, 14, 0x03020100, 0x07060504, 0x3221, kernel_vec, 10, 0, 0,
                        1); // k6*d[15-30] + k7*d[16-31]
            acc = mac16(acc, data_buf2t, 16, 0x03020100, 0x07060504, 0x2110, kernel_vec, 12, 0x00000000, 0x10000000,
                        1); // [15{0},k8]*d[16-31] + [15{k8},0]*d[17-32]

            // Store result
            data_out = srs(acc, SRS_SHIFT);
            *(ptr_out++) = data_out;
        }
    }
}

} // aie
} // cv
} // xf
