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
 * @file bitonic_sort.hpp
 * @brief bitonic join function.
 *
 * This file is part of Vitis Database Library.
 */

#ifndef XF_DATABASE_BITONIC_SORT_H
#define XF_DATABASE_BITONIC_SORT_H

#include <ap_int.h>
#include <hls_stream.h>

#include "xf_database/enums.hpp"

namespace xf {
namespace database {
namespace details {

/// sort 2 number
template <typename type_t>
int bitonic_sort2(type_t a[2], type_t b[2], bool sign) {
    if (sign) {
        // ascending
        if (a[0] > a[1]) {
            b[0] = a[1];
            b[1] = a[0];
        } else {
            b[0] = a[0];
            b[1] = a[1];
        }
    } else {
        // descending
        if (a[0] > a[1]) {
            b[0] = a[0];
            b[1] = a[1];
        } else {
            b[0] = a[1];
            b[1] = a[0];
        }
    }
    return 0;
}

/// sort instance
template <typename type_t, int Number>
struct bitonic_sort_inst {
    /// sort N, N must be a power of 2
    static int sub_sort(type_t a[Number], type_t b[Number], bool sign) {
#pragma HLS INLINE
        int i;
        type_t temp1[Number / 2];
#pragma HLS ARRAY_PARTITION variable = temp1 complete dim = 1
        type_t temp2[Number / 2];
#pragma HLS ARRAY_PARTITION variable = temp2 complete dim = 1
        type_t temp3[Number / 2];
#pragma HLS ARRAY_PARTITION variable = temp3 complete dim = 1
        type_t temp4[Number / 2];
#pragma HLS ARRAY_PARTITION variable = temp4 complete dim = 1
        type_t temp5[Number];
#pragma HLS ARRAY_PARTITION variable = temp5 complete dim = 1
        type_t temp6[Number];
#pragma HLS ARRAY_PARTITION variable = temp6 complete dim = 1

    sort_loop0:
        for (i = 0; i < Number / 2; i++) {
#pragma HLS UNROLL
            temp1[i] = a[i];
            temp2[i] = a[i + Number / 2];
        }

        bitonic_sort_inst<type_t, Number / 2>::sub_sort(temp1, temp3, 1);
        bitonic_sort_inst<type_t, Number / 2>::sub_sort(temp2, temp4, 0);

    sort_loop1:
        for (i = 0; i < Number / 2; i++) {
#pragma HLS UNROLL
            temp5[i] = temp3[i];
            temp5[i + Number / 2] = temp4[i];
        }

        bitonic_sort_inst<type_t, Number / 2>::sub_merge(temp5, temp6, sign);

    sort_loop2:
        for (i = 0; i < Number; i++) {
#pragma HLS UNROLL
            b[i] = temp6[i];
        }

        return 0;
    }

    /// merge a[N] to a[2*N], N must be a power of 2
    static int sub_merge(type_t a[2 * Number], type_t b[2 * Number], bool sign) {
#pragma HLS INLINE
        int i;
        type_t temp00[Number][2];
#pragma HLS ARRAY_PARTITION variable = temp00 complete
        type_t temp01[Number][2];
#pragma HLS ARRAY_PARTITION variable = temp01 complete

        type_t temp1[Number];
#pragma HLS ARRAY_PARTITION variable = temp1 complete dim = 1
        type_t temp2[Number];
#pragma HLS ARRAY_PARTITION variable = temp2 complete dim = 1
        type_t temp3[Number];
#pragma HLS ARRAY_PARTITION variable = temp3 complete dim = 1
        type_t temp4[Number];
#pragma HLS ARRAY_PARTITION variable = temp4 complete dim = 1

    merge_loop0:
        for (i = 0; i < Number; i++) {
#pragma HLS UNROLL
            temp00[i][0] = a[i];
            temp00[i][1] = a[i + Number];
        }

    merge_loop1:
        for (i = 0; i < Number; i++) {
#pragma HLS UNROLL
            details::bitonic_sort2<type_t>(temp00[i], temp01[i], sign);
        }

    merge_loop2:
        for (i = 0; i < Number; i++) {
#pragma HLS UNROLL
            temp1[i] = temp01[i][0];
            temp2[i] = temp01[i][1];
        }

        bitonic_sort_inst<type_t, Number / 2>::sub_merge(temp1, temp3, sign);
        bitonic_sort_inst<type_t, Number / 2>::sub_merge(temp2, temp4, sign);

    merge_loop3:
        for (i = 0; i < Number; i++) {
#pragma HLS UNROLL
            b[i] = temp3[i];
            b[i + Number] = temp4[i];
        }
        return 0;
    }
};

/// Template Termination condition
template <typename type_t>
struct bitonic_sort_inst<type_t, 2> {
    /// Sort Termination condition
    static int sub_sort(type_t a[2], type_t b[2], bool sign) {
        type_t temp[2];
        temp[0] = a[0];
        temp[1] = a[1];

        if (sign) {
            // ascending
            if (temp[0] > temp[1]) {
                b[0] = temp[1];
                b[1] = temp[0];
            } else {
                b[0] = temp[0];
                b[1] = temp[1];
            }
        } else {
            // descending
            if (temp[0] > temp[1]) {
                b[0] = temp[0];
                b[1] = temp[1];
            } else {
                b[0] = temp[1];
                b[1] = temp[0];
            }
        }

        return 0;
    }

    /// Merge Termination condition
    static int sub_merge(type_t a[4], type_t b[4], bool sign) {
#pragma HLS INLINE
        type_t temp1[2];
#pragma HLS ARRAY_PARTITION variable = temp1 complete dim = 1
        type_t temp2[2];
#pragma HLS ARRAY_PARTITION variable = temp2 complete dim = 1
        type_t temp3[2];
#pragma HLS ARRAY_PARTITION variable = temp3 complete dim = 1
        type_t temp4[2];
#pragma HLS ARRAY_PARTITION variable = temp4 complete dim = 1
        type_t temp5[2];
#pragma HLS ARRAY_PARTITION variable = temp5 complete dim = 1
        type_t temp6[2];
#pragma HLS ARRAY_PARTITION variable = temp6 complete dim = 1
        type_t temp7[2];
#pragma HLS ARRAY_PARTITION variable = temp7 complete dim = 1
        type_t temp8[2];
#pragma HLS ARRAY_PARTITION variable = temp8 complete dim = 1

        temp1[0] = a[0];
        temp1[1] = a[2];
        temp2[0] = a[1];
        temp2[1] = a[3];

        details::bitonic_sort2<type_t>(temp1, temp3, sign);
        details::bitonic_sort2<type_t>(temp2, temp4, sign);

        temp5[0] = temp3[0];
        temp5[1] = temp4[0];
        temp6[0] = temp3[1];
        temp6[1] = temp4[1];

        details::bitonic_sort2<type_t>(temp5, temp7, sign);
        details::bitonic_sort2<type_t>(temp6, temp8, sign);

        b[0] = temp7[0];
        b[1] = temp7[1];
        b[2] = temp8[0];
        b[3] = temp8[1];

        return 0;
    }
};

template <typename Key_Type, int BitonicSortNumber>
void bitonic_sort_top(hls::stream<Key_Type>& kin_strm,
                      hls::stream<bool>& din_strm_end,
                      hls::stream<Key_Type>& kout_strm,
                      hls::stream<bool>& dout_strm_end,
                      bool sign) {
#pragma HLS INLINE

#ifndef __SYNTHESIS__
    if (BitonicSortNumber < 2) {
        fprintf(stderr, "Error: size of array to sort should not be less than 2.\n");
        exit(EXIT_FAILURE);
    } else if ((BitonicSortNumber & (BitonicSortNumber - 1)) != 0) {
        fprintf(stderr, "Error: size of array to sort should be power of 2.\n");
        exit(EXIT_FAILURE);
    };
#endif

    Key_Type key_strm_in[BitonicSortNumber];
    Key_Type key_strm_out[BitonicSortNumber];

    bool strm_in_end = 0;

    strm_in_end = din_strm_end.read();
    while (!strm_in_end) {
        for (int i = 0; i < BitonicSortNumber; i++) {
            key_strm_in[i] = kin_strm.read();
            strm_in_end = din_strm_end.read();
        }

        details::bitonic_sort_inst<Key_Type, BitonicSortNumber>::sub_sort(key_strm_in, key_strm_out, sign);

        for (int i = 0; i < BitonicSortNumber; i++) {
            kout_strm.write(key_strm_out[i]);
            dout_strm_end.write(0);
        }
    }
    dout_strm_end.write(1);
}

} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {

/**
 * @brief Bitonic sort is parallel algorithm for sorting.
 *
 * This algorithms can sort a large vector of data in parallel, and by cascading the sorters into a network it can
 * offer good theoretical throughput.
 *
 * Although this algorithms is suitable for FPGA acceleration, it does not work well with the row-by-row streaming
 * interface in database library. **Please consider this primitive as a demo, and only use it by deriving from this
 * code.** Alternative sorting algorithms in this library are ``insertSort`` and ``mergeSort``.
 *
 * @tparam Key_Type the input and output key type
 * @tparam BitonicSortNumber the parallel number
 *
 * @param kin_strm input key stream
 * @param kin_strm_end end flag stream for input key
 * @param kout_strm output key stream
 * @param kout_strm_end end flag stream for output key
 * @param order 1 for ascending or 0 for descending sort
 */
template <typename Key_Type, int BitonicSortNumber>
void bitonicSort(hls::stream<Key_Type>& kin_strm,
                 hls::stream<bool>& kin_strm_end,
                 hls::stream<Key_Type>& kout_strm,
                 hls::stream<bool>& kout_strm_end,
                 bool order) {
    details::bitonic_sort_top<Key_Type, BitonicSortNumber>(kin_strm, kin_strm_end, kout_strm, kout_strm_end, order);
}

} // namespace database
} // namespace xf
#endif // XF_DATABASE_BITONIC_SORT_H
