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

#include "kernel3/encode_order.hpp"

int hls_FindIndexAndRemove(int val, int* s, int len) {
    int idx = 0;
    for (int i = 0; i < len; ++i) {
        if (s[i] == val) {
            s[i] = -1;
            break;
        } else if (s[i] != -1) {
            ++idx;
        }
    }
    return idx;
}

void cnt_nz_beforeVal(int32_t order_zigzag[64], int32_t std[64], int32_t lehmer[64], uint8_t& end) {
#pragma HLS INLINE OFF

    uint8_t tmp_end = 0;
    for (int i = 0; i < 64; ++i) {
#pragma HLS PIPELINE II = 1

        uint8_t val = order_zigzag[i]; // 0,2,1,4,3,5,8,7,9,
        uint8_t cnt = 0;
        for (int j = 0; j < 64; ++j) {
#pragma HLS UNROLL
            if (j < val) {
                cnt += ((std[j] != 0) ? 1 : 0);
            }
        }
        std[val] = 0; // clear i and cnt nz before
        lehmer[i] = cnt;
        if (cnt != 0) {
            tmp_end = i;
        }
    }
    end = tmp_end;
}

void hls_EncodeCoeffOrder(hls::stream<int>& strm_order,
                          int& num_bits, // pos
                          int& num,
                          hls::stream<nbits_t>& strm_nbits,
                          hls::stream<uint16_t>& strm_bits) {
#pragma HLS INLINE OFF

    num_bits = 0;
    num = 0;

    int32_t order_zigzag[64];
    _XF_IMAGE_PRINT("start lehmercode:\n");

    for (int i = 0; i < 64; ++i) {
#pragma HLS PIPELINE II = 1
        int tmp = strm_order.read();
        order_zigzag[i] = hls_kNaturalCoeffOrderLut8[tmp];
        _XF_IMAGE_PRINT("%d,", (int)(order_zigzag[i]));
    }
    _XF_IMAGE_PRINT("\n");

    int32_t lehmer[64];
    int32_t std[64];
    for (int i = 0; i < 64; ++i) {
#pragma HLS PIPELINE II = 1
        std[i] = i;
    }

    uint8_t end = 63;
    cnt_nz_beforeVal(order_zigzag, std, lehmer, end);

    for (int i = 0; i < 64; ++i) {
        _XF_IMAGE_PRINT("%d,", (int)(lehmer[i]));
    }
    _XF_IMAGE_PRINT("\n");

    for (int32_t i = 1; i <= end; ++i) {
#pragma HLS UNROLL
        ++lehmer[i];
    }

    for (int32_t i = 0; i < 64; i += hls_kCoeffOrderCodeSpan) {
        const int32_t start = (i > 0) ? i : 1;
        const int32_t end = i + hls_kCoeffOrderCodeSpan;
        int32_t has_non_zero = 0;

        for (int32_t j = start; j < end; ++j) {
#pragma HLS UNROLL
            has_non_zero |= lehmer[j];
        }
        if (!has_non_zero) { // all zero in the span -> escape
            hls_WriteBits_strm(1, 0, num_bits, num, strm_nbits, strm_bits);
        } else {
            hls_WriteBits_strm(1, 1, num_bits, num, strm_nbits, strm_bits);

            for (int32_t j = start; j < end; ++j) {
#pragma HLS PIPELINE II = 1

                // merge
                int32_t v;
                assert(lehmer[j] <= 64);
                for (v = lehmer[j]; v >= 7; v -= 7) {
                    hls_WriteBits_strm(3, 7, num_bits, num, strm_nbits, strm_bits);
                }
                hls_WriteBits_strm(3, v, num_bits, num, strm_nbits, strm_bits);
            }
        }
    }
}
