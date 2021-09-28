/*
 * (c) Copyright 2019-2021 Xilinx, Inc. All rights reserved.
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
 *
 */
#ifndef _XFCOMPRESSION_COMPRESS_UTILS_HPP_
#define _XFCOMPRESSION_COMPRESS_UTILS_HPP_

/**
 * @file compress_utils.hpp
 * @brief Header for common data types and functions used in internal modules in library.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include <stdint.h>
#include <ap_int.h>

constexpr int maxBitsUsed(int maxVal) {
    return (32 - __builtin_clz(maxVal));
}

constexpr int getDataPortWidth(int maxVal) {
    return maxBitsUsed(maxVal);
}

template <int DWIDTH>
uint8_t countSetBits(ap_uint<DWIDTH> val) {
    uint8_t cnt = 0;
    for (uint8_t i = 0; i < DWIDTH; ++i) {
#pragma HLS UNROLL
        cnt += val.range(i, i);
    }
    return cnt;
}

template <int DATAWIDTH, int VEC_LEN>
struct __attribute__((packed)) IntVectorStream_dt {
    ap_uint<DATAWIDTH> data[VEC_LEN];
    ap_uint<maxBitsUsed(VEC_LEN)> strobe;
};

template <class DST, int VEC_LEN>
struct __attribute__((packed)) DSVectorStream_dt {
    DST data[VEC_LEN];
    ap_uint<maxBitsUsed(VEC_LEN)> strobe;
};

template <int CODE_LEN>
struct __attribute__((packed)) HuffmanCode_dt {
    ap_uint<CODE_LEN> code;                // symbol code
    ap_uint<maxBitsUsed(CODE_LEN)> bitlen; // code bit-length
};

template <int DATAWIDTH, int VEC_LEN, int STROBE_DWIDTH = maxBitsUsed(VEC_LEN)>
struct IntVectorPack {
    ap_uint<(DATAWIDTH * VEC_LEN) + STROBE_DWIDTH>& all() { return _data; }
    // access entire data vector in single register
    ap_uint<DATAWIDTH * VEC_LEN>& data() {
        return _data.range((DATAWIDTH * VEC_LEN) + STROBE_DWIDTH - 1, STROBE_DWIDTH);
    }
#if STROBE_DWIDTH > 0
    // access strobe
    ap_uint<STROBE_DWIDTH>& strobe() { return _data.range(STROBE_DWIDTH - 1, 0); }
#endif
    // access data via indexing
    ap_uint<DATAWIDTH>& operator[](unsigned const i) {
        return _data.range((DATAWIDTH * (i + 1)) + STROBE_DWIDTH - 1, (DATAWIDTH * i) + STROBE_DWIDTH);
    }

   private:
    ap_uint<(DATAWIDTH * VEC_LEN) + STROBE_DWIDTH> _data;
};

template <class DST, int VEC_LEN, int DS_DWIDTH, int STROBE_DWIDTH = maxBitsUsed(VEC_LEN)>
struct DSVectorPack {
    ap_uint<(DS_DWIDTH * VEC_LEN) + STROBE_DWIDTH>& all() { return _data; }
    // access entire data vector in single register
    ap_uint<DS_DWIDTH * VEC_LEN>& data() {
        return _data.range((DS_DWIDTH * VEC_LEN) + STROBE_DWIDTH - 1, STROBE_DWIDTH);
    }
#if STROBE_DWIDTH > 0
    // access strobe
    ap_uint<STROBE_DWIDTH>& strobe() { return _data.range(STROBE_DWIDTH - 1, 0); }
#endif
    // access data via indexing
    ap_uint<DS_DWIDTH>& operator[](unsigned const i) {
        return _data.range((DS_DWIDTH * (i + 1)) + STROBE_DWIDTH - 1, (DS_DWIDTH * i) + STROBE_DWIDTH);
    }

   private:
    ap_uint<(DS_DWIDTH * VEC_LEN) + STROBE_DWIDTH> _data;
};

template <int CODE_LEN, int STROBE_DWIDTH = maxBitsUsed(CODE_LEN)>
struct HuffmanCodePack {
    ap_uint<CODE_LEN + STROBE_DWIDTH>& all() { return _data; }
    // access entire data vector in single register
    ap_uint<CODE_LEN>& code() { return _data.range(CODE_LEN + STROBE_DWIDTH - 1, STROBE_DWIDTH); }
    // access strobe
    ap_uint<STROBE_DWIDTH>& bitlen() { return _data.range(STROBE_DWIDTH - 1, 0); }

   private:
    ap_uint<CODE_LEN + STROBE_DWIDTH> _data;
};

#endif // _XFCOMPRESSION_COMPRESS_UTILS_HPP_
